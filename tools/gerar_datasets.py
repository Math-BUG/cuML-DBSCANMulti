#!/usr/bin/env python3
"""
Gera os datasets sintéticos e a grade de parâmetros do cuML-DBSCANMulti.

Os geradores são uma adaptação fiel de cluster_ufv/datasets_sinteticos.py do repositório
Math-BUG/SSCAD-2026, para que os conjuntos de dados produzidos aqui sejam os MESMOS
daquele trabalho — famílias, sementes, normalização e heurísticas de parâmetro. Sem isso
os tempos medidos não seriam comparáveis com os números publicados.

Diferenças em relação ao original, todas de formato de saída:
  - grava .f32 binário row-major, que é o que o executável CUDA lê;
  - grava um .json com N, D, semente e a grade (eps, minPts) já calculada, para alimentar
    tanto o nosso binário quanto o baseline cuML com exatamente os mesmos parâmetros;
  - sem matplotlib: geração é headless, gráficos ficam para a etapa de análise.

Uso:
    python tools/gerar_datasets.py --dataset moons_16d --n 100000 --out-dir data
    python tools/gerar_datasets.py --suite --out-dir data          # grade do artigo
    python tools/gerar_datasets.py --listar

Saída por dataset, em --out-dir:
    <nome>_n<N>.f32          pontos, float32 row-major, sem cabeçalho
    <nome>_n<N>.labels.i32   rótulos verdadeiros, int32 (-1 = ruído)
    <nome>_n<N>.json         metadados + grade de parâmetros
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def _limitar_threads():
    """Limita as bibliotecas nativas ao número de CPUs realmente alocadas.

    Precisa rodar ANTES de importar numpy: OpenBLAS lê essas variáveis uma única vez, na
    carga da biblioteca, e ignora mudanças posteriores.

    O nó do ClusterGPU tem muito mais núcleos do que o job pede, e o OpenBLAS distribuído
    nos wheels é compilado com um NUM_THREADS máximo. Ao ver mais núcleos do que esse
    limite ele avisa ("precompiled NUM_THREADS exceeded") e, com a contenção de um nó
    compartilhado, chega a abortar por falha de segmentação dentro do kNN. Além de evitar o
    crash, respeitar --cpus-per-task é o comportamento correto num nó compartilhado.
    """
    n = os.environ.get("SLURM_CPUS_PER_TASK") or ""
    if not n.isdigit():
        n = str(min(8, os.cpu_count() or 1))
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, n)
    return int(n)


N_THREADS = _limitar_threads()

import numpy as np  # noqa: E402  (depois de _limitar_threads, de propósito)
from sklearn.datasets import make_blobs, make_circles, make_moons  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

SEED = 42

# Famílias usadas no artigo. As demais do catálogo original continuam disponíveis por
# nome, mas não entram na suíte padrão.
FAMILIAS_ARTIGO = ("dense_blobs", "heterogeneous_blobs", "moons", "rings", "spiral")

FAMILIAS_CATALOGO = FAMILIAS_ARTIGO + (
    "dense_blobs_noise",
    "blobs",
    "anisotropic_blobs",
    "varied_blobs",
    "chain_bridge",
    "grid_blobs",
)

# Famílias acrescentadas para este trabalho. Cada uma existe para estressar uma suposição
# específica da execução multiparamétrica, não para variedade visual:
#
#   nested_blobs      dois níveis de agrupamento; nenhum eps único vê os dois. É onde o
#                     Multi-EPS entrega resultado diferente, não só tempo menor.
#   power_law_blobs   densidades em lei de potência; o grau por ponto varia ~10x, o que
#                     desbalanceia os kernels que usam um bloco por linha do CSR.
#   many_blobs        número de componentes cresce com N; estressa weak_cc e a fusão de
#                     rótulos, que é o custo POR CONFIGURAÇÃO da nossa implementação.
#   uniform           sem estrutura; nnz cresce suave com o raio, aproximando O(nnz·D) de
#                     O(N²·D). É o regime em que o ganho do multi-eps encolhe.
#   filaments         dimensão intrínseca 1 em D ambiente; grau cresce ~linearmente com o
#                     raio. Regime de nnz oposto ao de uniform.
#   core_halo         núcleo denso + halo difuso; a pertinência do halo depende do raio.
FAMILIAS_ESTRESSE = (
    "nested_blobs",
    "power_law_blobs",
    "many_blobs",
    "uniform",
    "filaments",
    "core_halo",
)

FAMILIAS = FAMILIAS_CATALOGO + FAMILIAS_ESTRESSE

# Tamanhos e dimensões do protocolo experimental do artigo.
NS_ARTIGO = (4_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000)
DIMS_ARTIGO = (2, 4, 10, 12, 14, 16, 20, 32)

# Presets da suíte. `escala` é o que interessa para medir o ganho do multi em função de N:
# varre 4k -> 1M com poucas dimensões, para o número de datasets não explodir.
PRESETS = {
    "artigo":   (FAMILIAS_ARTIGO,   DIMS_ARTIGO, NS_ARTIGO),
    "escala":   (FAMILIAS_ESTRESSE + FAMILIAS_ARTIGO, (2, 16), NS_ARTIGO),
    "estresse": (FAMILIAS_ESTRESSE, (2, 16, 32), NS_ARTIGO),
}


# =============================================================================
# Geração
# =============================================================================


def normalize_minmax(X):
    """Normaliza cada coluna para [0, 1]."""
    X = np.asarray(X, dtype=np.float32)
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    denom = mx - mn
    denom[denom == 0.0] = 1.0
    return ((X - mn) / denom).astype(np.float32)


def _augmentar_2d_para_dim(X2, target_dim, rng, ruido=0.025):
    """
    Expande um dataset 2D para target_dim colunas.

    As colunas extras são combinações lineares e não lineares das duas originais, com
    ruído gaussiano somado para não ficarem perfeitamente correlacionadas. É o que
    permite testar estruturas não convexas (luas, anéis, espirais) em dimensão alta.
    """
    X2 = np.asarray(X2, dtype=np.float32)
    target_dim = int(target_dim)
    if target_dim == 2:
        return X2

    x0, x1 = X2[:, 0], X2[:, 1]
    candidatos = [
        0.70 * x0 + 0.15 * x1,
        -0.20 * x0 + 0.65 * x1,
        np.sin(x0),
        np.cos(x1),
        x0 * x1,
        x0 * x0,
        x1 * x1,
        x0 - x1,
    ]

    extras = [
        candidatos[k % len(candidatos)] + rng.normal(0.0, ruido, size=X2.shape[0])
        for k in range(target_dim - 2)
    ]
    return np.column_stack([X2] + extras).astype(np.float32)


def _dim_from_name(name):
    """'moons_10d' -> 10"""
    m = re.search(r"_(\d+)d$", name)
    if not m:
        raise ValueError(f"Dataset sem sufixo de dimensão: {name}")
    return int(m.group(1))


def _centers_for_dim(dim, count=4, scale=4.0):
    """
    Centros reprodutíveis em dim dimensões. A semente depende de (dim, count), então os
    mesmos parâmetros sempre geram os mesmos centros. Cada centro é empurrado em um eixo
    diferente para garantir separação.
    """
    rng_centers = np.random.default_rng(1000 + dim + count)
    centers = rng_centers.normal(0.0, scale, size=(count, dim)).astype(np.float32)
    for c in range(count):
        centers[c, c % dim] += scale * (1.0 if c % 2 == 0 else -1.0)
    return centers


def make_synthetic_dataset(name, n_samples, seed=SEED):
    """Catálogo de datasets sintéticos. O prefixo do nome escolhe o gerador."""
    name = str(name).lower().strip()
    n_samples = int(n_samples)
    if n_samples < 8:
        raise ValueError("n_samples precisa ser >= 8")

    rng = np.random.default_rng(seed)
    dim = _dim_from_name(name)

    if name.startswith("dense_blobs_") and not name.startswith("dense_blobs_noise_"):
        # 4 clusters compactos e bem separados
        centers = _centers_for_dim(dim, count=4, scale=4.5)
        X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.16,
                          random_state=seed)
        desc = f"Blobs densos {dim}D, baixa variância intra-cluster"

    elif name.startswith("heterogeneous_blobs_"):
        # 3 grupos com densidades bem diferentes: desafia o DBSCAN com um único eps
        counts = [int(0.42 * n_samples), int(0.33 * n_samples)]
        counts.append(n_samples - sum(counts))
        centers = _centers_for_dim(dim, count=3, scale=4.5)
        stds = [0.08, 0.28, 0.72]

        parts, labels = [], []
        for idx, (count, std) in enumerate(zip(counts, stds)):
            Xi, _ = make_blobs(n_samples=count, centers=[centers[idx]], cluster_std=std,
                               n_features=dim, random_state=seed + idx + 1)
            parts.append(Xi)
            labels.append(np.full(count, idx, dtype=np.int32))

        X = np.vstack(parts)
        y = np.concatenate(labels)
        desc = f"Blobs {dim}D com densidades diferentes"

    elif name.startswith("dense_blobs_noise_"):
        n_noise = max(1, int(0.20 * n_samples))
        n_blob = n_samples - n_noise
        centers = _centers_for_dim(dim, count=4, scale=4.2)
        X_blob, y_blob = make_blobs(n_samples=n_blob, centers=centers, cluster_std=0.18,
                                    random_state=seed)
        low = float(np.min(centers) - 3.0)
        high = float(np.max(centers) + 3.0)
        noise = rng.uniform(low=low, high=high, size=(n_noise, dim))
        X = np.vstack([X_blob, noise])
        y = np.concatenate([y_blob.astype(np.int32), np.full(n_noise, -1, dtype=np.int32)])
        desc = f"Blobs densos {dim}D com ruído rotulado como -1"

    elif name.startswith("blobs_"):
        X, y = make_blobs(n_samples=n_samples, centers=6, n_features=dim, cluster_std=0.45,
                          random_state=seed)
        desc = f"Blobs simples {dim}D, esféricos e variância moderada"

    elif name.startswith("moons_"):
        X2, y = make_moons(n_samples=n_samples, noise=0.045, random_state=seed)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Duas luas com estrutura principal 2D em {dim}D"

    elif name.startswith("rings_"):
        X2, y = make_circles(n_samples=n_samples, factor=0.38, noise=0.025, random_state=seed)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Anéis concêntricos com estrutura principal 2D em {dim}D"

    elif name.startswith("anisotropic_blobs_"):
        X2, y = make_blobs(n_samples=n_samples,
                           centers=[(-3, -1.5), (0, 2.2), (3, -1.0), (4.5, 2.5)],
                           cluster_std=0.35, random_state=seed)
        transform = np.array([[0.85, -0.55], [0.35, 1.35]], dtype=np.float32)
        X2 = X2 @ transform
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Blobs anisotrópicos com estrutura principal 2D em {dim}D"

    elif name.startswith("varied_blobs_"):
        X2, y = make_blobs(n_samples=n_samples,
                           centers=[(-4, -2), (0, 2), (3.5, -1.5), (4, 3.2)],
                           cluster_std=[0.08, 0.18, 0.42, 0.75], random_state=seed)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Blobs com variância variada em {dim}D"

    elif name.startswith("spiral_"):
        # 3 braços em coordenadas polares: clusters curvos e conectados
        arms = 3
        counts = [n_samples // arms] * arms
        counts[-1] += n_samples - sum(counts)

        xs, ys = [], []
        for arm, count in enumerate(counts):
            theta = (np.linspace(0.35, 4.2 * np.pi, count, dtype=np.float32)
                     + arm * (2.0 * np.pi / arms))
            radius = np.linspace(0.08, 1.0, count, dtype=np.float32)
            noise = rng.normal(0.0, 0.025, size=(count, 2)).astype(np.float32)
            pts = np.column_stack([radius * np.cos(theta),
                                   radius * np.sin(theta)]).astype(np.float32) + noise
            xs.append(pts)
            ys.append(np.full(count, arm, dtype=np.int32))

        X2 = np.vstack(xs)
        y = np.concatenate(ys)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Espirais com estrutura principal 2D em {dim}D"

    elif name.startswith("chain_bridge_"):
        # dois blocos ligados por uma ponte esparsa: testa se o eps "vaza" e funde clusters
        n_bridge = max(8, int(0.16 * n_samples))
        n_left = (n_samples - n_bridge) // 2
        n_right = n_samples - n_bridge - n_left

        X_left, _ = make_blobs(n_samples=n_left, centers=[(-3.0, 0.0)], cluster_std=0.22,
                               random_state=seed + 1)
        X_right, _ = make_blobs(n_samples=n_right, centers=[(3.0, 0.0)], cluster_std=0.22,
                                random_state=seed + 2)
        bridge_x = np.linspace(-2.35, 2.35, n_bridge, dtype=np.float32)
        bridge_y = rng.normal(0.0, 0.055, size=n_bridge).astype(np.float32)
        X_bridge = np.column_stack([bridge_x, bridge_y])

        X2 = np.vstack([X_left, X_bridge, X_right])
        y = np.concatenate([
            np.zeros(n_left, dtype=np.int32),
            np.full(n_bridge, 2, dtype=np.int32),
            np.ones(n_right, dtype=np.int32),
        ])
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Dois blocos ligados por ponte esparsa em {dim}D"

    elif name.startswith("grid_blobs_"):
        centers = [(x, y0) for x in (-3, 0, 3) for y0 in (-3, 0, 3)]
        X2, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.10,
                           random_state=seed)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Grade 3x3 de microblobs com estrutura principal 2D em {dim}D"

    # -----------------------------------------------------------------------
    # Famílias acrescentadas para este trabalho. Cada uma existe para estressar
    # uma suposição específica da execução multiparamétrica — ver FAMILIAS_ESTRESSE.
    # -----------------------------------------------------------------------

    elif name.startswith("nested_blobs_"):
        # Dois níveis de agrupamento: 3 super-grupos, cada um com 4 sub-blobs.
        # Nenhum eps único enxerga os dois níveis — com raio pequeno saem 12 clusters,
        # com raio grande saem 3. É o caso em que Multi-EPS entrega o que uma execução
        # escalar não entrega, e não só mais rápido.
        n_super, n_sub = 3, 4
        supers = _centers_for_dim(dim, count=n_super, scale=7.0)

        rng_off = np.random.default_rng(2000 + dim)
        centers, tam = [], []
        base = n_samples // (n_super * n_sub)
        for s in range(n_super):
            for b in range(n_sub):
                desloc = rng_off.normal(0.0, 1.0, size=dim).astype(np.float32)
                desloc *= 1.3 / max(1e-6, np.linalg.norm(desloc))
                centers.append(supers[s] + desloc)
                tam.append(base)
        tam[-1] += n_samples - sum(tam)

        X, y = make_blobs(n_samples=tam, centers=np.asarray(centers, dtype=np.float32),
                          cluster_std=0.13, random_state=seed)
        desc = (f"Blobs aninhados {dim}D: {n_super} super-grupos x {n_sub} sub-blobs "
                f"(níveis distintos por raio)")

    elif name.startswith("power_law_blobs_"):
        # Tamanhos de cluster em lei de potência: o maior tem ~10x o menor, com o mesmo
        # desvio, então a DENSIDADE — e portanto o grau de cada ponto — varia na mesma
        # proporção. Estressa o balanceamento dos kernels que usam um bloco por linha:
        # linhas com grau muito diferente caem em blocos com carga muito diferente.
        n_cluster = 8
        pesos = np.asarray([1.0 / (i + 1) ** 1.2 for i in range(n_cluster)])
        pesos /= pesos.sum()
        tam = [max(4, int(p * n_samples)) for p in pesos]
        tam[0] += n_samples - sum(tam)

        centers = _centers_for_dim(dim, count=n_cluster, scale=5.0)
        X, y = make_blobs(n_samples=tam, centers=centers, cluster_std=0.20,
                          random_state=seed)
        desc = f"Blobs {dim}D com tamanhos em lei de potência (grau muito desbalanceado)"

    elif name.startswith("many_blobs_"):
        # Muitos componentes pequenos: o número de clusters cresce com N (até 512).
        # O custo por configuração da nossa implementação é a rotulagem — weak_cc mais
        # fusão de rótulos — e ela é justamente o que cresce com o número de componentes.
        n_cluster = int(np.clip(int(np.sqrt(n_samples) / 2), 16, 512))
        lado = int(np.ceil(np.sqrt(n_cluster)))
        centers = [(float(i), float(j)) for i in range(lado) for j in range(lado)]
        centers = centers[:n_cluster]

        X2, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.11,
                           random_state=seed)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"{n_cluster} microblobs em grade, estrutura principal 2D em {dim}D"

    elif name.startswith("uniform_"):
        # Sem estrutura nenhuma. É o pior caso da anotação do CSR: o número de vizinhos
        # cresce de forma suave e previsível com o raio, sem regiões esparsas que
        # limitem nnz. Serve de referência para o regime em que O(nnz·D) se aproxima
        # de O(N²·D) e o ganho do multi-eps encolhe.
        X = rng.uniform(0.0, 1.0, size=(n_samples, dim)).astype(np.float32)
        # Sem agrupamento verdadeiro: tudo é ruído por construção.
        y = np.full(n_samples, -1, dtype=np.int32)
        desc = f"Pontos uniformes em {dim}D, sem estrutura de cluster"

    elif name.startswith("filaments_"):
        # Estruturas localmente unidimensionais: dimensão intrínseca 1 em D ambiente.
        # A vizinhança é fortemente anisotrópica, então o grau cresce ~linearmente com o
        # raio, e não como r^D. Regime de nnz oposto ao de uniform_.
        n_fil = 6
        base = n_samples // n_fil
        tam = [base] * n_fil
        tam[-1] += n_samples - sum(tam)

        rng_fil = np.random.default_rng(3000 + dim)
        partes, rotulos = [], []
        for f in range(n_fil):
            inicio = rng_fil.normal(0.0, 3.0, size=dim).astype(np.float32)
            direcao = rng_fil.normal(0.0, 1.0, size=dim).astype(np.float32)
            direcao /= max(1e-6, np.linalg.norm(direcao))

            t = np.linspace(0.0, 6.0, tam[f], dtype=np.float32)[:, None]
            pts = inicio[None, :] + t * direcao[None, :]
            pts += rng_fil.normal(0.0, 0.06, size=pts.shape).astype(np.float32)
            partes.append(pts)
            rotulos.append(np.full(tam[f], f, dtype=np.int32))

        X = np.vstack(partes)
        y = np.concatenate(rotulos)
        desc = f"{n_fil} filamentos retilíneos em {dim}D (dimensão intrínseca 1)"

    elif name.startswith("core_halo_"):
        # Cada cluster tem um núcleo denso e um halo difuso no mesmo centro. A pertinência
        # do halo depende fortemente do raio: com eps pequeno o halo vira ruído, com eps
        # grande ele entra no cluster. É onde a grade de eps muda o RESULTADO, não só o
        # tempo — e onde a monotonicidade do ruído é mais visível.
        n_cluster = 4
        centers = _centers_for_dim(dim, count=n_cluster, scale=5.0)
        por_cluster = n_samples // n_cluster

        partes, rotulos = [], []
        for c in range(n_cluster):
            n_nucleo = int(0.6 * por_cluster)
            n_halo = por_cluster - n_nucleo
            if c == n_cluster - 1:
                n_halo += n_samples - n_cluster * por_cluster

            nucleo = rng.normal(centers[c], 0.10, size=(n_nucleo, dim))
            halo = rng.normal(centers[c], 0.50, size=(n_halo, dim))
            partes.append(np.vstack([nucleo, halo]).astype(np.float32))
            rotulos.append(np.full(n_nucleo + n_halo, c, dtype=np.int32))

        X = np.vstack(partes)
        y = np.concatenate(rotulos)
        desc = f"{n_cluster} clusters {dim}D com núcleo denso e halo difuso"

    else:
        raise ValueError(f"Dataset não cadastrado: {name}")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), desc


def sample_rows(X, y, n_samples, seed=SEED):
    """Subamostra sem reposição para no máximo n_samples linhas."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = min(int(n_samples), len(X))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))
    return X[idx], y[idx]


# =============================================================================
# Sugestão automática de eps e minPts
# =============================================================================


def _indices_amostrados(n, max_pontos=5000, seed=SEED):
    """Índices da amostra, e não as linhas: quem chama precisa alinhar y com X."""
    if n <= int(max_pontos):
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=int(max_pontos), replace=False)


def _amostrar_linhas(X, max_pontos=5000, seed=SEED):
    X = np.asarray(X, dtype=np.float32)
    idx = _indices_amostrados(X.shape[0], max_pontos, seed)
    return np.ascontiguousarray(X[idx], dtype=np.float32)


def estimar_dimensao_intrinseca(X, k=20, sample_size=5000, seed=SEED):
    """
    Estimador de máxima verossimilhança de Levina-Bickel sobre os k vizinhos mais
    próximos. Evita escolher minPts só pela dimensão ambiente: em luas e anéis a
    dimensão intrínseca fica perto de 1, não de D.
    """
    Xs = _amostrar_linhas(X, sample_size, seed)
    k_eff = int(min(max(4, k), Xs.shape[0] - 1))
    if k_eff < 3:
        return float(X.shape[1])

    # n_jobs=-1 abriria um worker por núcleo do nó, não por núcleo alocado ao job.
    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", n_jobs=N_THREADS)
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    dists = dists[:, 1:]  # remove o próprio ponto (distância 0)

    rk = dists[:, -1]
    eps = 1e-12
    logs = np.log((rk[:, None] + eps) / (dists[:, :-1] + eps))
    inv_dim = np.mean(logs, axis=1)
    dim_local = 1.0 / np.maximum(inv_dim, eps)

    dim = float(np.nanmedian(dim_local))
    return float(np.clip(dim, 1.0, max(1.0, 2.0 * X.shape[1])))


def sugerir_minpts(X, sample_size=5000, minpts_max=256, seed=SEED):
    """Candidatos de minPts a partir da dimensão intrínseca e de log2(N)."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]

    dim_int = estimar_dimensao_intrinseca(X, sample_size=sample_size, seed=seed)

    base_dim = int(np.ceil(2.0 * dim_int))       # regra prática minPts ~ 2 * dimensão
    base_log = int(np.ceil(np.log2(max(n, 2))))  # cresce com o tamanho do dataset
    base = max(4, base_dim, base_log)

    candidatos = [base // 2, base, 2 * base, 4 * base]
    candidatos = sorted({int(np.clip(c, 4, minpts_max)) for c in candidatos})
    return np.asarray(candidatos, dtype=np.int32), dim_int


def sugerir_eps_por_knn(X, min_pts, quantis=(0.50, 0.70, 0.85), max_pontos=60000, seed=SEED,
                        y=None):
    """
    Candidatos de eps por quantis da curva k-distance — versão automatizada do
    "k-distance plot" clássico, sem depender de leitura visual do cotovelo.

    minPts inclui o próprio ponto nesta implementação, então usa-se n_neighbors=minPts.

    `y` são os rótulos verdadeiros, quando existem. Os pontos de ruído (rótulo < 0)
    permanecem no índice — um ponto de ruído É um vizinho legítimo e conta para o grau —
    mas a k-distance DELES não entra nos quantis.

    Por quê: em dense_blobs_noise 20% dos pontos são ruído uniforme, e num cubo de 32
    dimensões a k-distance de um ponto isolado é enorme. O quantil 0.85 caía dentro dessa
    cauda e devolvia eps 40x maior que os outros dois da grade — um raio em que cada ponto
    alcançava 81% do dataset. Não é uma configuração de DBSCAN, é um único cluster; e foi
    o que derrubou os jobs 4878 e 4879 por falta de memória para o CSR.

    Excluir a cauda é o que um humano faria lendo o k-distance plot: o cotovelo está antes
    dela, e o platô à direita é exatamente o ruído.
    """
    X = np.asarray(X, dtype=np.float32)
    idx = _indices_amostrados(X.shape[0], max_pontos, seed)
    Xs = X[idx]
    k_eff = int(min(max(2, int(min_pts)), Xs.shape[0]))

    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=N_THREADS)
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs)

    kth = dists[:, -1]
    if y is not None:
        estrutura = np.asarray(y)[idx] >= 0
        # Só filtra se sobrar amostra suficiente para os quantis significarem algo.
        if estrutura.sum() >= max(50, 0.1 * kth.size):
            kth = kth[estrutura]

    kth = np.sort(kth)
    eps_values = np.asarray([np.quantile(kth, q) for q in quantis], dtype=np.float32)
    return np.unique(np.maximum(eps_values, np.float32(1e-6))).astype(np.float32)


def sugerir_grade(X, quantis=(0.50, 0.70, 0.85), seed=SEED, y=None):
    """Grade completa (eps, minPts): 3 x 4 = 12 configurações, como no artigo."""
    minpts_values, dim_int = sugerir_minpts(X, seed=seed)
    min_pts_ref = int(minpts_values[len(minpts_values) // 2])
    eps_values = sugerir_eps_por_knn(X, min_pts_ref, quantis=quantis, seed=seed, y=y)
    return eps_values, minpts_values, min_pts_ref, dim_int


# =============================================================================
# Escrita
# =============================================================================


def gerar_e_gravar(nome, n_samples, out_dir, seed=SEED, quantis=(0.50, 0.70, 0.85)):
    """Gera um dataset, calcula a grade e grava .f32 / .labels.i32 / .json."""
    X, y, desc = make_synthetic_dataset(nome, n_samples, seed=seed)
    X, y = sample_rows(X, y, n_samples, seed=seed + n_samples)
    X = normalize_minmax(X)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.int32)

    eps_values, minpts_values, min_pts_ref, dim_int = sugerir_grade(X, quantis, seed, y=y)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{nome}_n{X.shape[0]}"

    X.tofile(f"{base}.f32")
    y.tofile(f"{base}.labels.i32")

    meta = {
        "dataset": nome,
        "descricao": desc,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "seed": int(seed),
        "dtype": "float32",
        "layout": "row-major",
        "normalizacao": "min-max por coluna para [0, 1]",
        "dimensao_intrinseca": round(float(dim_int), 4),
        "min_pts_referencia": int(min_pts_ref),
        "quantis_eps": list(quantis),
        # A grade já sai ordenada: pré-condição dos Algoritmos 2 e 3 e do nosso binário.
        "eps": [float(v) for v in eps_values],
        "min_samples": [int(v) for v in minpts_values],
        "configuration_count": int(len(eps_values) * len(minpts_values)),
        "config_order": "eps_major",
        "arquivos": {
            "points": f"{base.name}.f32",
            "labels_verdadeiros": f"{base.name}.labels.i32",
        },
    }
    Path(f"{base}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="nome com sufixo de dimensão, ex.: moons_16d")
    p.add_argument("--n", type=int, help="número de pontos")
    p.add_argument("--out-dir", default="data", help="diretório de saída (padrão: data)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--suite", action="store_true",
                   help="gera uma grade de famílias x dimensões x tamanhos (ver --preset)")
    p.add_argument("--preset", default="artigo", choices=sorted(PRESETS),
                   help="grade da suíte: artigo (5 famílias x 8 dims x 8 N), escala "
                        "(todas as famílias x 2 dims x N de 4k a 1M) ou estresse (só as "
                        "famílias novas x 3 dims x N de 4k a 1M)")
    p.add_argument("--familias", help="sobrepõe as famílias do preset")
    p.add_argument("--dims", help="sobrepõe as dimensões do preset")
    p.add_argument("--ns", help="sobrepõe os tamanhos do preset")
    p.add_argument("--dry-run", action="store_true",
                   help="lista o que seria gerado e o espaço em disco, sem gerar")
    p.add_argument("--listar", action="store_true", help="lista as famílias disponíveis")
    p.add_argument("--eps-quantis", default="0.50,0.70,0.85",
                   help="quantis da curva k-distância que viram os candidatos de eps. O "
                        "padrão é estreito de propósito (três valores dentro de ~20%%), o "
                        "que dá agrupamentos parecidos entre si. Para uma grade que "
                        "realmente varra a faixa, algo como 0.2,0.4,0.6,0.8 separa mais.")
    args = p.parse_args()

    quantis = tuple(float(q) for q in args.eps_quantis.split(",") if q.strip())
    if not quantis or any(not 0.0 < q < 1.0 for q in quantis):
        p.error("--eps-quantis deve ser uma lista de valores em (0, 1)")

    if args.listar:
        print("famílias do artigo (suíte padrão):")
        for f in FAMILIAS_ARTIGO:
            print(f"  {f}")
        print("demais famílias do catálogo original:")
        for f in FAMILIAS_CATALOGO:
            if f not in FAMILIAS_ARTIGO:
                print(f"  {f}")
        print("famílias acrescentadas para estressar a execução multiparamétrica:")
        for f in FAMILIAS_ESTRESSE:
            print(f"  {f}")
        print("\npresets de --suite:")
        for nome, (fam, dims, ns) in sorted(PRESETS.items()):
            print(f"  {nome:<9} {len(fam)} famílias x {len(dims)} dims x {len(ns)} N "
                  f"= {len(fam) * len(dims) * len(ns)} datasets"
                  f"  (N de {min(ns):,} a {max(ns):,})".replace(",", "."))
        print("\nuse o sufixo de dimensão no nome, ex.: moons_16d")
        return 0

    if args.suite or args.dry_run:
        fam_preset, dims_preset, ns_preset = PRESETS[args.preset]
        familias = ([f.strip() for f in args.familias.split(",") if f.strip()]
                    if args.familias else list(fam_preset))
        dims = ([int(d) for d in args.dims.split(",") if d.strip()]
                if args.dims else list(dims_preset))
        ns = ([int(n) for n in args.ns.split(",") if n.strip()]
              if args.ns else list(ns_preset))

        desconhecidas = [f for f in familias if f not in FAMILIAS]
        if desconhecidas:
            p.error(f"famílias não cadastradas: {', '.join(desconhecidas)}")

        # Os pontos dominam o disco: N*D float32, mais N int32 de rótulos.
        bytes_totais = sum(n * d * 4 + n * 4 for _ in familias for d in dims for n in ns)
        planejados = len(familias) * len(dims) * len(ns)
        print(f"preset {args.preset}: {len(familias)} famílias x {len(dims)} dims x "
              f"{len(ns)} tamanhos = {planejados} datasets, "
              f"~{bytes_totais / 1e9:.1f} GB em {args.out_dir}")
        print(f"N de {min(ns)} a {max(ns)}; dims {dims}")

        if args.dry_run:
            for familia in familias:
                for d in dims:
                    for n in ns:
                        print(f"  {familia}_{d}d  N={n}")
            return 0

        total = 0
        for familia in familias:
            for d in dims:
                for n in ns:
                    meta = gerar_e_gravar(f"{familia}_{d}d", n, args.out_dir, args.seed,
                                          quantis)
                    total += 1
                    print(f"[{total}/{planejados}] {meta['dataset']:>28}  "
                          f"N={meta['n']:>7}  D={meta['d']:>2}  "
                          f"eps={[round(e, 4) for e in meta['eps']]}  "
                          f"minPts={meta['min_samples']}", flush=True)
        print(f"\n{total} datasets em {args.out_dir}")
        return 0

    if not args.dataset or not args.n:
        p.error("informe --dataset e --n, ou use --suite")

    meta = gerar_e_gravar(args.dataset, args.n, args.out_dir, args.seed, quantis)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
