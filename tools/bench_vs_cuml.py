"""Compara o binário multiparamétrico com chamadas sequenciais do cuml.cluster.DBSCAN.

É a medição que decide se a abordagem se sustenta. O ganho teórico vem de compartilhar a
distância par-a-par entre as k*l configurações, mas o nosso kernel de vizinhança é escrito
à mão, enquanto o cuML usa o `epsilon_neighborhood::compute` do cuVS, que é ajustado. Se o
nosso custa T vezes mais por par, o Multi-EPS só ganha quando k > T. O modo --imposto mede
esse T diretamente, rodando uma configuração de cada vez dos dois lados.

Protocolo (as duas metades medem a mesma coisa, que é a única forma de a comparação valer):
  - só o ajuste é cronometrado, com os dados já residentes na GPU; leitura de arquivo,
    cópia H2D e devolução dos rótulos ficam de fora dos dois lados;
  - cudaEvent nos dois casos (do lado do Python, cupy.cuda.Event é o mesmo mecanismo);
  - --warmup execuções descartadas antes de medir, porque a primeira chamada do processo
    paga o carregamento do módulo CUDA;
  - mediana de --repeat execuções, não média: num nó compartilhado um pico de outro
    processo desloca a média e não mexe na mediana;
  - `calc_core_sample_indices=False` no cuML, porque a nossa implementação também não
    calcula os índices de core points — cobrar dele um trabalho que não fazemos inflaria o
    ganho;
  - quando --max-mbytes-per-batch > 0, o mesmo limite explícito é passado aos dois lados;
    com 0, cada implementação usa sua própria política automática e o JSON marca o
    protocolo como não controlado (não se alega igualdade de orçamento nesse caso).

Uso:
    python tools/bench_vs_cuml.py --meta data/moons_16d_n100000.json --validar
    python tools/bench_vs_cuml.py --meta data/moons_16d_n100000.json --imposto
    python tools/bench_vs_cuml.py --input X.f32 --n 100000 --d 16 \
        --eps 0.25,0.35,0.5 --min-samples 5,10,20 --repeat 5

A última linha do stdout é o JSON com todos os números; --out grava o mesmo em arquivo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    # Direct execution (python tools/bench_vs_cuml.py).
    from dbscan_validation import (
        OracleLimitError,
        build_epsilon_graph,
        canonicalize_labels,
        compare_semantically,
        partition_metrics,
        partition_only_validation,
        write_failure_artifact,
    )
except ImportError:  # pragma: no cover - imported as tools.bench_vs_cuml
    from tools.dbscan_validation import (
        OracleLimitError,
        build_epsilon_graph,
        canonicalize_labels,
        compare_semantically,
        partition_metrics,
        partition_only_validation,
        write_failure_artifact,
    )

try:
    from gerar_datasets import DATASET_PROTOCOL, validar_meta_protocolo
except ImportError:  # pragma: no cover - imported as tools.bench_vs_cuml
    from tools.gerar_datasets import DATASET_PROTOCOL, validar_meta_protocolo


# =============================================================================
# Execução do nosso binário
# =============================================================================


_UNIDENTIFIED_BUILD_VALUES = {"", "unknown", "none", "null", "n/a", "unidentified"}


def sha256_arquivo(path):
    """Calcula SHA-256 por streaming, sem carregar artefatos grandes na memória."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registro_pacote(*nomes):
    """Retorna distribuicao/versao instalada sem importar extensoes GPU."""

    for nome in nomes:
        try:
            return {
                "distribution": nome,
                "version": importlib.metadata.version(nome),
            }
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def coletar_ambiente(runtime_cuda, *, gpu, cuml_version, cupy_version):
    """Monta proveniencia serializavel do host, Python, RAPIDS, Slurm e CUDA."""

    cuda = runtime_cuda if isinstance(runtime_cuda, dict) else {}
    pacotes = {
        "numpy": _registro_pacote("numpy"),
        "scikit_learn": _registro_pacote("scikit-learn"),
        "cuml": _registro_pacote("cuml-cu12", "cuml-cu13", "cuml"),
        "cupy": _registro_pacote("cupy-cuda12x", "cupy-cuda13x", "cupy"),
        "libraft": _registro_pacote("libraft-cu12", "libraft-cu13", "libraft"),
        "librmm": _registro_pacote("librmm-cu12", "librmm-cu13", "librmm"),
        "libcuvs": _registro_pacote("libcuvs-cu12", "libcuvs-cu13", "libcuvs"),
    }
    if isinstance(gpu, bytes):
        gpu = gpu.decode(errors="replace")
    elif gpu is not None:
        gpu = str(gpu)

    return {
        # Chaves antigas preservadas para consumidores existentes.
        "gpu": gpu,
        "cuml": str(cuml_version),
        "cupy": str(cupy_version),
        "hostname": platform.node() or None,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": (pacotes["numpy"] or {}).get("version", np.__version__),
        "sklearn": (pacotes["scikit_learn"] or {}).get("version"),
        "libraft": (pacotes["libraft"] or {}).get("version"),
        "librmm": (pacotes["librmm"] or {}).get("version"),
        "libcuvs": (pacotes["libcuvs"] or {}).get("version"),
        "packages": pacotes,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_name": os.environ.get("SLURMD_NODENAME"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        # Copiados do JSON do C++, portanto correspondem ao processo medido.
        "cuda_runtime_version": cuda.get("runtime_version"),
        "cuda_driver_version": cuda.get("driver_version"),
    }


def validar_hash_input_meta(hash_real, hash_esperado):
    """Confere a ligação criptográfica entre o metadata e o points.f32."""

    if hash_esperado is None:
        return
    if hash_real.lower() != hash_esperado.lower():
        raise SystemExit(
            "erro: SHA-256 do input diverge de meta.sha256.points: "
            f"real={hash_real.lower()} esperado={hash_esperado.lower()}"
        )


def build_possui_identidade(build):
    """True para uma revisão Git ou hash de árvore-fonte acompanhado de build ID."""

    if not isinstance(build, dict):
        return False
    git_sha = build.get("git_sha")
    build_id = build.get("build_id")
    return bool(
        isinstance(git_sha, str)
        and git_sha.lower() not in _UNIDENTIFIED_BUILD_VALUES
        and re.fullmatch(r"[0-9a-fA-F]{7,64}", git_sha)
        and isinstance(build_id, str)
        and build_id.lower() not in _UNIDENTIFIED_BUILD_VALUES
    )


def _eps_runtime_equivalente(recebido, solicitado):
    try:
        recebido_array = np.asarray(recebido, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    solicitado_array = np.asarray(
        [np.float32(float(f"{valor:.10g}")) for valor in solicitado], dtype=np.float64
    )
    return bool(
        recebido_array.shape == solicitado_array.shape
        and np.isfinite(recebido_array).all()
        # O C++ serializa floats com a precisão padrão do ostream (6 dígitos). A
        # comparação continua estrita o bastante para detectar outra grade, sem rejeitar
        # apenas a representação textual arredondada do mesmo float32.
        and np.allclose(recebido_array, solicitado_array, rtol=5e-6, atol=1e-12)
    )


def validar_contrato_runtime(
    resultado,
    *,
    backend,
    n,
    d,
    eps,
    min_samples,
    index="auto",
    permitir_binario_legado=False,
):
    """Rejeita silenciosamente impossível: binário/backend/grade diferentes do pedido."""

    erros = []
    esperado_configuracoes = len(eps) * len(min_samples)
    if resultado.get("backend") != backend:
        erros.append(
            f"backend efetivo {resultado.get('backend')!r}; solicitado {backend!r}"
        )
    runtime_index = resultado.get("index")
    if index in {"int32", "int64"} and runtime_index != index:
        erros.append(f"index efetivo {runtime_index!r}; solicitado {index!r}")
    elif index == "auto" and runtime_index not in {"int32", "int64"}:
        erros.append(f"index efetivo inválido para auto: {runtime_index!r}")
    if resultado.get("configuration_count") != esperado_configuracoes:
        erros.append(
            f"configuration_count={resultado.get('configuration_count')!r}; "
            f"esperado {esperado_configuracoes}"
        )
    if resultado.get("eps_count") != len(eps):
        erros.append(f"eps_count={resultado.get('eps_count')!r}; esperado {len(eps)}")
    if resultado.get("min_samples_count") != len(min_samples):
        erros.append(
            f"min_samples_count={resultado.get('min_samples_count')!r}; "
            f"esperado {len(min_samples)}"
        )
    if not _eps_runtime_equivalente(resultado.get("eps"), eps):
        erros.append(f"eps efetivo {resultado.get('eps')!r}; solicitado {list(eps)!r}")
    raw_min_samples = resultado.get("min_samples")
    min_samples_runtime = (
        raw_min_samples
        if isinstance(raw_min_samples, list)
        and all(type(value) is int for value in raw_min_samples)
        else None
    )
    if min_samples_runtime != [int(value) for value in min_samples]:
        erros.append(
            f"min_samples efetivo {resultado.get('min_samples')!r}; "
            f"solicitado {[int(value) for value in min_samples]!r}"
        )
    if resultado.get("config_order") != "eps_major":
        erros.append(f"config_order={resultado.get('config_order')!r}; esperado 'eps_major'")
    if resultado.get("n") != int(n) or resultado.get("d") != int(d):
        erros.append(
            f"shape runtime N={resultado.get('n')!r}, D={resultado.get('d')!r}; "
            f"solicitado N={int(n)}, D={int(d)}"
        )

    build = resultado.get("build")
    configured_backend = build.get("configured_backend") if isinstance(build, dict) else None
    compiled_backends = build.get("compiled_backends") if isinstance(build, dict) else None
    if not permitir_binario_legado and not build_possui_identidade(build):
        erros.append(
            "revisão de fonte/build_id ausentes ou não identificados; recompile com o "
            "Makefile atual ou use --permitir-binario-legado explicitamente"
        )
    configured_backend_unidentified = (
        configured_backend is None
        or (
            isinstance(configured_backend, str)
            and configured_backend.lower() in _UNIDENTIFIED_BUILD_VALUES
        )
    )
    if configured_backend_unidentified:
        if not permitir_binario_legado:
            erros.append("build.configured_backend ausente ou não identificado")
    elif configured_backend not in {"codes", "cuvs"}:
        erros.append(f"build.configured_backend inválido: {configured_backend!r}")
    elif backend == "cuvs" and configured_backend != "cuvs":
        erros.append(
            "runtime alegou backend cuvs, mas build.configured_backend="
            f"{configured_backend!r}"
        )

    compiled_unidentified = compiled_backends is None or compiled_backends == "unknown"
    if compiled_unidentified:
        if not permitir_binario_legado:
            erros.append("build.compiled_backends ausente ou não identificado")
    elif not isinstance(compiled_backends, list):
        erros.append("build.compiled_backends deve ser uma lista")
    else:
        known_backends = {"codes", "cuvs"}
        compiled_values_valid = all(
            type(item) is str and item in known_backends for item in compiled_backends
        )
        if not compiled_values_valid:
            erros.append(
                f"build.compiled_backends contém valor desconhecido: {compiled_backends!r}"
            )
        elif len(compiled_backends) != len(set(compiled_backends)):
            erros.append(f"build.compiled_backends contém duplicatas: {compiled_backends!r}")
        if backend not in compiled_backends:
            erros.append(
                f"backend efetivo {backend!r} ausente de build.compiled_backends="
                f"{compiled_backends!r}"
            )
        if configured_backend in {"codes", "cuvs"} and configured_backend not in compiled_backends:
            erros.append(
                f"build.configured_backend={configured_backend!r} ausente de "
                f"compiled_backends={compiled_backends!r}"
            )
        if configured_backend == "codes" and "cuvs" in compiled_backends:
            erros.append("build configurado como codes não pode declarar cuvs compilado")

    if erros:
        raise SystemExit("erro: contrato JSON do binário inválido:\n  - " + "\n  - ".join(erros))
    return resultado


def ler_labels_config_major(path, n, configuration_count):
    """Lê labels int32 com diagnóstico explícito antes do reshape config-major."""

    rotulos_brutos = np.fromfile(path, dtype=np.int32)
    esperado = int(configuration_count) * int(n)
    if rotulos_brutos.size != esperado:
        raise SystemExit(
            f"erro: '{path}' contém {rotulos_brutos.size} labels int32; esperado {esperado} "
            f"({int(configuration_count)} configurações x N={int(n)})"
        )
    return rotulos_brutos.reshape(int(configuration_count), int(n))


def rodar_binario(binario, input_path, n, d, eps, min_samples, repeat, warmup,
                   max_mbytes, backend, index, neigh_per_row, saida_labels=None,
                  solo=False, permitir_binario_legado=False):
    """Roda o executável CUDA e devolve (json_resultado, rotulos ou None).

    Os rótulos saem em ordem config-major: bloco de n int32 por configuração, com as
    configurações em ordem eps-major (config = e * l + m).
    """
    cmd = [
        str(binario),
        "--input", str(input_path),
        "--n", str(n),
        "--d", str(d),
        "--eps", ",".join(f"{v:.10g}" for v in eps),
        "--min-samples", ",".join(str(int(v)) for v in min_samples),
        "--repeat", str(repeat),
        "--warmup", str(warmup),
        "--backend", backend,
        "--index", index,
        "--json",
    ]
    if neigh_per_row:
        cmd += ["--neigh-per-row", str(neigh_per_row)]
    if solo:
        cmd += ["--solo"]
    if max_mbytes:
        cmd += ["--max-mbytes-per-batch", str(max_mbytes)]
    if saida_labels is not None:
        cmd += ["--output", str(saida_labels)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"erro: {binario} saiu com código {proc.returncode}")

    if proc.stderr:
        # O runner CUDA usa stderr para avisos de lote, allocator e diagnosticos. Mesmo
        # com exit=0 isso faz parte da evidencia operacional e precisa chegar ao log Slurm.
        sys.stderr.write(proc.stderr)

    linhas = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not linhas:
        raise SystemExit(f"erro: {binario} não produziu saída JSON")
    resultado = json.loads(linhas[-1])
    validar_contrato_runtime(
        resultado,
        backend=backend,
        n=n,
        d=d,
        eps=eps,
        min_samples=min_samples,
        index=index,
        permitir_binario_legado=permitir_binario_legado,
    )

    rotulos = None
    if saida_labels is not None:
        n_cfg = len(eps) * len(min_samples)
        rotulos = ler_labels_config_major(saida_labels, n, n_cfg)
    return resultado, rotulos


# =============================================================================
# Baseline cuML
# =============================================================================


CUML_BASELINE = {
    "implementation": "cuml.cluster.DBSCAN",
    "algorithm": "brute",
    "metric": "euclidean",
    "calc_core_sample_indices": False,
}


def medir_cuml(X_gpu, eps, min_samples, repeat, warmup, max_mbytes, guardar_rotulos):
    """Cronometra uma configuração do cuml.cluster.DBSCAN com os dados já na GPU."""
    import cupy as cp
    from cuml.cluster import DBSCAN

    kwargs = dict(
        eps=float(eps),
        min_samples=int(min_samples),
        algorithm=CUML_BASELINE["algorithm"],
        metric=CUML_BASELINE["metric"],
        # Não calculamos os índices de core points; cobrá-los do baseline inflaria o ganho.
        calc_core_sample_indices=CUML_BASELINE["calc_core_sample_indices"],
        output_type="cupy",
    )
    if max_mbytes:
        kwargs["max_mbytes_per_batch"] = int(max_mbytes)

    modelo = DBSCAN(**kwargs)

    for _ in range(warmup):
        modelo.fit(X_gpu)
    cp.cuda.Stream.null.synchronize()

    tempos = []
    inicio, fim = cp.cuda.Event(), cp.cuda.Event()
    for _ in range(repeat):
        inicio.record()
        modelo.fit(X_gpu)
        fim.record()
        fim.synchronize()
        tempos.append(float(cp.cuda.get_elapsed_time(inicio, fim)))

    rotulos = None
    if guardar_rotulos:
        rotulos = cp.asnumpy(modelo.labels_).astype(np.int32, copy=False)
    return tempos, rotulos


# =============================================================================
# Validação
# =============================================================================


def canonizar(rotulos):
    """Renumera os clusters pela ordem de primeira aparição, preservando -1 (ruído).

    Rótulos de DBSCAN são invariantes a permutação: duas execuções corretas podem numerar
    os mesmos grupos de formas diferentes. Canonizando, a igualdade exata das partições
    vira uma comparação direta de vetores.
    """
    return canonicalize_labels(rotulos)


def comparar(nossos, deles):
    """ARI, concordância de ruído e igualdade exata da partição."""
    return partition_metrics(nossos, deles)


def validar_cientificamente(
    pontos,
    nossos,
    deles,
    eps,
    min_samples,
    *,
    modo="auto",
    max_n=5000,
    cache_grafos=None,
):
    """Executa o oráculo semântico independente conforme a política selecionada.

    ``auto`` verifica semanticamente partições divergentes, ``sempre`` verifica todas as
    configurações e ``nunca`` aceita somente igualdade particional canônica. Uma divergência
    que excede ``max_n`` é rejeitada, pois não pode ser atribuída com segurança a uma borda.
    """

    if modo not in {"auto", "sempre", "nunca"}:
        raise ValueError(f"modo de oráculo desconhecido: {modo}")
    metricas = partition_metrics(nossos, deles)
    executar = modo == "sempre" or (modo == "auto" and not metricas["particao_identica"])
    if not executar:
        motivo = (
            "partição canônica idêntica; use --oraculo-semantico sempre para "
            "validação independente completa"
            if modo == "auto"
            else "oráculo desativado por --oraculo-semantico nunca"
        )
        return partition_only_validation(nossos, deles, reason=motivo)

    cache_grafos = cache_grafos if cache_grafos is not None else {}
    try:
        if float(eps) not in cache_grafos:
            cache_grafos[float(eps)] = build_epsilon_graph(
                pontos, float(eps), max_n=int(max_n)
            )
        estrutura = cache_grafos[float(eps)].structure(int(min_samples))
    except OracleLimitError as error:
        resultado = partition_only_validation(nossos, deles, reason=str(error))
        # ``sempre`` é um gate científico, não uma solicitação best-effort.
        if modo == "sempre":
            resultado["valida"] = False
            resultado["status"] = "oraculo_obrigatorio_nao_executado"
        return resultado

    return compare_semantically(nossos, deles, estrutura)


# Divergência de partição que o oráculo semântico não pôde arbitrar. Só acontece acima de
# --oraculo-max-n, onde o oráculo exato é O(N²) e inviável: a campanha de desempenho vive
# inteira nessa faixa.
STATUS_SEM_ARBITRO = "divergencia_sem_oraculo"


def validacao_exit_code(por_config, tolerar_sem_oraculo=False):
    """
    Código não zero quando pelo menos uma validação solicitada falha.

    Com `tolerar_sem_oraculo`, uma divergência que o oráculo NÃO pôde arbitrar deixa de ser
    fatal — ela é contada e o artefato preservado, mas o processo segue. Serve à campanha de
    desempenho, que roda em N onde o oráculo não cabe e por isso morria no primeiro empate
    de fronteira: ~1,5% das configurações divergem assim, com ARI > 0,9999 e mesma contagem
    de clusters.

    O que continua fatal em qualquer modo: divergência que o oráculo JULGOU incorreta, e
    rótulo malformado. Tolerar significa "ninguém arbitrou ainda", nunca "está errado e
    seguimos assim" — a arbitragem sai depois, offline, com
    tools/validate_dbscan_matrix.py --artifact.
    """
    for c in por_config:
        validacao = c.get("validacao", {})
        if validacao.get("valida", False):
            continue
        if tolerar_sem_oraculo and validacao.get("status") == STATUS_SEM_ARBITRO:
            continue
        return 2
    return 0


# =============================================================================
# CLI
# =============================================================================


def mediana(valores):
    return float(np.median(np.asarray(valores, dtype=np.float64)))


def _csv_floats(raw, flag="--eps"):
    items = raw.split(",")
    if not items or any(not item.strip() for item in items):
        raise SystemExit(f"erro: {flag} contém item vazio")
    try:
        values = [float(item) for item in items]
    except ValueError as error:
        raise SystemExit(f"erro: {flag} contém número inválido: {error}") from error
    if any(not np.isfinite(value) for value in values):
        raise SystemExit(f"erro: {flag} contém NaN ou infinito")
    return values


def _csv_ints(raw, flag="--min-samples"):
    items = raw.split(",")
    if not items or any(not item.strip() for item in items):
        raise SystemExit(f"erro: {flag} contém item vazio")
    try:
        return [int(item) for item in items]
    except ValueError as error:
        raise SystemExit(f"erro: {flag} contém inteiro inválido: {error}") from error


def carregar_config(args):
    """Resolve entrada e grade a partir do --meta do gerador ou dos argumentos soltos."""
    meta_points_sha256 = None
    if args.meta:
        # O posto amostral do kNN usado para sugerir eps foi corrigido: metadata anterior
        # infla o raio para N > 60 mil, e uma campanha que misture os dois protocolos
        # compara grades diferentes sem que nada no resultado denuncie isso. O gerador já
        # sabia rejeitar metadata legado; o benchmark não chamava, e a garantia dependia de
        # apontar o --meta para o diretório certo.
        try:
            meta = validar_meta_protocolo(args.meta)
        except ValueError as erro:
            raise SystemExit(
                "\n".join(
                    (
                        f"erro: '{args.meta}' não pertence ao protocolo {DATASET_PROTOCOL!r}.",
                        f"  {erro}",
                        "  Regenere com tools/gerar_datasets.py em um diretório novo; os",
                        "  datasets do protocolo anterior seguem válidos para os resultados",
                        "  históricos, mas não podem entrar na mesma campanha.",
                    )
                )
            ) from erro
        base = Path(args.meta).parent
        input_path = base / meta["arquivos"]["points"]
        n, d = int(meta["n"]), int(meta["d"])
        eps = [float(v) for v in meta["eps"]]
        min_samples = [int(v) for v in meta["min_samples"]]
        nome = meta["dataset"]
        meta_sha = meta.get("sha256")
        if isinstance(meta_sha, dict):
            meta_points_sha256 = meta_sha.get("points")
        if meta_points_sha256 is not None and (
            not isinstance(meta_points_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", meta_points_sha256) is None
        ):
            raise SystemExit("erro: meta.sha256.points não é um SHA-256 válido")
        if meta_points_sha256 is None and not args.permitir_binario_legado:
            raise SystemExit(
                "erro: meta não contém sha256.points; gere-o novamente ou use "
                "--permitir-binario-legado explicitamente"
            )
    else:
        if not (args.input and args.n and args.d and args.eps and args.min_samples):
            raise SystemExit("erro: informe --meta, ou --input/--n/--d/--eps/--min-samples")
        input_path = Path(args.input)
        n, d = int(args.n), int(args.d)
        eps = _csv_floats(args.eps)
        min_samples = _csv_ints(args.min_samples)
        nome = input_path.stem

    if args.eps and args.meta:
        eps = _csv_floats(args.eps)
    if args.min_samples and args.meta:
        min_samples = _csv_ints(args.min_samples)

    if n <= 0 or d <= 0:
        raise SystemExit("erro: N e D devem ser positivos")
    if not eps or any(not np.isfinite(value) or value <= 0 for value in eps):
        raise SystemExit("erro: eps deve conter somente valores finitos e positivos")
    if not min_samples or any(value <= 0 for value in min_samples):
        raise SystemExit("erro: min-samples deve conter somente inteiros positivos")
    eps = sorted(set(eps))
    min_samples = sorted(set(min_samples))
    return nome, input_path, n, d, eps, min_samples, meta_points_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", help="JSON gerado por tools/gerar_datasets.py")
    p.add_argument("--input", help="arquivo .f32 (alternativa a --meta)")
    p.add_argument("--n", type=int)
    p.add_argument("--d", type=int)
    p.add_argument("--eps", help="lista separada por vírgula; sobrepõe a grade do --meta")
    p.add_argument("--min-samples", help="lista separada por vírgula; sobrepõe a do --meta")
    p.add_argument("--binario", default="build/dbscan_multi")
    p.add_argument(
        "--permitir-binario-legado",
        action="store_true",
        help=(
            "permite JSON sem build.git_sha/build_id e meta sem sha256.points; reduz a "
            "rastreabilidade e nunca desativa backend/grade nem um hash presente"
        ),
    )
    p.add_argument("--backend", default="cuvs", choices=("cuvs", "codes"),
                   help="cuvs (padrão) usa a mesma busca de vizinhança do cuML")
    p.add_argument("--index", default="auto", choices=("auto", "int32", "int64"),
                   help="tipo de índice do nosso binário. O cuml.cluster.DBSCAN usa int32 "
                        "e não expõe escolha, então int64 é uma vantagem que ele não tem — "
                        "veja ganho_multi_puro para separar os dois efeitos")
    p.add_argument("--neigh-per-row", type=int, default=0,
                   help="vizinhos esperados por linha ao dimensionar o lote (0 = pior caso "
                        "N, como o cuML). Também não é exposto pelo cuml.cluster.DBSCAN")
    p.add_argument("--repeat", type=int, default=5, help="execuções medidas (mediana)")
    p.add_argument("--warmup", type=int, default=1, help="execuções descartadas antes")
    p.add_argument("--max-mbytes-per-batch", type=int, default=0,
                   help="mesmo limite de lote nos dois lados; 0 = automático")
    p.add_argument("--validar", action="store_true",
                   help="compara os rótulos com os do cuML (ARI, ruído, partição idêntica)")
    p.add_argument(
        "--oraculo-semantico",
        choices=("auto", "sempre", "nunca"),
        default="auto",
        help=(
            "auto verifica divergências; sempre verifica toda configuração; nunca exige "
            "igualdade particional. O oráculo CPU é independente de cuML/CUDA"
        ),
    )
    p.add_argument(
        "--oraculo-max-n",
        type=int,
        default=5000,
        help="maior N permitido no grafo exato CPU (0 = sem limite; padrão: 5000)",
    )
    p.add_argument(
        "--tolerar-divergencia-sem-oraculo",
        action="store_true",
        help=(
            "não encerra com erro quando o oráculo não pôde arbitrar uma divergência "
            "(N acima de --oraculo-max-n); conta a ocorrência e preserva o artefato. "
            "Para a campanha de DESEMPENHO: a alegação de correção continua vindo de "
            "tools/run_validation_matrix.py, onde o oráculo cabe"
        ),
    )
    p.add_argument(
        "--falhas-dir",
        default="validation_failures",
        help="diretório dos artefatos reproduzíveis gerados por validações inválidas",
    )
    p.add_argument("--imposto", action="store_true",
                   help="mede também o custo de uma configuração isolada (k=1, l=1) contra "
                        "uma chamada única do cuML — é o T do qual depende o ganho")
    p.add_argument("--imposto-por-processo", action="store_true",
                   help="mede cada configuração isolada num processo separado. Mais caro "
                        "(um contexto CUDA por configuração) e assimétrico com o baseline, "
                        "que roda num processo Python só. O padrão mede tudo num processo, "
                        "com workspace reaproveitado — ver README. Este modo existe para "
                        "conferir os dois protocolos, e porque inclui a alocação do "
                        "workspace em cada configuração, como o cuml.cluster.DBSCAN faz")
    p.add_argument("--out", help="grava o JSON de resultado neste arquivo")
    args = p.parse_args()

    if args.oraculo_max_n < 0:
        p.error("--oraculo-max-n deve ser >= 0")
    if args.max_mbytes_per_batch < 0:
        p.error("--max-mbytes-per-batch deve ser >= 0")

    nome, input_path, n, d, eps, min_samples, meta_points_sha256 = carregar_config(args)
    k, l = len(eps), len(min_samples)
    n_cfg = k * l

    if not Path(args.binario).exists():
        raise SystemExit(f"erro: binário não encontrado em '{args.binario}' (rode make)")
    if not Path(input_path).exists():
        raise SystemExit(f"erro: entrada não encontrada em '{input_path}'")

    hashes_sha256 = {
        "binario": sha256_arquivo(args.binario),
        "input": sha256_arquivo(input_path),
        "meta": sha256_arquivo(args.meta) if args.meta else None,
        "input_esperado_no_meta": meta_points_sha256,
    }
    validar_hash_input_meta(hashes_sha256["input"], meta_points_sha256)

    try:
        import cupy as cp  # noqa: F401
        import cuml
    except ImportError as e:
        raise SystemExit(
            f"erro: {e}. O baseline precisa do RAPIDS:\n"
            "  INSTALL_CUML=1 bash scripts/setup_env.sh"
        )

    print(f"dataset: {nome}  N={n}  D={d}")
    print(f"grade:   {k} eps x {l} minPts = {n_cfg} configurações")
    print(f"eps:     {[round(e, 5) for e in eps]}")
    print(f"minPts:  {min_samples}")
    print(f"backend: {args.backend}  índice: {args.index}"
          f"{f'  neigh/linha: {args.neigh_per_row}' if args.neigh_per_row else ''}")
    print(f"medição: warmup={args.warmup} repeat={args.repeat} (mediana)\n")
    if args.max_mbytes_per_batch == 0:
        print(
            "aviso: orçamento automático não é equivalente: o multi usa memória livre "
            "e workspace reutilizável; cuML usa sua própria política sobre memória total. "
            "Use --max-mbytes-per-batch > 0 para controlar esse fator.\n"
        )

    X = np.fromfile(input_path, dtype=np.float32)
    esperado = n * d
    if X.size != esperado:
        raise SystemExit(f"erro: '{input_path}' tem {X.size} floats, esperado {esperado}")
    X = X.reshape(n, d)
    X_gpu = cp.asarray(X)  # fora de qualquer região cronometrada, dos dois lados

    tmpdir = tempfile.TemporaryDirectory()
    labels_path = Path(tmpdir.name) / "labels.i32" if args.validar else None

    # --- nosso binário, grade inteira de uma vez -----------------------------
    # Uma invocação só: a grade de uma vez e, com --imposto, cada configuração isolada no
    # mesmo processo. É o que torna a medição isolada simétrica com a do baseline, que
    # percorre as configurações num processo Python único.
    nosso, nossos_rotulos = rodar_binario(
        args.binario, input_path, n, d, eps, min_samples,
        args.repeat, args.warmup, args.max_mbytes_per_batch, args.backend,
        args.index, args.neigh_per_row, labels_path,
        solo=args.imposto and not args.imposto_por_processo,
        permitir_binario_legado=args.permitir_binario_legado)
    multi_ms = float(nosso["fit_ms"])
    build_runtime = nosso.get("build")
    build_identificado = bool(
        build_possui_identidade(build_runtime)
        and build_runtime.get("configured_backend") in {"codes", "cuvs"}
        and isinstance(build_runtime.get("compiled_backends"), list)
        and len(build_runtime["compiled_backends"])
        == len(set(build_runtime["compiled_backends"]))
        and all(
            backend_name in {"codes", "cuvs"}
            for backend_name in build_runtime["compiled_backends"]
        )
    )
    proveniencia_incompleta = bool(
        not build_identificado or (args.meta and meta_points_sha256 is None)
    )

    # --- baseline cuML, uma chamada por configuração -------------------------
    print("--- baseline cuML (sequencial) ---")
    por_config = []
    cache_grafos = {}
    artefatos_falha = []
    for e_i, e in enumerate(eps):
        for m_i, m in enumerate(min_samples):
            tempos, rotulos = medir_cuml(X_gpu, e, m, args.repeat, args.warmup,
                                         args.max_mbytes_per_batch, args.validar)
            entrada = {
                "config": e_i * l + m_i,
                "eps": float(e),
                "min_samples": int(m),
                "cuml_ms": mediana(tempos),
                "cuml_ms_all": tempos,
            }
            if args.validar:
                cfg = e_i * l + m_i
                validacao = validar_cientificamente(
                    X,
                    nossos_rotulos[cfg],
                    rotulos,
                    e,
                    m,
                    modo=args.oraculo_semantico,
                    max_n=args.oraculo_max_n,
                    cache_grafos=cache_grafos,
                )
                if not validacao["valida"]:
                    artefato = write_failure_artifact(
                        args.falhas_dir,
                        dataset_name=nome,
                        points=X,
                        labels={"nosso": nossos_rotulos[cfg], "cuml": rotulos},
                        eps=e,
                        min_samples=m,
                        validation=validacao,
                        source_path=input_path,
                        context={
                            "config": cfg,
                            "backend": args.backend,
                            "index_solicitado": args.index,
                            "binario": str(args.binario),
                            "oraculo_semantico": args.oraculo_semantico,
                            "oraculo_max_n": args.oraculo_max_n,
                            "sha256": hashes_sha256,
                            "build": nosso.get("build"),
                            "cuda": nosso.get("cuda"),
                            "execution": nosso.get("execution"),
                            "argv": nosso.get("argv"),
                        },
                    )
                    validacao["artefato_falha"] = str(artefato)
                    artefatos_falha.append(str(artefato))
                entrada["validacao"] = validacao
            por_config.append(entrada)
            print(f"  eps={e:<8.5g} minPts={m:<4d} {entrada['cuml_ms']:10.3f} ms")

    sequencial_ms = sum(c["cuml_ms"] for c in por_config)
    speedup = sequencial_ms / multi_ms if multi_ms > 0 else float("nan")

    print(f"  {'total':<22} {sequencial_ms:10.3f} ms\n")
    print("--- multiparamétrico (uma execução) ---")
    print(f"  {'fit_ms (mediana)':<22} {multi_ms:10.3f} ms")
    print(f"  {'tempos':<22} {[round(t, 3) for t in nosso['fit_ms_all']]}\n")
    print("--- resultado ---")
    print(f"  speedup                {speedup:10.2f}x  sobre {n_cfg} configurações")
    print(f"  tempo por configuração {multi_ms / n_cfg:10.3f} ms  "
          f"(cuML: {sequencial_ms / n_cfg:.3f} ms)")

    resultado = {
        "dataset": nome,
        "n": n,
        "d": d,
        "eps": eps,
        "min_samples": min_samples,
        "configuration_count": n_cfg,
        "config_order": "eps_major",
        "backend": nosso["backend"],
        "backend_solicitado": args.backend,
        "index": nosso["index"],
        "index_solicitado": args.index,
        "permitiu_binario_legado": args.permitir_binario_legado,
        "proveniencia_incompleta": proveniencia_incompleta,
        "neigh_per_row": args.neigh_per_row,
        "imposto_protocolo": "processo" if args.imposto_por_processo else "in-process",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "max_mbytes_per_batch": args.max_mbytes_per_batch,
        "batch_budget_protocol": {
            "controlled_equal_request": args.max_mbytes_per_batch > 0,
            "requested_mbytes_each": (
                args.max_mbytes_per_batch if args.max_mbytes_per_batch > 0 else None
            ),
            "multi_requested_bytes": (
                args.max_mbytes_per_batch * 1000000
                if args.max_mbytes_per_batch > 0
                else None
            ),
            "multi_effective_bytes": (
                (nosso.get("execution") or {}).get("effective_max_bytes_per_batch")
            ),
            "multi_auto_policy": (
                None
                if args.max_mbytes_per_batch > 0
                else "80_percent_free_memory_with_reused_external_workspace"
            ),
            "cuml_requested_bytes": (
                args.max_mbytes_per_batch * 1000000
                if args.max_mbytes_per_batch > 0
                else None
            ),
            "cuml_effective_bytes": None,
            "cuml_effective_observable": False,
            "cuml_auto_policy": (
                None
                if args.max_mbytes_per_batch > 0
                else "cuml_default_80_percent_total_memory_minus_dataset"
            ),
        },
        "multi_ms": multi_ms,
        "multi_ms_all": nosso["fit_ms_all"],
        "cuml_sequencial_ms": sequencial_ms,
        "speedup": speedup,
        "por_config": por_config,
        "baseline": dict(CUML_BASELINE),
        "sha256": hashes_sha256,
        # Preservados sem reinterpretar: são a evidência produzida pelo próprio runtime.
        "build": nosso.get("build"),
        "cuda": nosso.get("cuda"),
        "execution": nosso.get("execution"),
        "argv": nosso.get("argv"),
        "ambiente": coletar_ambiente(
            nosso.get("cuda"),
            gpu=cp.cuda.runtime.getDeviceProperties(0)["name"],
            cuml_version=cuml.__version__,
            cupy_version=cp.__version__,
        ),
    }

    # --- imposto por configuração isolada ------------------------------------
    # Sem isso não dá para separar "o multiparamétrico ganha" de "o nosso kernel é lento e
    # o ganho vem só de amortizar a lentidão". Com T = nosso_isolado / cuml na mesma
    # configuração, o Multi-EPS só compensa quando k > T.
    if args.imposto:
        print("\n--- imposto por configuração (k=1, l=1 contra chamada única) ---")
        if args.imposto_por_processo:
            print("  (um processo por configuração; inclui a alocação do workspace em "
                  "cada uma, como o cuML)")
            solos = []
            for c in por_config:
                r, _ = rodar_binario(
                    args.binario, input_path, n, d, [c["eps"]], [c["min_samples"]],
                    args.repeat, args.warmup, args.max_mbytes_per_batch, args.backend,
                    args.index, args.neigh_per_row, None,
                    permitir_binario_legado=args.permitir_binario_legado)
                solos.append(float(r["fit_ms"]))
        else:
            solos = nosso.get("solo_ms")
        if not solos or len(solos) != len(por_config):
            raise SystemExit(
                f"erro: o binário devolveu {len(solos or [])} tempos isolados para "
                f"{len(por_config)} configurações. Recompile: --solo é recente."
            )
        impostos = []
        for c, solo_ms in zip(por_config, solos):
            solo_ms = float(solo_ms)
            t = solo_ms / c["cuml_ms"] if c["cuml_ms"] > 0 else float("nan")
            c["nosso_solo_ms"] = solo_ms
            c["imposto"] = t
            impostos.append(t)
            print(f"  eps={c['eps']:<8.5g} minPts={c['min_samples']:<4d} "
                  f"nosso={solo_ms:9.3f} ms  cuML={c['cuml_ms']:9.3f} ms  T={t:6.2f}x")
        t_mediano = mediana(impostos)
        resultado["imposto_mediano"] = t_mediano
        resultado["imposto_max"] = float(max(impostos))

        # Ganho do multi ISOLADO: nosso binário rodando a grade de uma vez contra o mesmo
        # binário rodando uma configuração de cada vez. Não depende do tipo de índice nem
        # de --neigh-per-row, porque as duas metades usam os mesmos ajustes — é o único
        # número que mede só o compartilhamento entre configurações.
        solo_total = sum(c["nosso_solo_ms"] for c in por_config)
        ganho_puro = solo_total / multi_ms if multi_ms > 0 else float("nan")
        resultado["nosso_sequencial_ms"] = solo_total
        resultado["ganho_multi_puro"] = ganho_puro

        print(f"\n  T mediano = {t_mediano:.2f}x  (nosso contra cuML, 1 configuração)")
        print(f"  ganho do multi isolado = {ganho_puro:.2f}x  "
              f"({solo_total:.1f} ms sequencial no nosso binário -> {multi_ms:.1f} ms)")
        if t_mediano < 0.95:
            print(f"  atenção: T < 1. Um fator de {1/t_mediano:.2f}x do speedup vem de "
                  f"ajustes que o cuml.cluster.DBSCAN não expõe (--index/--neigh-per-row), "
                  f"não do multi")

    # --- validação -----------------------------------------------------------
    codigo_saida = 0
    if args.validar:
        print("\n--- validação contra o cuML ---")
        identicas = sum(1 for c in por_config if c["validacao"]["particao_identica"])
        aprovadas = sum(1 for c in por_config if c["validacao"]["valida"])
        oraculos_executados = sum(
            1 for c in por_config if c["validacao"]["oraculo_executado"]
        )
        ari_min = min(c["validacao"]["ari"] for c in por_config)
        for c in por_config:
            v = c["validacao"]
            if v["valida"] and v["particao_identica"]:
                marca = "identica"
            elif v["valida"]:
                marca = "borda ambigua valida"
            else:
                marca = "INVALIDA"
            print(f"  eps={c['eps']:<8.5g} minPts={c['min_samples']:<4d} {marca:>18}  "
                  f"clusters {v['n_clusters_nosso']}/{v['n_clusters_cuml']}  "
                  f"ruido {v['n_ruido_nosso']}/{v['n_ruido_cuml']}")
            if not v["valida"]:
                print(f"    status: {v['status']}; artefato: {v.get('artefato_falha')}")
        resultado["particoes_identicas"] = identicas
        resultado["ari_minimo"] = ari_min
        resultado["validacoes_aprovadas"] = aprovadas
        resultado["validacao_aprovada"] = aprovadas == n_cfg
        resultado["oraculos_semanticos_executados"] = oraculos_executados
        resultado["validacao_semantica_completa"] = oraculos_executados == n_cfg
        resultado["oraculo_semantico"] = args.oraculo_semantico
        resultado["oraculo_max_n"] = args.oraculo_max_n
        resultado["artefatos_falha"] = artefatos_falha

        # Sem árbitro não é o mesmo que aprovado. O contador e o modo ficam no JSON para que
        # nenhum leitor confunda uma execução tolerante com uma execução limpa, e para que
        # a arbitragem pendente seja localizável pelos artefatos.
        sem_arbitro = [
            c for c in por_config
            if not c.get("validacao", {}).get("valida", False)
            and c.get("validacao", {}).get("status") == STATUS_SEM_ARBITRO
        ]
        resultado["divergencias_sem_oraculo"] = len(sem_arbitro)
        resultado["tolerou_divergencia_sem_oraculo"] = bool(
            args.tolerar_divergencia_sem_oraculo
        )

        codigo_saida = validacao_exit_code(
            por_config, tolerar_sem_oraculo=args.tolerar_divergencia_sem_oraculo
        )
        print(
            f"\n  {identicas}/{n_cfg} partições idênticas; "
            f"{aprovadas}/{n_cfg} validações aprovadas; "
            f"{oraculos_executados}/{n_cfg} oráculos semânticos; ARI mínimo {ari_min:.6f}"
        )
        if sem_arbitro:
            print(
                f"  {len(sem_arbitro)}/{n_cfg} divergências sem árbitro "
                f"(N acima de --oraculo-max-n={args.oraculo_max_n}); artefatos preservados"
                + (" — toleradas por --tolerar-divergencia-sem-oraculo"
                   if args.tolerar_divergencia_sem_oraculo else "")
            )
        if codigo_saida:
            print(f"  FALHA científica: processo encerrará com código {codigo_saida}")

    tmpdir.cleanup()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nresultado gravado em {args.out}")

    print()
    print(json.dumps(resultado))
    return codigo_saida


if __name__ == "__main__":
    raise SystemExit(main())
