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
  - o mesmo --max-mbytes-per-batch dos dois lados, já que o tamanho do lote muda o número
    de fusões de rótulos.

Uso:
    python tools/bench_vs_cuml.py --meta data/moons_16d_n100000.json --validar
    python tools/bench_vs_cuml.py --meta data/moons_16d_n100000.json --imposto
    python tools/bench_vs_cuml.py --input X.f32 --n 100000 --d 16 \
        --eps 0.25,0.35,0.5 --min-samples 5,10,20 --repeat 5

A última linha do stdout é o JSON com todos os números; --out grava o mesmo em arquivo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# =============================================================================
# Execução do nosso binário
# =============================================================================


def rodar_binario(binario, input_path, n, d, eps, min_samples, repeat, warmup,
                  max_mbytes, backend, index, neigh_per_row, saida_labels=None,
                  solo=False):
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

    linhas = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not linhas:
        raise SystemExit(f"erro: {binario} não produziu saída JSON")
    resultado = json.loads(linhas[-1])

    rotulos = None
    if saida_labels is not None:
        n_cfg = len(eps) * len(min_samples)
        rotulos = np.fromfile(saida_labels, dtype=np.int32).reshape(n_cfg, n)
    return resultado, rotulos


# =============================================================================
# Baseline cuML
# =============================================================================


def medir_cuml(X_gpu, eps, min_samples, repeat, warmup, max_mbytes, guardar_rotulos):
    """Cronometra uma configuração do cuml.cluster.DBSCAN com os dados já na GPU."""
    import cupy as cp
    from cuml.cluster import DBSCAN

    kwargs = dict(
        eps=float(eps),
        min_samples=int(min_samples),
        # Não calculamos os índices de core points; cobrá-los do baseline inflaria o ganho.
        calc_core_sample_indices=False,
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
    saida = np.full_like(rotulos, -1)
    mapa = {}
    for i, v in enumerate(rotulos):
        if v < 0:
            continue
        if v not in mapa:
            mapa[v] = len(mapa)
        saida[i] = mapa[v]
    return saida


def comparar(nossos, deles):
    """ARI, concordância de ruído e igualdade exata da partição."""
    from sklearn.metrics import adjusted_rand_score

    # Contagem por rótulos distintos, não por max+1: os dois lados produzem numeração
    # contígua, mas contar assim não depende disso.
    def n_clusters(v):
        return int(np.unique(v[v >= 0]).size)

    return {
        "ari": float(adjusted_rand_score(deles, nossos)),
        "concordancia_ruido": float(np.mean((nossos < 0) == (deles < 0))),
        "particao_identica": bool(np.array_equal(canonizar(nossos), canonizar(deles))),
        "n_clusters_nosso": n_clusters(nossos),
        "n_clusters_cuml": n_clusters(deles),
        "n_ruido_nosso": int((nossos < 0).sum()),
        "n_ruido_cuml": int((deles < 0).sum()),
    }


# =============================================================================
# CLI
# =============================================================================


def mediana(valores):
    return float(np.median(np.asarray(valores, dtype=np.float64)))


def carregar_config(args):
    """Resolve entrada e grade a partir do --meta do gerador ou dos argumentos soltos."""
    if args.meta:
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        base = Path(args.meta).parent
        input_path = base / meta["arquivos"]["points"]
        n, d = int(meta["n"]), int(meta["d"])
        eps = [float(v) for v in meta["eps"]]
        min_samples = [int(v) for v in meta["min_samples"]]
        nome = meta["dataset"]
    else:
        if not (args.input and args.n and args.d and args.eps and args.min_samples):
            raise SystemExit("erro: informe --meta, ou --input/--n/--d/--eps/--min-samples")
        input_path = Path(args.input)
        n, d = int(args.n), int(args.d)
        eps = [float(v) for v in args.eps.split(",") if v]
        min_samples = [int(v) for v in args.min_samples.split(",") if v]
        nome = input_path.stem

    if args.eps and args.meta:
        eps = [float(v) for v in args.eps.split(",") if v]
    if args.min_samples and args.meta:
        min_samples = [int(v) for v in args.min_samples.split(",") if v]

    eps = sorted(set(eps))
    min_samples = sorted(set(min_samples))
    return nome, input_path, n, d, eps, min_samples


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

    nome, input_path, n, d, eps, min_samples = carregar_config(args)
    k, l = len(eps), len(min_samples)
    n_cfg = k * l

    if not Path(args.binario).exists():
        raise SystemExit(f"erro: binário não encontrado em '{args.binario}' (rode make)")
    if not Path(input_path).exists():
        raise SystemExit(f"erro: entrada não encontrada em '{input_path}'")

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
        solo=args.imposto and not args.imposto_por_processo)
    multi_ms = float(nosso["fit_ms"])

    # --- baseline cuML, uma chamada por configuração -------------------------
    print("--- baseline cuML (sequencial) ---")
    por_config = []
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
                entrada["validacao"] = comparar(nossos_rotulos[cfg], rotulos)
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
        "backend": args.backend,
        "index": args.index,
        "neigh_per_row": args.neigh_per_row,
        "imposto_protocolo": "processo" if args.imposto_por_processo else "in-process",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "max_mbytes_per_batch": args.max_mbytes_per_batch,
        "multi_ms": multi_ms,
        "multi_ms_all": nosso["fit_ms_all"],
        "cuml_sequencial_ms": sequencial_ms,
        "speedup": speedup,
        "por_config": por_config,
        "ambiente": {
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuml": cuml.__version__,
            "cupy": cp.__version__,
        },
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
                    args.index, args.neigh_per_row, None)
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
    if args.validar:
        print("\n--- validação contra o cuML ---")
        identicas = sum(1 for c in por_config if c["validacao"]["particao_identica"])
        ari_min = min(c["validacao"]["ari"] for c in por_config)
        for c in por_config:
            v = c["validacao"]
            marca = "identica" if v["particao_identica"] else f"ARI={v['ari']:.6f}"
            print(f"  eps={c['eps']:<8.5g} minPts={c['min_samples']:<4d} {marca:>18}  "
                  f"clusters {v['n_clusters_nosso']}/{v['n_clusters_cuml']}  "
                  f"ruido {v['n_ruido_nosso']}/{v['n_ruido_cuml']}")
        resultado["particoes_identicas"] = identicas
        resultado["ari_minimo"] = ari_min
        print(f"\n  {identicas}/{n_cfg} partições idênticas, ARI mínimo {ari_min:.6f}")

    tmpdir.cleanup()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nresultado gravado em {args.out}")

    print()
    print(json.dumps(resultado))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
