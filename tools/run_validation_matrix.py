#!/usr/bin/env python3
"""Executa a matriz cuVS/codes/cuML e o oráculo semântico no ClusterGPU.

Este é um gate de correção, não um benchmark. Usa entradas pequenas/adversariais,
compara rotas, índices e batching e retorna código 2 ao primeiro conjunto inválido.
Para a campanha extensa, aumente ``--random-seeds`` para 100.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

try:
    from bench_vs_cuml import validar_contrato_runtime
    from dbscan_validation import write_failure_artifact
    from validate_dbscan_matrix import validate_matrix
except ImportError:  # pragma: no cover - importado como tools.run_validation_matrix
    from tools.bench_vs_cuml import validar_contrato_runtime
    from tools.dbscan_validation import write_failure_artifact
    from tools.validate_dbscan_matrix import validate_matrix


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_last_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("executável não produziu uma linha JSON")


def _run_binary(
    binary: Path,
    points_path: Path,
    output_path: Path,
    points: np.ndarray,
    eps: list[float],
    min_samples: list[int],
    *,
    backend: str,
    index: str,
    route: str = "auto",
    max_mbytes: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    command = [
        str(binary),
        "--input", str(points_path),
        "--output", str(output_path),
        "--n", str(points.shape[0]),
        "--d", str(points.shape[1]),
        "--eps", ",".join(format(float(value), ".10g") for value in eps),
        "--min-samples", ",".join(str(int(value)) for value in min_samples),
        "--backend", backend,
        "--index", index,
        "--route", route,
        "--warmup", "0",
        "--repeat", "1",
        "--json",
    ]
    if max_mbytes:
        command += ["--max-mbytes-per-batch", str(max_mbytes)]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"{backend}/{index}/{route} terminou com código {completed.returncode}"
        )
    if completed.stderr:
        # Warnings de batching/allocator em execucoes aprovadas continuam relevantes para
        # reproduzir a matriz e nao podem desaparecer do log do job.
        sys.stderr.write(completed.stderr)
    metadata = _json_last_line(completed.stdout)
    try:
        validar_contrato_runtime(
            metadata,
            backend=backend,
            index=index,
            n=points.shape[0],
            d=points.shape[1],
            eps=eps,
            min_samples=min_samples,
            permitir_binario_legado=False,
        )
    except SystemExit as error:
        raise RuntimeError(str(error)) from error
    if metadata.get("backend") != backend:
        raise RuntimeError(
            f"backend efetivo {metadata.get('backend')!r}, solicitado {backend!r}"
        )
    if metadata.get("index") != index:
        raise RuntimeError(
            f"índice efetivo {metadata.get('index')!r}, solicitado {index!r}"
        )
    if metadata.get("requested_route") != route:
        raise RuntimeError(
            f"rota registrada {metadata.get('requested_route')!r}, solicitada {route!r}"
        )
    expected_configurations = len(eps) * len(min_samples)
    if int(metadata.get("configuration_count", -1)) != expected_configurations:
        raise RuntimeError("configuration_count divergente no JSON do executável")
    build = metadata.get("build") or {}
    if build.get("git_sha") in (None, "", "unknown") or build.get("build_id") in (
        None,
        "",
        "unknown",
    ):
        raise RuntimeError("binário sem proveniência; recompile com o Makefile atual")
    if backend not in (build.get("compiled_backends") or []):
        raise RuntimeError(f"backend {backend!r} não consta nos backends compilados")

    labels = np.fromfile(output_path, dtype=np.int32)
    expected_labels = expected_configurations * points.shape[0]
    if labels.size != expected_labels:
        raise RuntimeError(
            f"arquivo de labels contém {labels.size} inteiros; esperado {expected_labels}"
        )
    return labels.reshape(expected_configurations, points.shape[0]), metadata


def _cluster(center: tuple[float, float], count: int, spread: float, rng) -> np.ndarray:
    return rng.normal(center, spread, size=(count, 2)).astype(np.float32)


def adversarial_cases(random_seeds: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    cases.append(
        {
            "name": "all_noise_1d",
            "points": (np.arange(24, dtype=np.float32) * 10.0).reshape(-1, 1),
            "eps": [0.1, 1.0],
            "min_samples": [2, 4],
        }
    )
    cases.append(
        {
            "name": "one_cluster_duplicates_2d",
            "points": np.zeros((32, 2), dtype=np.float32),
            "eps": [
                float(np.nextafter(np.float32(0.1), np.float32(-np.inf))),
                0.1,
                float(np.nextafter(np.float32(0.1), np.float32(np.inf))),
            ],
            "min_samples": [1, 8, 32, 33],
        }
    )

    left = np.asarray(
        [[-0.50, 0.00], [-0.80, 0.00], [-0.80, 0.05], [-0.80, -0.05],
         [-0.85, 0.00], [-0.75, 0.00]],
        dtype=np.float32,
    )
    right = left.copy()
    right[:, 0] *= -1
    ambiguous = np.vstack([left, [[0.0, 0.0]], right]).astype(np.float32)
    cases.append(
        {
            "name": "ambiguous_border_2d",
            "points": ambiguous,
            "eps": [0.49, 0.51],
            "min_samples": [4, 5, 7],
        }
    )

    boundary = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
    cases.append(
        {
            "name": "epsilon_nextafter_1d",
            "points": boundary,
            "eps": [
                float(np.nextafter(np.float32(1.0), np.float32(-np.inf))),
                1.0,
                float(np.nextafter(np.float32(1.0), np.float32(np.inf))),
            ],
            "min_samples": [2, 3],
        }
    )

    # Mesmo problema geométrico em todos os ramos dimensionais do kernel.
    rng = np.random.default_rng(2026)
    base = np.vstack(
        [_cluster((-1.0, 0.0), 20, 0.05, rng), _cluster((1.0, 0.0), 20, 0.05, rng)]
    )
    for dimension in (1, 2, 16, 33):
        points = np.zeros((base.shape[0], dimension), dtype=np.float32)
        points[:, : min(2, dimension)] = base[:, : min(2, dimension)]
        cases.append(
            {
                "name": f"dimension_{dimension}",
                "points": points,
                "eps": [0.08, 0.18, 0.40],
                "min_samples": [3, 6],
            }
        )

    # Exercita o fallback codes sem shared memory, mas mantém N pequeno para o oráculo.
    high = np.zeros((24, 8193), dtype=np.float32)
    high[:12, 0] = np.linspace(-1.05, -0.95, 12, dtype=np.float32)
    high[12:, 0] = np.linspace(0.95, 1.05, 12, dtype=np.float32)
    cases.append(
        {
            "name": "high_dimension_8193",
            "points": high,
            "eps": [0.03, 0.15],
            "min_samples": [2, 5],
        }
    )

    # N=1000 torna 1 MB suficiente para forçar múltiplos lotes em pelo menos um modo.
    rng = np.random.default_rng(3000)
    batched = np.vstack(
        [_cluster((-1.0, 0.0), 500, 0.08, rng), _cluster((1.0, 0.0), 500, 0.08, rng)]
    )
    cases.append(
        {
            "name": "forced_batches_2d",
            "points": batched,
            "eps": [0.04, 0.10],
            "min_samples": [4, 12],
            "must_batch": True,
        }
    )

    for seed in range(random_seeds):
        rng = np.random.default_rng(10_000 + seed)
        points = np.vstack(
            [
                _cluster((-1.0, -0.2), 20, 0.10, rng),
                _cluster((0.0, 1.0), 20, 0.10, rng),
                _cluster((1.0, -0.2), 20, 0.10, rng),
                rng.uniform(-1.5, 1.5, size=(8, 2)).astype(np.float32),
            ]
        )
        cases.append(
            {
                "name": f"random_seed_{seed:03d}",
                "points": points,
                "eps": [0.10, 0.20, 0.35],
                "min_samples": [3, 5, 9, 15],
            }
        )
    return cases


def _run_cuml(points: np.ndarray, eps: list[float], min_samples: list[int]):
    import cupy as cp
    from cuml.cluster import DBSCAN

    device_points = cp.asarray(points)
    outputs = []
    for epsilon in eps:
        for minimum in min_samples:
            model = DBSCAN(
                eps=float(epsilon),
                min_samples=int(minimum),
                metric="euclidean",
                algorithm="brute",
                calc_core_sample_indices=False,
                output_type="cupy",
            )
            model.fit(device_points)
            cp.cuda.Stream.null.synchronize()
            outputs.append(cp.asnumpy(model.labels_).astype(np.int32, copy=False))
    return np.asarray(outputs, dtype=np.int32)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path,
                        help="build cuVS atual; ele também contém o backend codes")
    parser.add_argument("--random-seeds", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--oraculo-max-n", type=int, default=5000)
    parser.add_argument("--falhas-dir", default="validation_failures")
    parser.add_argument("--out", default="results/validation-matrix.json")
    args = parser.parse_args(argv)
    if args.random_seeds < 0 or args.repetitions < 2:
        parser.error("--random-seeds deve ser >= 0 e --repetitions deve ser >= 2")
    if not args.binary.is_file():
        parser.error(f"binário não encontrado: {args.binary}")

    modes = [
        {"name": "cuvs_auto_i32", "backend": "cuvs", "index": "int32"},
        {"name": "cuvs_annotated_i32", "backend": "cuvs", "index": "int32",
         "route": "annotated"},
        {"name": "cuvs_dense_i32", "backend": "cuvs", "index": "int32",
         "route": "dense"},
        {"name": "cuvs_auto_i64", "backend": "cuvs", "index": "int64"},
        {"name": "codes_i32", "backend": "codes", "index": "int32"},
        {"name": "codes_i64", "backend": "codes", "index": "int64"},
        {"name": "cuvs_batched", "backend": "cuvs", "index": "int32", "max_mbytes": 1},
        {"name": "codes_batched", "backend": "codes", "index": "int32", "max_mbytes": 1},
    ]

    result_output = Path(args.out)
    result_output.parent.mkdir(parents=True, exist_ok=True)
    case_results = []
    all_valid = True
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        for case_index, case in enumerate(adversarial_cases(args.random_seeds)):
            name = case["name"]
            points = np.ascontiguousarray(case["points"], dtype=np.float32)
            eps = sorted({float(value) for value in case["eps"]})
            minimums = sorted({int(value) for value in case["min_samples"]})
            points_path = temporary_path / f"{case_index:03d}_{name}.f32"
            points.tofile(points_path)

            labels_by_backend: dict[str, np.ndarray] = {}
            repetition_labels: dict[str, np.ndarray] = {}
            executions: dict[str, Any] = {}
            deterministic = True
            determinism_failures: list[dict[str, Any]] = []
            determinism_pairs: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
            for mode in modes:
                repetitions = []
                metadata = None
                mode_repetitions = (
                    args.repetitions
                    if mode["name"] in {"cuvs_auto_i32", "codes_i32"}
                    else 1
                )
                for repetition in range(mode_repetitions):
                    labels_output = temporary_path / (
                        f"{case_index}_{mode['name']}_{repetition}.i32"
                    )
                    labels, metadata = _run_binary(
                        args.binary.resolve(),
                        points_path,
                        labels_output,
                        points,
                        eps,
                        minimums,
                        backend=mode["backend"],
                        index=mode["index"],
                        route=mode.get("route", "auto"),
                        max_mbytes=mode.get("max_mbytes", 0),
                    )
                    repetitions.append(labels)
                for repetition_index, item in enumerate(repetitions[1:], start=1):
                    differences = np.argwhere(repetitions[0] != item)
                    if differences.size:
                        config_index, point_index = (int(value) for value in differences[0])
                        deterministic = False
                        failure = {
                            "source": mode["name"],
                            "reference_repetition": 0,
                            "different_repetition": repetition_index,
                            "config": config_index,
                            "point": point_index,
                            "label_rep0": int(repetitions[0][config_index, point_index]),
                            "label_different": int(item[config_index, point_index]),
                        }
                        determinism_failures.append(failure)
                        determinism_pairs[mode["name"]] = (
                            repetitions[0],
                            item,
                            repetition_index,
                        )
                        break
                for repetition_index, item in enumerate(repetitions):
                    repetition_labels[f"{mode['name']}_rep{repetition_index}"] = item
                labels_by_backend[mode["name"]] = repetitions[0]
                executions[mode["name"]] = {
                    "repetitions": mode_repetitions,
                    "result": metadata,
                }

            cuml_repetitions = [_run_cuml(points, eps, minimums) for _ in range(args.repetitions)]
            for repetition_index, item in enumerate(cuml_repetitions[1:], start=1):
                differences = np.argwhere(cuml_repetitions[0] != item)
                if differences.size:
                    config_index, point_index = (int(value) for value in differences[0])
                    deterministic = False
                    failure = {
                        "source": "cuml",
                        "reference_repetition": 0,
                        "different_repetition": repetition_index,
                        "config": config_index,
                        "point": point_index,
                        "label_rep0": int(cuml_repetitions[0][config_index, point_index]),
                        "label_different": int(item[config_index, point_index]),
                    }
                    determinism_failures.append(failure)
                    determinism_pairs["cuml"] = (
                        cuml_repetitions[0],
                        item,
                        repetition_index,
                    )
                    break
            for repetition_index, item in enumerate(cuml_repetitions):
                repetition_labels[f"cuml_rep{repetition_index}"] = item
            labels_by_backend["cuml"] = cuml_repetitions[0]

            # The TemporaryDirectory is an execution detail, not scientific storage. Keep
            # every repeated label vector plus X/grid beside the final matrix JSON so a PASS
            # remains independently auditable after the job exits.
            safe_name = "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in name
            )
            labels_evidence_path = result_output.parent / (
                f"{result_output.stem}.{case_index:03d}_{safe_name}.labels.npz"
            )
            np.savez_compressed(
                labels_evidence_path,
                points=points,
                eps=np.asarray(eps, dtype=np.float64),
                min_samples=np.asarray(minimums, dtype=np.int64),
                **{
                    f"labels_{source}": values
                    for source, values in repetition_labels.items()
                },
            )
            labels_evidence = {
                "path": str(labels_evidence_path),
                "sha256": sha256_file(labels_evidence_path),
                "config_order": "eps_major",
                "arrays": sorted(repetition_labels),
            }

            validation = validate_matrix(
                points,
                eps,
                minimums,
                labels_by_backend,
                max_n=args.oraculo_max_n,
            )
            batched_ok = True
            if case.get("must_batch"):
                batched_ok = all(
                    int(executions[name]["result"]["execution"]["batches"]) > 1
                    for name in ("cuvs_batched", "codes_batched")
                )
            routes_ok = True
            if len(eps) > 1:
                annotated = executions["cuvs_annotated_i32"]["result"]["execution"]
                dense = executions["cuvs_dense_i32"]["result"]["execution"]
                routes_ok = (
                    int(annotated["annotated_batches"]) > 0
                    and int(annotated["dense_batches"]) == 0
                    and int(dense["dense_batches"]) > 0
                    and int(dense["annotated_batches"]) == 0
                )

            approved = bool(
                validation["validacao_aprovada"]
                and deterministic
                and batched_ok
                and routes_ok
            )
            artifacts = []
            if not approved:
                for configuration in validation["configuracoes"]:
                    config = int(configuration["config"])
                    config_determinism = [
                        failure
                        for failure in determinism_failures
                        if int(failure["config"]) == config
                    ]
                    global_execution_failure = (not batched_ok or not routes_ok) and config == 0
                    if (
                        configuration["aprovada"]
                        and not config_determinism
                        and not global_execution_failure
                    ):
                        continue
                    artifact_labels = {
                        key: value[config] for key, value in labels_by_backend.items()
                    }
                    for failure in config_determinism:
                        source = failure["source"]
                        rep0, divergent, repetition_index = determinism_pairs[source]
                        artifact_labels[f"{source}_rep0"] = rep0[config]
                        artifact_labels[f"{source}_rep{repetition_index}"] = divergent[config]

                    semantic_first = next(
                        (
                            pair["primeiro_ponto_divergente"]
                            for pair in configuration["pares"].values()
                            if not pair["valida"]
                        ),
                        None,
                    )
                    first_point = (
                        int(config_determinism[0]["point"])
                        if config_determinism
                        else semantic_first
                    )
                    failure = write_failure_artifact(
                        args.falhas_dir,
                        dataset_name=name,
                        points=points,
                        labels=artifact_labels,
                        eps=configuration["eps"],
                        min_samples=configuration["min_samples"],
                        validation={
                            "valida": False,
                            "status": "matriz_gpu_invalida",
                            "primeiro_ponto_divergente": first_point,
                            "detalhes": configuration,
                        },
                        source_path=points_path,
                        context={
                            "deterministic": deterministic,
                            "determinism_failures": config_determinism,
                            "batched_ok": batched_ok,
                            "routes_ok": routes_ok,
                            "config": config,
                            "first_point": first_point,
                        },
                    )
                    artifacts.append(str(failure))

            all_valid &= approved
            case_results.append(
                {
                    "name": name,
                    "approved": approved,
                    "deterministic": deterministic,
                    "determinism_failures": determinism_failures,
                    "batched_ok": batched_ok,
                    "routes_ok": routes_ok,
                    "points_sha256": sha256_file(points_path),
                    "n": int(points.shape[0]),
                    "d": int(points.shape[1]),
                    "eps": eps,
                    "min_samples": minimums,
                    "validation": validation,
                    "executions": executions,
                    "failure_artifacts": artifacts,
                    "labels_evidence": labels_evidence,
                }
            )
            print(f"[{case_index + 1}] {name}: {'PASS' if approved else 'FAIL'}", flush=True)

    result = {
        "schema_version": 1,
        "validation_approved": all_valid,
        "binary": str(args.binary.resolve()),
        "binary_sha256": sha256_file(args.binary),
        "matrix_script_sha256": sha256_file(Path(__file__).resolve()),
        "random_seeds": args.random_seeds,
        "repetitions": args.repetitions,
        "oracle_max_n": args.oraculo_max_n,
        "protocol": {
            "metric": "euclidean",
            "cuml_algorithm": "brute",
            "config_order": "eps_major",
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "numpy": _package_version("numpy"),
            "cupy": _package_version("cupy-cuda12x"),
            "cuml": _package_version("cuml-cu12"),
            "libraft": _package_version("libraft-cu12"),
            "librmm": _package_version("librmm-cu12"),
            "libcuvs": _package_version("libcuvs-cu12"),
        },
        "cases": case_results,
    }
    result_output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"validation_approved": all_valid, "out": str(result_output)}))
    return 0 if all_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
