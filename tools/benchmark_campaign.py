#!/usr/bin/env python3
"""Prepara, executa e agrega campanhas oficiais do cuML-DBSCANMulti.

O processo coordenador nunca cria um contexto CUDA. Cada amostra e executada em um
worker novo, de modo que o binario experimental e o baseline Python/cuML nao ocupem a
GPU simultaneamente. Os tempos oficiais usam CUDA events com os dados ja residentes;
tempos de processo/end-to-end sao preservados separadamente.

Subcomandos publicos:

* ``plan``: expande a especificacao sem executar GPU;
* ``prepare``: gera datasets em staging e os publica atomicamente;
* ``validate-inputs``: valida protocolo, grade e hashes, sem escrever;
* ``run``: exige gate semantico aprovado e executa a campanha;
* ``aggregate``: reconstrui os agregados somente a partir dos registros raw.

``_worker`` e interno. Nao o use diretamente em jobs Slurm.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
SAMPLE_SCHEMA = "../../schemas/benchmark-sample.schema.json"
MANIFEST_SCHEMA = "schemas/benchmark-run-manifest.schema.json"
DATASET_PROTOCOL = "knn-sample-rank-v2"
POOL_METHOD = "geometric-rounded-deduplicated-filled-v1"
TIMING_BOUNDARY = "device-resident-input-to-device-labels"
SCHEMA_VERSION = "1.0.0"
METHOD_SEQUENTIAL = "experimental_sequential"
METHOD_CUML = "cuml_sequential"
METHOD_MULTI = "multi"
OFFICIAL_GATE_RANDOM_SEEDS = 3
OFFICIAL_GATE_REPETITIONS = 2
OFFICIAL_GATE_ORACLE_MAX_N = 5000
# Limite por amostra/metodo. O worker experimental usa um prazo interno para
# encerrar tambem o binario filho; o coordenador conserva 30 s para serializar a falha.
METHOD_TIMEOUT_SECONDS = 900
WORKER_SHUTDOWN_GRACE_SECONDS = 30
OFFICIAL_GATE_MODES = (
    "cuvs_auto_i32",
    "cuvs_annotated_i32",
    "cuvs_dense_i32",
    "cuvs_auto_i64",
    "codes_i32",
    "codes_i64",
    "cuvs_batched",
    "codes_batched",
)
OFFICIAL_GATE_REPEATED_MODES = frozenset({"cuvs_auto_i32", "codes_i32"})


class CampaignError(RuntimeError):
    """Erro de protocolo que deve bloquear a campanha, sem esconder artefatos."""


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"JSON invalido em {path}: {exc}") from exc


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def validate_schema(instance: Any, schema_name: str) -> None:
    """Usa jsonschema quando disponivel; o gate do CI sempre instala a dependencia."""

    schema = read_json(SCHEMAS / schema_name)
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        missing = set(schema.get("required", [])) - set(instance if isinstance(instance, dict) else {})
        if missing:
            raise CampaignError(f"{schema_name}: campos obrigatorios ausentes: {sorted(missing)}")
        return
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(instance),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        messages = []
        for error in errors[:20]:
            where = ".".join(str(part) for part in error.absolute_path) or "<raiz>"
            messages.append(f"{where}: {error.message}")
        raise CampaignError(f"{schema_name}: instancia invalida:\n- " + "\n- ".join(messages))


def load_spec(path: Path) -> dict[str, Any]:
    spec = read_json(path)
    validate_schema(spec, "benchmark-campaign.schema.json")
    protocol = spec["protocol"]
    if protocol["measured_samples"] % 2:
        raise CampaignError("block_design=symmetric exige measured_samples par")
    if protocol["dataset_protocol"] != DATASET_PROTOCOL:
        raise CampaignError(f"dataset_protocol deve ser {DATASET_PROTOCOL}")
    if protocol["minpts_pool_method"] != POOL_METHOD:
        raise CampaignError(f"minpts_pool_method deve ser {POOL_METHOD}")
    ids = [case["id"] for case in spec["cases"]]
    if len(ids) != len(set(ids)):
        raise CampaignError("IDs de casos duplicados")
    for case in spec["cases"]:
        if "auto" not in case["routes"]:
            raise CampaignError(f"{case['id']}: routes deve conter auto")
    return spec


def source_tree_sha256() -> str:
    """SHA-256 completo usando exatamente o escopo de source_tree_hash.py."""

    try:
        from scripts.source_tree_hash import source_files
    except ImportError:
        sys.path.insert(0, str(ROOT))
        from scripts.source_tree_hash import source_files

    digest = hashlib.sha256()
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def minpts_pool(meta_min_samples: Iterable[int], size: int) -> list[int]:
    """Implementa geometric-rounded-deduplicated-filled-v1."""

    values = sorted({int(value) for value in meta_min_samples})
    if not values or values[0] < 1:
        raise CampaignError("meta.min_samples deve conter inteiros positivos")
    low, high = values[0], values[-1]
    if high - low + 1 < size:
        raise CampaignError(
            f"intervalo minPts [{low},{high}] nao contem {size} inteiros distintos"
        )
    if size == 1:
        return [low]
    generated = {
        int(round(math.exp(math.log(low) + i * (math.log(high) - math.log(low)) / (size - 1))))
        for i in range(size)
    }
    generated.update((low, high))
    generated = {min(high, max(low, value)) for value in generated}
    for value in range(low, high + 1):
        if len(generated) >= size:
            break
        generated.add(value)
    result = sorted(generated)
    if len(result) > size:
        # Preserve extremos; remova candidatos mais proximos entre si de forma deterministica.
        while len(result) > size:
            removable = range(1, len(result) - 1)
            index = min(
                removable,
                key=lambda i: (min(result[i] - result[i - 1], result[i + 1] - result[i]), result[i]),
            )
            result.pop(index)
    if len(result) != size:
        raise CampaignError(f"pool minPts resultou em {len(result)} valores, esperado {size}")
    return result


def dataset_key(case: dict[str, Any]) -> tuple[str, int]:
    return str(case["dataset"]), int(case["n"])


def dataset_paths(data_dir: Path, dataset: str, n: int) -> dict[str, Path]:
    base = data_dir / f"{dataset}_n{n}"
    return {
        "points": Path(f"{base}.f32"),
        "labels": Path(f"{base}.labels.i32"),
        "meta": Path(f"{base}.json"),
    }


def validate_dataset(
    data_dir: Path, dataset: str, n: int, protocol: dict[str, Any]
) -> dict[str, Any]:
    paths = dataset_paths(data_dir, dataset, n)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise CampaignError("dataset incompleto/ausente: " + ", ".join(missing))
    meta = read_json(paths["meta"])
    expected = {
        "protocolo_dataset": protocol["dataset_protocol"],
        "dataset": dataset,
        "n": n,
        "seed": protocol["seed"],
        "config_order": "eps_major",
        "dtype": "float32",
        "layout": "row-major",
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            raise CampaignError(
                f"{paths['meta']}: {field}={meta.get(field)!r}; esperado {value!r}"
            )
    dimension = meta.get("d")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise CampaignError(f"{paths['meta']}: d deve ser inteiro positivo")
    quantiles = [float(value) for value in meta.get("quantis_eps", [])]
    if quantiles != [float(value) for value in protocol["eps_quantiles"]]:
        raise CampaignError(f"{paths['meta']}: quantis_eps incompatíveis: {quantiles}")
    eps = [float(value) for value in meta.get("eps", [])]
    if (
        len(eps) != len(protocol["eps_quantiles"])
        or any(not math.isfinite(value) or value <= 0 for value in eps)
        or eps != sorted(set(eps))
    ):
        raise CampaignError(f"{paths['meta']}: grade eps deve ter valores unicos e ordenados")
    raw_minimums = meta.get("min_samples", [])
    if not isinstance(raw_minimums, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in raw_minimums
    ):
        raise CampaignError(f"{paths['meta']}: min_samples deve conter inteiros positivos")
    pool = minpts_pool(raw_minimums, int(protocol["minpts_pool_size"]))
    real_points = sha256_file(paths["points"])
    expected_points = (meta.get("sha256") or {}).get("points")
    if real_points != expected_points:
        raise CampaignError(
            f"SHA-256 points diverge em {paths['points']}: {real_points} != {expected_points}"
        )
    real_labels = sha256_file(paths["labels"])
    expected_labels = (meta.get("sha256") or {}).get("labels_verdadeiros")
    if real_labels != expected_labels:
        raise CampaignError(
            f"SHA-256 labels diverge em {paths['labels']}: {real_labels} != {expected_labels}"
        )
    expected_generator = sha256_file(ROOT / "tools" / "gerar_datasets.py")
    recorded_generator = (meta.get("sha256") or {}).get("gerador")
    if recorded_generator != expected_generator:
        raise CampaignError(
            f"SHA-256 do gerador diverge em {paths['meta']}: "
            f"{recorded_generator} != {expected_generator}; regenere em diretorio vazio"
        )
    expected_files = {
        "points": paths["points"].name,
        "labels_verdadeiros": paths["labels"].name,
    }
    if meta.get("arquivos") != expected_files:
        raise CampaignError(
            f"{paths['meta']}: arquivos={meta.get('arquivos')!r}; esperado {expected_files!r}"
        )
    expected_size = int(meta["n"]) * dimension * 4
    if paths["points"].stat().st_size != expected_size:
        raise CampaignError(
            f"tamanho de {paths['points']} invalido: {paths['points'].stat().st_size} != {expected_size}"
        )
    expected_label_size = int(meta["n"]) * 4
    if paths["labels"].stat().st_size != expected_label_size:
        raise CampaignError(
            f"tamanho de {paths['labels']} invalido: "
            f"{paths['labels'].stat().st_size} != {expected_label_size}"
        )
    dimension_match = re.search(r"_([1-9][0-9]*)d$", dataset)
    if dimension_match is None or int(dimension_match.group(1)) != dimension:
        raise CampaignError(f"{paths['meta']}: dimensao nao corresponde ao nome {dataset!r}")
    points = np.memmap(paths["points"], dtype=np.float32, mode="r")
    try:
        if not np.isfinite(points).all():
            raise CampaignError(f"{paths['points']}: pontos contem NaN/Inf")
    finally:
        del points
    return {
        "dataset": dataset,
        "n": n,
        "d": int(meta["d"]),
        "seed": int(meta["seed"]),
        "eps_pool": eps,
        "minpts_pool": pool,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "sha256": {
            "points": real_points,
            "labels": real_labels,
            "meta": sha256_file(paths["meta"]),
        },
        "metadata": meta,
    }


def all_datasets(spec: dict[str, Any]) -> list[tuple[str, int]]:
    return sorted({dataset_key(case) for case in spec["cases"]})


def validate_all_inputs(spec: dict[str, Any], data_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        key: validate_dataset(data_dir, key[0], key[1], spec["protocol"])
        for key in all_datasets(spec)
    }


def resolved_cases(
    spec: dict[str, Any], datasets: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    resolved = []
    for case in spec["cases"]:
        dataset = datasets[dataset_key(case)]
        try:
            eps = [dataset["eps_pool"][i] for i in case["eps_indices"]]
            minimums = [dataset["minpts_pool"][i] for i in case["minpts_indices"]]
        except IndexError as exc:
            raise CampaignError(f"{case['id']}: indice de grade fora do pool") from exc
        if eps != sorted(set(eps)) or minimums != sorted(set(minimums)):
            raise CampaignError(f"{case['id']}: grade resolvida deve ser unica e ordenada")
        resolved.append(
            {
                **case,
                "d": dataset["d"],
                "seed": dataset["seed"],
                "eps": eps,
                "min_samples": minimums,
                "k": len(eps),
                "l": len(minimums),
                "configuration_count": len(eps) * len(minimums),
                "index": case.get("index", spec["protocol"]["index"]),
                "tier": case.get("tier", "pilot" if spec["phase"] == "pilot" else "core"),
                "dataset_paths": dataset["paths"],
                "dataset_sha256": dataset["sha256"],
            }
        )
    return resolved


def case_methods(case: dict[str, Any]) -> list[tuple[str, str]]:
    return (
        [(METHOD_MULTI, route) for route in case["routes"]]
        + [(METHOD_SEQUENTIAL, "not-applicable"), (METHOD_CUML, "not-applicable")]
    )


def plan_counts(spec: dict[str, Any]) -> dict[str, Any]:
    samples = int(spec["protocol"]["measured_samples"])
    method_variants = sum(len(case["routes"]) + 2 for case in spec["cases"])
    measured_records = method_variants * samples
    measured_fits_per_block = 0
    for case in spec["cases"]:
        configurations = len(case["eps_indices"]) * len(case["minpts_indices"])
        measured_fits_per_block += len(case["routes"]) + 2 * configurations
    warmup = int(spec["protocol"]["warmup"])
    return {
        "campaign_id": spec["id"],
        "phase": spec["phase"],
        "cases": len(spec["cases"]),
        "multi_route_variants_per_block": sum(len(case["routes"]) for case in spec["cases"]),
        "method_variants_per_block": method_variants,
        "measured_samples_per_method": samples,
        "symmetric_block_pairs": samples // 2,
        "raw_measured_records": measured_records,
        "underlying_measured_dbscan_fits": measured_fits_per_block * samples,
        "underlying_warmup_dbscan_fits": measured_fits_per_block * samples * warmup,
        "underlying_total_dbscan_fits": measured_fits_per_block * samples * (warmup + 1),
    }


def prepare_datasets(spec: dict[str, Any], data_dir: Path) -> list[dict[str, Any]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset, n in all_datasets(spec):
        paths = dataset_paths(data_dir, dataset, n)
        present = [path.exists() for path in paths.values()]
        if all(present):
            results.append(validate_dataset(data_dir, dataset, n, spec["protocol"]))
            continue
        if any(present):
            raise CampaignError(
                f"dataset parcial em {data_dir}: recusei sobrescrever {dataset} N={n}"
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset}_n{n}-", dir=data_dir))
        try:
            command = [
                sys.executable,
                str(ROOT / "tools" / "gerar_datasets.py"),
                "--dataset",
                dataset,
                "--n",
                str(n),
                "--seed",
                str(spec["protocol"]["seed"]),
                "--eps-quantis",
                ",".join(str(value) for value in spec["protocol"]["eps_quantiles"]),
                "--out-dir",
                str(staging),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode:
                raise CampaignError(
                    f"gerador falhou para {dataset} N={n}:\n{completed.stdout}\n{completed.stderr}"
                )
            staged = validate_dataset(staging, dataset, n, spec["protocol"])
            for key, target in paths.items():
                source = Path(staged["paths"][key])
                os.replace(source, target)
            results.append(validate_dataset(data_dir, dataset, n, spec["protocol"]))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return results


def binary_identity(binary: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), "--build-info"], text=True, capture_output=True, check=False, timeout=30
    )
    if completed.returncode:
        raise CampaignError(f"--build-info falhou: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        build = payload["build"]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise CampaignError(f"--build-info invalido: {completed.stdout!r}") from exc
    revision = str(build.get("git_sha", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise CampaignError(f"build.git_sha deve identificar snapshot com 40 hex: {revision!r}")
    if build.get("configured_backend") != "cuvs":
        raise CampaignError("campanha oficial exige binario configurado com backend cuvs")
    return {
        "binary": str(binary.resolve()),
        "binary_sha256": sha256_file(binary),
        "build": build,
        "cuda": payload.get("cuda"),
    }


def _gate_artifact_path(
    descriptor: Any, matrix_path: Path, label: str
) -> Path:
    """Resolve evidence emitted beside the matrix, including a relative ``--out``."""

    if not isinstance(descriptor, dict):
        raise CampaignError(f"{label}: descritor de artefato ausente")
    rendered, expected_sha = descriptor.get("path"), descriptor.get("sha256")
    if not isinstance(rendered, str) or not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha)):
        raise CampaignError(f"{label}: descritor de artefato invalido")
    raw = Path(rendered)
    candidates = [raw] if raw.is_absolute() else [
        matrix_path.resolve().parent / raw,
        matrix_path.resolve().parent / raw.name,
        ROOT / raw,
    ]
    artifact = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if artifact is None:
        raise CampaignError(f"{label}: artefato ausente: {rendered}")
    observed_sha = sha256_file(artifact)
    if observed_sha != expected_sha:
        raise CampaignError(f"{label}: SHA-256 diverge: {observed_sha} != {expected_sha}")
    return artifact


def validate_gate(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Valida toda a cobertura oficial; nenhum PASS agregado e aceito isoladamente."""

    gate = read_json(path)
    if not isinstance(gate, dict) or gate.get("schema_version") != 1:
        raise CampaignError("matriz semantica possui schema_version invalido")
    if gate.get("validation_approved") is not True:
        raise CampaignError("matriz semantica nao possui validation_approved=true")
    if gate.get("binary_sha256") != identity["binary_sha256"]:
        raise CampaignError(
            "matriz semantica pertence a outro binario: "
            f"{gate.get('binary_sha256')} != {identity['binary_sha256']}"
        )
    expected_script_sha = sha256_file(ROOT / "tools" / "run_validation_matrix.py")
    if gate.get("matrix_script_sha256") != expected_script_sha:
        raise CampaignError("matriz semantica foi gerada por outro run_validation_matrix.py")
    official_counts = {
        "random_seeds": OFFICIAL_GATE_RANDOM_SEEDS,
        "repetitions": OFFICIAL_GATE_REPETITIONS,
        "oracle_max_n": OFFICIAL_GATE_ORACLE_MAX_N,
    }
    wrong_counts = [
        f"{field}={gate.get(field)!r}, esperado {expected!r}"
        for field, expected in official_counts.items()
        if gate.get(field) != expected
    ]
    if wrong_counts:
        raise CampaignError("cobertura da matriz semantica incompleta: " + "; ".join(wrong_counts))
    expected_protocol = {
        "metric": "euclidean",
        "cuml_algorithm": "brute",
        "config_order": "eps_major",
    }
    if gate.get("protocol") != expected_protocol:
        raise CampaignError("protocolo da matriz semantica diverge do protocolo oficial")
    if not isinstance(gate.get("environment"), dict):
        raise CampaignError("matriz semantica nao contem environment")

    try:
        from tools.run_validation_matrix import adversarial_cases
    except ImportError:  # pragma: no cover - execucao direta a partir de tools/
        from run_validation_matrix import adversarial_cases

    expected_cases = adversarial_cases(OFFICIAL_GATE_RANDOM_SEEDS)
    cases = gate.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CampaignError("matriz semantica nao contem casos")
    expected_names = [str(case["name"]) for case in expected_cases]
    observed_names = [case.get("name") if isinstance(case, dict) else None for case in cases]
    if observed_names != expected_names or len(set(observed_names)) != len(observed_names):
        raise CampaignError(
            f"cobertura de casos da matriz diverge: {observed_names!r} != {expected_names!r}"
        )

    mode_contracts = {
        "cuvs_auto_i32": ("cuvs", "int32", "auto"),
        "cuvs_annotated_i32": ("cuvs", "int32", "annotated"),
        "cuvs_dense_i32": ("cuvs", "int32", "dense"),
        "cuvs_auto_i64": ("cuvs", "int64", "auto"),
        "codes_i32": ("codes", "int32", "auto"),
        "codes_i64": ("codes", "int64", "auto"),
        "cuvs_batched": ("cuvs", "int32", "auto"),
        "codes_batched": ("codes", "int32", "auto"),
    }
    expected_backends = sorted((*OFFICIAL_GATE_MODES, "cuml"))
    expected_label_arrays = sorted(
        [
            f"labels_{mode}_rep{repetition}"
            for mode in OFFICIAL_GATE_MODES
            for repetition in range(
                OFFICIAL_GATE_REPETITIONS if mode in OFFICIAL_GATE_REPEATED_MODES else 1
            )
        ]
        + [f"labels_cuml_rep{repetition}" for repetition in range(OFFICIAL_GATE_REPETITIONS)]
    )
    expected_pair_names = {
        f"{left}_vs_{right}"
        for left_index, left in enumerate(expected_backends)
        for right in expected_backends[left_index + 1 :]
    }
    revision = identity["build"]["git_sha"]
    for case_index, (case, expected_case) in enumerate(zip(cases, expected_cases)):
        label = f"matriz cases[{case_index}] {expected_case['name']}"
        required_true = ("approved", "deterministic", "batched_ok", "routes_ok")
        if any(case.get(field) is not True for field in required_true):
            raise CampaignError(f"{label}: caso nao esta integralmente aprovado/deterministico")
        if case.get("determinism_failures") != [] or case.get("failure_artifacts") != []:
            raise CampaignError(f"{label}: caso aprovado ainda contem artefatos/falhas")

        points = np.ascontiguousarray(expected_case["points"], dtype=np.float32)
        eps = sorted({float(value) for value in expected_case["eps"]})
        minimums = sorted({int(value) for value in expected_case["min_samples"]})
        expected_case_metadata = {
            "n": int(points.shape[0]),
            "d": int(points.shape[1]),
            "eps": eps,
            "min_samples": minimums,
            "points_sha256": hashlib.sha256(points.tobytes(order="C")).hexdigest(),
        }
        metadata_mismatches = [
            field
            for field, expected in expected_case_metadata.items()
            if case.get(field) != expected
        ]
        if metadata_mismatches:
            raise CampaignError(f"{label}: metadata divergente em {metadata_mismatches}")

        executions = case.get("executions")
        if not isinstance(executions, dict) or set(executions) != set(OFFICIAL_GATE_MODES):
            raise CampaignError(f"{label}: cobertura de modos incompleta")
        for mode in OFFICIAL_GATE_MODES:
            execution = executions[mode]
            repetitions = (
                OFFICIAL_GATE_REPETITIONS if mode in OFFICIAL_GATE_REPEATED_MODES else 1
            )
            if not isinstance(execution, dict) or execution.get("repetitions") != repetitions:
                raise CampaignError(f"{label}: repetitions invalido para {mode}")
            result = execution.get("result")
            if not isinstance(result, dict):
                raise CampaignError(f"{label}: resultado ausente para {mode}")
            backend, index, route = mode_contracts[mode]
            expected_result = {
                "backend": backend,
                "index": index,
                "requested_route": route,
                "configuration_count": len(eps) * len(minimums),
                "config_order": "eps_major",
                "n": int(points.shape[0]),
                "d": int(points.shape[1]),
                "eps": eps,
                "min_samples": minimums,
            }
            divergent = [
                field for field, expected in expected_result.items() if result.get(field) != expected
            ]
            if divergent:
                raise CampaignError(f"{label}: contrato runtime de {mode} diverge em {divergent}")
            if ((result.get("build") or {}).get("git_sha")) != revision:
                raise CampaignError(f"{label}: revisao ausente/divergente em {mode}")
            runtime_execution = result.get("execution")
            if not isinstance(runtime_execution, dict):
                raise CampaignError(f"{label}: telemetria de execucao ausente em {mode}")
            if mode == "cuvs_annotated_i32" and not (
                int(runtime_execution.get("annotated_batches", 0)) > 0
                and int(runtime_execution.get("dense_batches", -1)) == 0
            ):
                raise CampaignError(f"{label}: rota annotated nao foi coberta")
            if mode == "cuvs_dense_i32" and not (
                int(runtime_execution.get("dense_batches", 0)) > 0
                and int(runtime_execution.get("annotated_batches", -1)) == 0
            ):
                raise CampaignError(f"{label}: rota dense nao foi coberta")
            if expected_case.get("must_batch") and mode in {"cuvs_batched", "codes_batched"}:
                if int(runtime_execution.get("batches", 0)) <= 1:
                    raise CampaignError(f"{label}: batching forcado nao foi coberto por {mode}")

        validation = case.get("validation")
        if not isinstance(validation, dict) or validation.get("validacao_aprovada") is not True:
            raise CampaignError(f"{label}: validacao semantica nao aprovada")
        expected_validation_metadata = {
            "schema_version": 1,
            "config_order": "eps_major",
            "n": int(points.shape[0]),
            "d": int(points.shape[1]),
            "eps": eps,
            "min_samples": minimums,
            "backends": expected_backends,
        }
        divergent = [
            field
            for field, expected in expected_validation_metadata.items()
            if validation.get(field) != expected
        ]
        if divergent:
            raise CampaignError(f"{label}: cobertura semantica diverge em {divergent}")
        configurations = validation.get("configuracoes")
        expected_grid = [
            (config, epsilon, minimum)
            for config, (epsilon, minimum) in enumerate(
                (epsilon, minimum) for epsilon in eps for minimum in minimums
            )
        ]
        if not isinstance(configurations, list) or len(configurations) != len(expected_grid):
            raise CampaignError(f"{label}: grade semantica incompleta")
        for configuration, (config, epsilon, minimum) in zip(configurations, expected_grid):
            if not isinstance(configuration, dict) or any(
                (
                    configuration.get("config") != config,
                    configuration.get("eps") != epsilon,
                    configuration.get("min_samples") != minimum,
                    configuration.get("aprovada") is not True,
                )
            ):
                raise CampaignError(f"{label}: configuracao semantica {config} invalida")
            semantic = configuration.get("semantica")
            pairs = configuration.get("pares")
            if not isinstance(semantic, dict) or set(semantic) != set(expected_backends) or any(
                not isinstance(item, dict) or item.get("valido") is not True
                for item in semantic.values()
            ):
                raise CampaignError(f"{label}: cobertura de backends invalida na config {config}")
            if not isinstance(pairs, dict) or set(pairs) != expected_pair_names or any(
                not isinstance(item, dict) or item.get("valida") is not True
                for item in pairs.values()
            ):
                raise CampaignError(f"{label}: cobertura de pares invalida na config {config}")

        evidence = case.get("labels_evidence")
        evidence_path = _gate_artifact_path(evidence, path, f"{label}.labels_evidence")
        if evidence.get("config_order") != "eps_major" or evidence.get("arrays") != expected_label_arrays:
            raise CampaignError(f"{label}: indice de arrays do evidence esta incompleto")
        with np.load(evidence_path, allow_pickle=False) as archive:
            expected_npz_keys = {"points", "eps", "min_samples", *expected_label_arrays}
            if set(archive.files) != expected_npz_keys:
                raise CampaignError(f"{label}: arrays do NPZ divergem do contrato")
            if not np.array_equal(archive["points"], points):
                raise CampaignError(f"{label}: points no evidence divergem do caso oficial")
            if not np.array_equal(archive["eps"], np.asarray(eps, dtype=np.float64)):
                raise CampaignError(f"{label}: eps no evidence diverge")
            if not np.array_equal(
                archive["min_samples"], np.asarray(minimums, dtype=np.int64)
            ):
                raise CampaignError(f"{label}: min_samples no evidence diverge")
            expected_shape = (len(eps) * len(minimums), int(points.shape[0]))
            for array_name in expected_label_arrays:
                labels = archive[array_name]
                if labels.dtype != np.int32 or labels.shape != expected_shape:
                    raise CampaignError(f"{label}: shape/dtype invalido em {array_name}")
                if labels.size and (int(labels.min()) < -1 or int(labels.max()) >= points.shape[0]):
                    raise CampaignError(f"{label}: labels fora de [-1, N-1] em {array_name}")
            for source in (*OFFICIAL_GATE_REPEATED_MODES, "cuml"):
                if not np.array_equal(
                    archive[f"labels_{source}_rep0"], archive[f"labels_{source}_rep1"]
                ):
                    raise CampaignError(f"{label}: evidence nao deterministico para {source}")
    return gate


def package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unavailable"


def check_lockfile(path: Path) -> dict[str, Any]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise CampaignError(f"lockfile ausente ou vazio: {path}")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise CampaignError(f"pip freeze --all falhou: {completed.stderr}")
    expected = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observed = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
    if expected != observed:
        expected_set, observed_set = set(expected), set(observed)
        raise CampaignError(
            "ambiente diverge de requirements.lock.txt; "
            f"ausentes={sorted(expected_set - observed_set)[:5]} extras={sorted(observed_set - expected_set)[:5]}"
        )
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "lines": len(expected)}


def validate_artifact_descriptor(
    descriptor: dict[str, str] | None, base: Path, label: str
) -> Path:
    if not isinstance(descriptor, dict):
        raise CampaignError(f"{label}: descritor de artefato ausente")
    rendered = descriptor.get("path")
    expected = descriptor.get("sha256")
    if not isinstance(rendered, str) or not isinstance(expected, str):
        raise CampaignError(f"{label}: descritor de artefato invalido")
    path = Path(rendered)
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise CampaignError(f"{label}: artefato ausente: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise CampaignError(f"{label}: SHA-256 diverge: {observed} != {expected}")
    return path


def validate_pilot_manifest(path: Path, source_sha: str) -> dict[str, Any]:
    """Exige um PILOT completo, integro e pertencente ao mesmo snapshot da CORE."""

    manifest = read_json(path)
    validate_schema(manifest, "benchmark-run-manifest.schema.json")
    if manifest.get("phase") != "pilot" or manifest.get("status") != "completed":
        raise CampaignError("manifesto PILOT deve ter phase=pilot e status=completed")
    if manifest.get("failures"):
        raise CampaignError("manifesto PILOT contem falhas; CORE bloqueada")
    if (manifest.get("validation_gate") or {}).get("validation_approved") is not True:
        raise CampaignError("manifesto PILOT nao possui gate semantico aprovado")
    snapshot = manifest.get("snapshot") or {}
    if snapshot.get("source_tree_sha256") != source_sha:
        raise CampaignError(
            "PILOT pertence a outro snapshot: "
            f"{snapshot.get('source_tree_sha256')} != {source_sha}"
        )
    if snapshot.get("source_revision") != source_sha[:40]:
        raise CampaignError("source_revision do PILOT nao corresponde ao SHA-256 da arvore")
    planned, observed = manifest.get("planned") or {}, manifest.get("observed") or {}
    required_equal = ("cases", "method_runs", "warmups", "measured_samples")
    mismatches = [
        f"{field}: {observed.get(field)} != {planned.get(field)}"
        for field in required_equal
        if observed.get(field) != planned.get(field)
    ]
    if observed.get("valid") != planned.get("valid"):
        mismatches.append(
            f"valid: {observed.get('valid')} != valid planejado {planned.get('valid')}"
        )
    if observed.get("failed") != 0 or observed.get("semantic_rejected") != 0:
        mismatches.append(
            f"failed/semantic_rejected={observed.get('failed')}/{observed.get('semantic_rejected')}"
        )
    if mismatches:
        raise CampaignError("PILOT incompleto; CORE bloqueada:\n- " + "\n- ".join(mismatches))

    base = path.resolve().parent
    pilot_spec_path = validate_artifact_descriptor(
        manifest.get("campaign_spec"), base, "PILOT campaign_spec"
    )
    pilot_spec = load_spec(pilot_spec_path)
    if pilot_spec.get("phase") != "pilot" or pilot_spec.get("id") != manifest.get("campaign_id"):
        raise CampaignError("campaign_spec do PILOT diverge do manifesto")
    official_pilot = ROOT / "scripts" / "campaigns" / "pilot.json"
    if sha256_file(pilot_spec_path) != sha256_file(official_pilot):
        raise CampaignError("manifesto nao pertence ao spec PILOT oficial deste snapshot")

    artifacts = manifest.get("artifacts") or {}
    required_artifacts = (
        "environment_json",
        "cases_json",
        "raw_jsonl",
        "summary_json",
        "summary_csv",
        "stdout_log",
        "stderr_log",
        "validation_matrix",
        "lockfile",
        "dataset_hashes",
        "campaign_schema",
        "sample_schema",
        "run_manifest_schema",
    )
    artifact_paths = {
        name: validate_artifact_descriptor(
            artifacts.get(name), base, f"PILOT artifacts.{name}"
        )
        for name in required_artifacts
    }
    if artifacts.get("pilot_manifest") is not None:
        raise CampaignError("manifesto PILOT nao deve referenciar outro pilot_manifest")
    expected_locations = {
        "cases_json": base / "cases.json",
        "summary_json": base / "summaries" / "summary.json",
        "summary_csv": base / "summaries" / "summary.csv",
    }
    misplaced = [
        name
        for name, expected_path in expected_locations.items()
        if artifact_paths[name].resolve() != expected_path.resolve()
    ]
    if path.resolve() != (base / "manifest.json").resolve() or misplaced:
        raise CampaignError(f"PILOT usa artefatos fora das localizacoes canonicas: {misplaced}")
    current_schemas = {
        "campaign_schema": SCHEMAS / "benchmark-campaign.schema.json",
        "sample_schema": SCHEMAS / "benchmark-sample.schema.json",
        "run_manifest_schema": SCHEMAS / "benchmark-run-manifest.schema.json",
    }
    for name, current in current_schemas.items():
        if sha256_file(artifact_paths[name]) != sha256_file(current):
            raise CampaignError(f"PILOT artifacts.{name} diverge do schema deste snapshot")

    cases, context_manifest, samples = _aggregation_context(base)
    if context_manifest != manifest:
        raise CampaignError("manifest.json mudou durante a validacao do PILOT")
    spec_cases = pilot_spec["cases"]
    if len(cases) != len(spec_cases):
        raise CampaignError("cases.json nao cobre todos os casos do PILOT oficial")
    datasets = read_json(artifact_paths["dataset_hashes"])
    if not isinstance(datasets, list) or not datasets:
        raise CampaignError("dataset_hashes do PILOT deve conter lista nao vazia")
    dataset_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise CampaignError("dataset_hashes contem entrada invalida")
        key = (str(dataset.get("dataset")), int(dataset.get("n", 0)))
        if key in dataset_lookup:
            raise CampaignError(f"dataset_hashes contem entrada duplicada {key}")
        metadata_path = validate_artifact_descriptor(
            dataset.get("metadata_artifact"), base, f"PILOT dataset_hashes {key}.metadata"
        )
        if dataset.get("metadata") != read_json(metadata_path):
            raise CampaignError(f"dataset_hashes {key}: metadata capturada diverge")
        if (dataset.get("sha256") or {}).get("meta") != sha256_file(metadata_path):
            raise CampaignError(f"dataset_hashes {key}: hash da metadata diverge")
        dataset_lookup[key] = dataset
    for case_index, (case, spec_case) in enumerate(zip(cases, spec_cases)):
        static_mismatches = [
            field for field, expected in spec_case.items() if case.get(field) != expected
        ]
        expected_defaults = {
            "index": spec_case.get("index", pilot_spec["protocol"]["index"]),
            "tier": spec_case.get("tier", "pilot"),
            "seed": pilot_spec["protocol"]["seed"],
        }
        static_mismatches.extend(
            field
            for field, expected in expected_defaults.items()
            if case.get(field) != expected
        )
        dataset = dataset_lookup.get((spec_case["dataset"], spec_case["n"]))
        if dataset is None:
            raise CampaignError(f"{spec_case['id']}: dataset ausente em dataset_hashes")
        expected_eps = [dataset["eps_pool"][index] for index in spec_case["eps_indices"]]
        expected_minimums = [
            dataset["minpts_pool"][index] for index in spec_case["minpts_indices"]
        ]
        resolved_expected = {
            "d": dataset["d"],
            "dataset_paths": dataset["paths"],
            "dataset_sha256": dataset["sha256"],
            "eps": expected_eps,
            "min_samples": expected_minimums,
            "k": len(expected_eps),
            "l": len(expected_minimums),
            "configuration_count": len(expected_eps) * len(expected_minimums),
        }
        static_mismatches.extend(
            field
            for field, expected in resolved_expected.items()
            if case.get(field) != expected
        )
        if static_mismatches:
            raise CampaignError(
                f"cases[{case_index}] {spec_case['id']} diverge em {sorted(set(static_mismatches))}"
            )

    records = load_raw_records(base)
    complete_case_ids, excluded_cases = aggregation_completeness(base, records)
    if complete_case_ids != {case["id"] for case in cases} or excluded_cases:
        raise CampaignError(
            f"PILOT raw incompleto; completos={sorted(complete_case_ids)} excluidos={excluded_cases}"
        )
    observed_from_raw = {
        "cases": len(complete_case_ids),
        "method_runs": len(records),
        "warmups": sum(
            (
                1
                if record["method"] == METHOD_MULTI
                else int(record["parameters"]["configuration_count"])
            )
            * int(record["parameters"]["warmup_count"])
            for record in records
        ),
        "measured_samples": len(records),
        "valid": len(records),
        "failed": 0,
        "semantic_rejected": 0,
    }
    if manifest["observed"] != observed_from_raw:
        raise CampaignError(
            f"PILOT observed nao e reconstruivel dos raw: {manifest['observed']!r} "
            f"!= {observed_from_raw!r}"
        )
    records_dir = base / "raw" / "records"
    expected_jsonl = canonical_jsonl(records_dir).encode("utf-8")
    if artifact_paths["raw_jsonl"].read_bytes() != expected_jsonl:
        raise CampaignError("PILOT raw_jsonl nao e a serializacao canonica de raw/records")

    stored_summary = read_json(artifact_paths["summary_json"])
    with tempfile.TemporaryDirectory(prefix="dbm-pilot-rebuild-") as temporary:
        rebuilt_dir = Path(temporary)
        shutil.copy2(path, rebuilt_dir / "manifest.json")
        shutil.copy2(artifact_paths["cases_json"], rebuilt_dir / "cases.json")
        shutil.copytree(base / "raw", rebuilt_dir / "raw")
        rebuilt_summary = aggregate_records(rebuilt_dir)
        rebuilt_csv = (rebuilt_dir / "summaries" / "summary.csv").read_bytes()
    if not isinstance(stored_summary, dict) or not isinstance(stored_summary.get("generated_at"), str):
        raise CampaignError("PILOT summary_json invalido")
    comparable_stored = {key: value for key, value in stored_summary.items() if key != "generated_at"}
    comparable_rebuilt = {key: value for key, value in rebuilt_summary.items() if key != "generated_at"}
    if comparable_stored != comparable_rebuilt:
        raise CampaignError("PILOT summary_json nao corresponde aos raw reagrupados")
    if artifact_paths["summary_csv"].read_bytes() != rebuilt_csv:
        raise CampaignError("PILOT summary_csv nao corresponde aos raw reagrupados")
    if (
        stored_summary.get("completeness_checked") is not True
        or stored_summary.get("excluded_cases") != []
        or stored_summary.get("raw_records") != len(records)
        or stored_summary.get("included_raw_records") != len(records)
    ):
        raise CampaignError("PILOT summary nao comprova completude integral")
    return manifest


def parse_nvcc_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvcc", "--version"], text=True, capture_output=True, check=False, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)+)", completed.stdout + completed.stderr)
    return match.group(1) if match else None


def query_sample_telemetry(gpu_identifier: str | None) -> dict[str, Any] | None:
    """Telemetria best-effort da GPU alocada, identificada por UUID/PCI, nunca pela linha 0."""

    if not gpu_identifier:
        return None
    fields = (
        "temperature.gpu,power.draw,clocks.sm,clocks.mem,"
        "utilization.gpu,memory.used"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_identifier}",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        if completed.returncode:
            return None
        values = [float(value.strip()) for value in completed.stdout.strip().split(",")]
        if len(values) != 6 or any(not math.isfinite(value) for value in values):
            return None
        return {
            "timestamp": now_utc(),
            "temperature_c": values[0],
            "power_w": values[1],
            "sm_clock_mhz": values[2],
            "memory_clock_mhz": values[3],
            "utilization_percent": values[4],
            "memory_used_mib": values[5],
        }
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def run_worker_process(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Executa um worker isolado e encerra todo o grupo se ele exceder o prazo."""

    use_process_group = os.name == "posix"
    try:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=use_process_group,
        )
    except OSError as exc:
        raise CampaignError(f"nao foi possivel iniciar worker: {exc}") from exc
    try:
        stdout, stderr = process.communicate(
            timeout=METHOD_TIMEOUT_SECONDS + WORKER_SHUTDOWN_GRACE_SECONDS
        )
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        if use_process_group:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - campanha GPU e executada em Linux
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            if use_process_group:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:  # pragma: no cover - campanha GPU e executada em Linux
                process.kill()
            stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(
            command,
            124,
            stdout or "",
            (stderr or "") + f"\nworker excedeu timeout de {METHOD_TIMEOUT_SECONDS} s",
        )


def normalized_environment(gpu: dict[str, Any]) -> dict[str, Any]:
    return {
        "gpu_model": str(gpu.get("name") or "unavailable"),
        "gpu_uuid": gpu.get("uuid"),
        "compute_capability": str(gpu.get("compute_capability") or "0.0"),
        "vram_bytes": int(gpu.get("total_memory_bytes") or 0),
        "driver_version": str(
            gpu.get("nvidia_driver_version")
            or gpu.get("cuda_driver_version")
            or "unavailable"
        ),
        "cuda_runtime_version": str(gpu.get("cuda_runtime_version") or "unavailable"),
        "cuda_toolkit_version": parse_nvcc_version(),
        "cuml_version": package_version("cuml-cu12", "cuml"),
        "cuvs_version": package_version("libcuvs-cu12", "libcuvs"),
        "raft_version": package_version("libraft-cu12", "libraft"),
        "rmm_version": package_version("librmm-cu12", "librmm"),
        "hostname": platform.node() or "unavailable",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run_binary_once(
    *,
    binary: Path,
    input_path: Path,
    labels_path: Path,
    n: int,
    d: int,
    eps: list[float],
    minimums: list[int],
    backend: str,
    index: str,
    neigh_per_row: int,
    budget: int,
    warmup: int,
    route: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float, str, str]:
    command = [
        str(binary), "--input", str(input_path), "--output", str(labels_path),
        "--n", str(n), "--d", str(d),
        "--eps", ",".join(f"{value:.10g}" for value in eps),
        "--min-samples", ",".join(str(value) for value in minimums),
        "--repeat", "1", "--warmup", str(warmup), "--backend", backend,
        "--index", index, "--route", route, "--json",
        "--max-mbytes-per-batch", str(budget),
    ]
    if neigh_per_row:
        command += ["--neigh-per-row", str(neigh_per_row)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(1.0, float(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        raise CampaignError(
            f"binario excedeu timeout de {float(timeout_seconds):.1f} s"
        ) from exc
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode:
        raise CampaignError(
            f"binario saiu com {completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
        )
    try:
        runtime = json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise CampaignError(f"saida JSON ausente/invalida: {completed.stdout!r}") from exc
    expected_cfg = len(eps) * len(minimums)
    checks = {
        "backend": backend,
        "index": index,
        "requested_route": route,
        "configuration_count": expected_cfg,
        "n": n,
        "d": d,
        "repeat": 1,
        "warmup": warmup,
    }
    for field, expected in checks.items():
        if runtime.get(field) != expected:
            raise CampaignError(f"runtime.{field}={runtime.get(field)!r}; esperado {expected!r}")
    runtime_eps = np.asarray(runtime.get("eps", []), dtype=np.float64)
    expected_eps = np.asarray(
        [np.float32(float(f"{value:.10g}")) for value in eps], dtype=np.float64
    )
    if runtime_eps.shape != expected_eps.shape or not np.allclose(
        runtime_eps, expected_eps, rtol=5e-6, atol=1e-12
    ):
        raise CampaignError(f"runtime.eps={runtime.get('eps')!r}; esperado {eps!r}")
    if runtime.get("min_samples") != minimums:
        raise CampaignError(
            f"runtime.min_samples={runtime.get('min_samples')!r}; esperado {minimums!r}"
        )
    if runtime.get("neigh_per_row") != neigh_per_row:
        raise CampaignError(
            f"runtime.neigh_per_row={runtime.get('neigh_per_row')!r}; esperado {neigh_per_row}"
        )
    if runtime.get("max_bytes_per_batch") != budget * 1_000_000:
        raise CampaignError(
            f"runtime.max_bytes_per_batch={runtime.get('max_bytes_per_batch')!r}; "
            f"esperado {budget * 1_000_000}"
        )
    timings = runtime.get("fit_ms_all")
    if (
        not isinstance(timings, list)
        or len(timings) != 1
        or not isinstance(timings[0], (int, float))
        or not math.isfinite(float(timings[0]))
        or float(timings[0]) <= 0
    ):
        raise CampaignError(f"runtime.fit_ms_all invalido: {timings!r}")
    execution = runtime.get("execution")
    if not isinstance(execution, dict) or execution.get("stats_scope") != "last_measured_repeat":
        raise CampaignError("runtime.execution ausente ou stats_scope inesperado")
    batches = execution.get("batches")
    batch_routes = execution.get("batch_routes")
    if not isinstance(batches, int) or batches < 1:
        raise CampaignError(f"runtime.execution.batches invalido: {batches!r}")
    if not isinstance(batch_routes, list) or len(batch_routes) != batches:
        raise CampaignError(
            f"runtime.execution.batch_routes tem {len(batch_routes) if isinstance(batch_routes, list) else 'tipo invalido'}; "
            f"esperado {batches}"
        )
    dense_count = sum(value == "dense" for value in batch_routes)
    annotated_count = sum(value == "annotated" for value in batch_routes)
    not_applicable_count = sum(value == "not-applicable" for value in batch_routes)
    if any(value not in {"dense", "annotated", "not-applicable"} for value in batch_routes):
        raise CampaignError(f"rotas por lote invalidas: {batch_routes!r}")
    if execution.get("dense_batches") != dense_count or execution.get("annotated_batches") != annotated_count:
        raise CampaignError("contadores de rota divergem de batch_routes")
    multiparameter_cuvs = backend == "cuvs" and len(eps) > 1
    observed_route = execution.get("route_observed")
    if not multiparameter_cuvs:
        if not_applicable_count != batches or observed_route != "not-applicable":
            raise CampaignError("execucao escalar/codes deve registrar rota not-applicable")
    else:
        if dense_count + annotated_count != batches:
            raise CampaignError("execucao Multi-cuVS deve classificar todos os lotes")
        expected_observed = (
            "mixed"
            if dense_count and annotated_count
            else "dense" if dense_count else "annotated"
        )
        if observed_route != expected_observed:
            raise CampaignError(
                f"route_observed={observed_route!r}; derivado de batch_routes={expected_observed!r}"
            )
        if route in {"dense", "annotated"} and (
            observed_route != route or any(value != route for value in batch_routes)
        ):
            raise CampaignError(f"rota forcada {route!r} nao foi respeitada")
    if not labels_path.is_file() or labels_path.stat().st_size != n * expected_cfg * 4:
        raise CampaignError(f"arquivo de labels ausente ou com tamanho invalido: {labels_path}")
    return runtime, wall_ms, completed.stdout, completed.stderr


def labels_summary(labels_path: Path, n: int, configuration_count: int) -> tuple[np.ndarray, list[int], list[int]]:
    labels = np.fromfile(labels_path, dtype=np.int32)
    if labels.size != n * configuration_count:
        raise CampaignError(f"labels possui {labels.size} inteiros; esperado {n * configuration_count}")
    labels = labels.reshape(configuration_count, n)
    if np.any(labels < -1) or np.any(labels >= n):
        raise CampaignError("labels contem valor fora do contrato [-1, N-1]")
    clusters = [int(np.unique(row[row >= 0]).size) for row in labels]
    noise = [int(np.count_nonzero(row < 0)) for row in labels]
    return labels, clusters, noise


def worker_multi(args: argparse.Namespace) -> dict[str, Any]:
    runtime, wall_ms, stdout, stderr = run_binary_once(
        binary=args.binary,
        input_path=args.input,
        labels_path=args.labels,
        n=args.n,
        d=args.d,
        eps=args.eps,
        minimums=args.min_samples,
        backend=args.backend,
        index=args.index,
        neigh_per_row=args.neigh_per_row,
        budget=args.max_mbytes_per_batch,
        warmup=args.warmup,
        route=args.route,
        timeout_seconds=METHOD_TIMEOUT_SECONDS,
    )
    labels, clusters, noise = labels_summary(
        args.labels, args.n, len(args.eps) * len(args.min_samples)
    )
    return {
        "fit_ms": float(runtime["fit_ms_all"][0]),
        "configuration_fit_ms": [],
        "setup_ms": None,
        "h2d_ms": None,
        "internal_setup_ms": None,
        "d2h_ms": None,
        "end_to_end_ms": wall_ms,
        "runtime": runtime,
        "clusters": clusters,
        "noise": noise,
        "labels_sha256": sha256_file(args.labels),
        "stdout": stdout,
        "stderr": stderr,
    }


def worker_sequential(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = time.monotonic() + METHOD_TIMEOUT_SECONDS
    runtimes: list[dict[str, Any]] = []
    timings = []
    label_rows = []
    stdout_parts, stderr_parts = [], []
    with tempfile.TemporaryDirectory(prefix="dbm-sequential-") as directory:
        for config, (eps, minimum) in enumerate(
            (pair for eps in args.eps for pair in ((eps, minimum) for minimum in args.min_samples))
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                raise CampaignError(
                    f"sweep experimental excedeu timeout de {METHOD_TIMEOUT_SECONDS} s"
                )
            path = Path(directory) / f"labels-{config:03d}.i32"
            runtime, _, stdout, stderr = run_binary_once(
                binary=args.binary,
                input_path=args.input,
                labels_path=path,
                n=args.n,
                d=args.d,
                eps=[eps],
                minimums=[minimum],
                backend=args.backend,
                index=args.index,
                neigh_per_row=args.neigh_per_row,
                budget=args.max_mbytes_per_batch,
                warmup=args.warmup,
                route="auto",
                timeout_seconds=remaining,
            )
            value = float(runtime["fit_ms_all"][0])
            timings.append(
                {"config_index": config, "eps": eps, "min_samples": minimum, "fit_ms": value}
            )
            runtimes.append(runtime)
            label_rows.append(np.fromfile(path, dtype=np.int32))
            stdout_parts.append(stdout)
            stderr_parts.append(stderr)
    labels = np.stack(label_rows)
    temporary = args.labels.with_name(f".{args.labels.name}.tmp-{os.getpid()}")
    labels.tofile(temporary)
    os.replace(temporary, args.labels)
    _, clusters, noise = labels_summary(args.labels, args.n, len(timings))
    return {
        "fit_ms": float(sum(item["fit_ms"] for item in timings)),
        "configuration_fit_ms": timings,
        "setup_ms": None,
        "h2d_ms": None,
        "internal_setup_ms": None,
        "d2h_ms": None,
        "end_to_end_ms": (time.perf_counter() - started) * 1000.0,
        "runtime": {"scalar_executions": runtimes},
        "clusters": clusters,
        "noise": noise,
        "labels_sha256": sha256_file(args.labels),
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
    }


def worker_cuml(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    load_started = time.perf_counter()
    points = np.fromfile(args.input, dtype=np.float32)
    if points.size != args.n * args.d:
        raise CampaignError(f"input possui {points.size} floats; esperado {args.n * args.d}")
    points = points.reshape(args.n, args.d)
    setup_ms = (time.perf_counter() - load_started) * 1000.0

    import cupy as cp
    from cuml.cluster import DBSCAN

    h2d_start, h2d_end = cp.cuda.Event(), cp.cuda.Event()
    h2d_start.record()
    points_gpu = cp.asarray(points)
    h2d_end.record()
    h2d_end.synchronize()
    h2d_ms = float(cp.cuda.get_elapsed_time(h2d_start, h2d_end))

    timings = []
    label_rows = []
    d2h_ms = 0.0
    for config, (eps, minimum) in enumerate(
        (pair for eps in args.eps for pair in ((eps, minimum) for minimum in args.min_samples))
    ):
        model = DBSCAN(
            eps=float(eps),
            min_samples=int(minimum),
            algorithm="brute",
            metric="euclidean",
            calc_core_sample_indices=False,
            output_type="cupy",
            max_mbytes_per_batch=int(args.max_mbytes_per_batch),
        )
        for _ in range(args.warmup):
            model.fit(points_gpu)
        cp.cuda.Stream.null.synchronize()
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
        model.fit(points_gpu)
        end.record()
        end.synchronize()
        value = float(cp.cuda.get_elapsed_time(begin, end))
        timings.append(
            {"config_index": config, "eps": eps, "min_samples": minimum, "fit_ms": value}
        )
        copy_start, copy_end = cp.cuda.Event(), cp.cuda.Event()
        copy_start.record()
        labels = cp.asnumpy(model.labels_).astype(np.int32, copy=False)
        copy_end.record()
        copy_end.synchronize()
        d2h_ms += float(cp.cuda.get_elapsed_time(copy_start, copy_end))
        label_rows.append(labels)
    labels = np.stack(label_rows)
    temporary = args.labels.with_name(f".{args.labels.name}.tmp-{os.getpid()}")
    labels.tofile(temporary)
    os.replace(temporary, args.labels)
    _, clusters, noise = labels_summary(args.labels, args.n, len(timings))
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties.get("name")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return {
        "fit_ms": float(sum(item["fit_ms"] for item in timings)),
        "configuration_fit_ms": timings,
        "setup_ms": setup_ms,
        "h2d_ms": h2d_ms,
        "internal_setup_ms": None,
        "d2h_ms": d2h_ms,
        "end_to_end_ms": (time.perf_counter() - started) * 1000.0,
        "runtime": {
            "implementation": "cuml.cluster.DBSCAN",
            "algorithm": "brute",
            "metric": "euclidean",
            "gpu": str(name),
            "cuml_version": package_version("cuml-cu12", "cuml"),
        },
        "clusters": clusters,
        "noise": noise,
        "labels_sha256": sha256_file(args.labels),
        "stdout": "",
        "stderr": "",
    }


def worker_main(args: argparse.Namespace) -> int:
    try:
        if args.method == METHOD_MULTI:
            result = worker_multi(args)
        elif args.method == METHOD_SEQUENTIAL:
            result = worker_sequential(args)
        else:
            result = worker_cuml(args)
        atomic_write_json(args.output, result)
        return 0
    except Exception as exc:
        atomic_write_json(
            args.output,
            {"worker_failed": True, "error_type": type(exc).__name__, "message": str(exc)},
        )
        print(f"erro worker {args.method}: {exc}", file=sys.stderr)
        return 2


def routes_per_batch(execution: dict[str, Any]) -> list[dict[str, Any]]:
    raw = execution.get("batch_routes", []) if isinstance(execution, dict) else []
    return [
        {"batch": index, "route": str(route), "nnz_max_eps": None}
        for index, route in enumerate(raw)
    ]


def execution_record(
    method: str,
    worker: dict[str, Any],
    n: int,
    runtime_artifact: dict[str, str],
) -> dict[str, Any]:
    if method == METHOD_MULTI:
        execution = worker["runtime"].get("execution") or {}
        total_nnz = execution.get("total_nnz_max_eps")
        density = (
            float(total_nnz) / float(n * n)
            if isinstance(total_nnz, int) and n > 0
            else None
        )
        return {
            "requested_budget_bytes": int(worker["runtime"]["max_bytes_per_batch"]),
            "effective_budget_bytes": execution.get("effective_max_bytes_per_batch"),
            "batch_size": execution.get("batch_size"),
            "batches": execution.get("batches"),
            "attempts": execution.get("attempts"),
            "batch_corrections": execution.get("batch_corrections"),
            "route_observed": execution.get("route_observed"),
            "routes_per_batch": routes_per_batch(execution),
            "configuration_executions": [],
            "max_nnz": execution.get("max_nnz"),
            "total_nnz_max_eps": total_nnz,
            "density_max_eps": density,
            "peak_device_memory_bytes": None,
            "runtime_artifact": runtime_artifact,
            "telemetry_before": worker.get("telemetry_before"),
            "telemetry_after": worker.get("telemetry_after"),
        }
    scalar_runtimes = (worker.get("runtime") or {}).get("scalar_executions", [])
    per_configuration = []
    configuration_count = len(worker.get("configuration_fit_ms", []))
    for config_index in range(configuration_count):
        runtime = scalar_runtimes[config_index] if config_index < len(scalar_runtimes) else {}
        execution = runtime.get("execution") or {}
        total_nnz = execution.get("total_nnz_max_eps")
        per_configuration.append(
            {
                "config_index": config_index,
                "effective_budget_bytes": execution.get("effective_max_bytes_per_batch"),
                "batch_size": execution.get("batch_size"),
                "batches": execution.get("batches"),
                "attempts": execution.get("attempts"),
                "batch_corrections": execution.get("batch_corrections"),
                "route_observed": execution.get("route_observed", "not-applicable"),
                "max_nnz": execution.get("max_nnz"),
                "total_nnz_max_eps": total_nnz,
                "density_max_eps": (
                    float(total_nnz) / float(n * n)
                    if isinstance(total_nnz, int) and n > 0
                    else None
                ),
            }
        )
    return {
        "requested_budget_bytes": None if method == METHOD_CUML else None,
        "effective_budget_bytes": None,
        "batch_size": None,
        "batches": None,
        "attempts": None,
        "batch_corrections": None,
        "route_observed": "not-applicable",
        "routes_per_batch": [],
        "configuration_executions": per_configuration,
        "max_nnz": None,
        "total_nnz_max_eps": None,
        "density_max_eps": None,
        "peak_device_memory_bytes": None,
        "runtime_artifact": runtime_artifact,
        "telemetry_before": worker.get("telemetry_before"),
        "telemetry_after": worker.get("telemetry_after"),
    }


def canonicalize(labels: np.ndarray) -> np.ndarray:
    result = np.full(labels.shape, -1, dtype=np.int32)
    mapping: dict[int, int] = {}
    for index, raw in enumerate(labels):
        value = int(raw)
        if value < 0:
            continue
        if value not in mapping:
            mapping[value] = len(mapping)
        result[index] = mapping[value]
    return result


def labels_identical(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    return all(np.array_equal(canonicalize(a), canonicalize(b)) for a, b in zip(left, right))


def artifact_descriptor(path: Path | None, base: Path) -> dict[str, str] | None:
    if path is None or not path.is_file():
        return None
    resolved = path.resolve()
    try:
        rendered = resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def canonical_jsonl(records_dir: Path) -> str:
    lines = []
    for path in sorted(records_dir.glob("*.json")):
        value = read_json(path)
        lines.append(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines) + ("\n" if lines else "")


def rebuild_jsonl(records_dir: Path, destination: Path) -> None:
    atomic_write_text(destination, canonical_jsonl(records_dir))


def sample_record(
    *,
    spec: dict[str, Any],
    run_id: str,
    case: dict[str, Any],
    method: str,
    route: str,
    block_index: int,
    order_index: int,
    repetition: int,
    identity: dict[str, Any],
    source_sha: str,
    environment: dict[str, Any],
    worker: dict[str, Any],
    runtime_artifact: dict[str, str],
) -> dict[str, Any]:
    execution = execution_record(method, worker, case["n"], runtime_artifact)
    # O mesmo budget e solicitado nos dois lados, embora o cuML nao exponha seu lote efetivo.
    execution["requested_budget_bytes"] = int(spec["protocol"]["max_mbytes_per_batch"]) * 1_000_000
    sample_id = f"{case['id']}.b{block_index:02d}.p{order_index:02d}.{method}"
    if method == METHOD_MULTI:
        sample_id += f".{route}"
    record = {
        "$schema": SAMPLE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": spec["id"],
        "run_id": run_id,
        "case_id": case["id"],
        "sample_id": sample_id,
        "phase": spec["phase"],
        "method": method,
        "sample_kind": "measured",
        "status": "ok",
        "block_index": block_index,
        "pair_index": block_index // 2,
        "block_direction": "forward" if block_index % 2 == 0 else "reverse",
        "order_index": order_index,
        "repetition": repetition,
        "recorded_at": now_utc(),
        "identity": {
            "source_revision": identity["build"]["git_sha"],
            "revision_kind": identity["build"]["revision_kind"],
            "source_tree_sha256": source_sha,
            "source_dirty": identity["build"].get("git_dirty"),
            "binary_sha256": identity["binary_sha256"],
            "build_id": identity["build"]["build_id"],
            "dataset_sha256": case["dataset_sha256"]["points"],
            "dataset_metadata_sha256": case["dataset_sha256"]["meta"],
        },
        "parameters": {
            "dataset": case["dataset"],
            "n": case["n"],
            "d": case["d"],
            "seed": case["seed"],
            "backend": "cuml" if method == METHOD_CUML else spec["protocol"]["backend"],
            "route_requested": route if method == METHOD_MULTI else "not-applicable",
            "index": (
                "implementation-default"
                if method == METHOD_CUML
                else case["index"]
            ),
            "neigh_per_row": spec["protocol"]["neigh_per_row"],
            "max_mbytes_per_batch": spec["protocol"]["max_mbytes_per_batch"],
            "warmup_count": spec["protocol"]["warmup"],
            "k": case["k"],
            "l": case["l"],
            "configuration_count": case["configuration_count"],
            "tier": case["tier"],
            "eps": case["eps"],
            "min_samples": case["min_samples"],
            "config_order": "eps_major",
            "precision": "float32",
            "metric": "L2",
        },
        "environment": environment,
        "timings": {
            "boundary": TIMING_BOUNDARY,
            "setup_ms": worker["setup_ms"],
            "h2d_ms": worker["h2d_ms"],
            "internal_setup_ms": worker["internal_setup_ms"],
            "fit_ms": worker["fit_ms"],
            "d2h_ms": worker["d2h_ms"],
            "end_to_end_ms": worker["end_to_end_ms"],
            "configuration_fit_ms": worker["configuration_fit_ms"],
        },
        "execution": execution,
        "result": {
            "clusters": worker["clusters"],
            "noise": worker["noise"],
            "validation_status": "approved-snapshot",
        },
        "error": None,
    }
    validate_schema(record, "benchmark-sample.schema.json")
    return record


def worker_command(
    *,
    output: Path,
    labels: Path,
    method: str,
    route: str,
    case: dict[str, Any],
    spec: dict[str, Any],
    binary: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--method", method,
        "--route", route,
        "--binary", str(binary),
        "--input", case["dataset_paths"]["points"],
        "--labels", str(labels),
        "--output", str(output),
        "--n", str(case["n"]),
        "--d", str(case["d"]),
        "--eps", ",".join(f"{value:.17g}" for value in case["eps"]),
        "--min-samples", ",".join(str(value) for value in case["min_samples"]),
        "--backend", spec["protocol"]["backend"],
        "--index", case["index"],
        "--neigh-per-row", str(spec["protocol"]["neigh_per_row"]),
        "--max-mbytes-per-batch", str(spec["protocol"]["max_mbytes_per_batch"]),
        "--warmup", str(spec["protocol"]["warmup"]),
    ]


def preserve_failure_labels(
    campaign_dir: Path,
    case_id: str,
    block_index: int,
    labels: dict[str, np.ndarray],
    context: dict[str, Any],
) -> Path:
    target = campaign_dir / "labels" / f"{case_id}.b{block_index:02d}.failure.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}.npz")
    arrays = {f"labels_{key}": value for key, value in labels.items()}
    arrays["context_json_utf8"] = np.frombuffer(
        json.dumps(context, ensure_ascii=False, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)
    return target


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def bootstrap_ci(values: list[float], key: str, iterations: int = 10_000) -> dict[str, Any]:
    if not values:
        return {"method": "percentile-bootstrap-median", "iterations": iterations, "low": None, "high": None}
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    generator = random.Random(seed)
    medians = []
    for _ in range(iterations):
        sample = [values[generator.randrange(len(values))] for _ in values]
        medians.append(float(statistics.median(sample)))
    return {
        "method": "percentile-bootstrap-median",
        "iterations": iterations,
        "confidence": 0.95,
        "seed": seed,
        "low": percentile(medians, 2.5),
        "high": percentile(medians, 97.5),
        "conclusive": len(values) >= 5,
    }


def descriptive(values: list[float], key: str) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    p25, p75 = percentile(values, 25), percentile(values, 75)
    return {
        "n": len(values),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "stddev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
        "confidence_interval": bootstrap_ci(values, key),
    }


def load_raw_records(campaign_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((campaign_dir / "raw" / "records").glob("*.json")):
        record = read_json(path)
        validate_schema(record, "benchmark-sample.schema.json")
        records.append(record)
    return records


def _aggregation_context(
    campaign_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Carrega os dois artefatos autoritativos e valida seus eixos declarados."""

    cases_path = campaign_dir / "cases.json"
    manifest_path = campaign_dir / "manifest.json"
    missing = [str(path) for path in (cases_path, manifest_path) if not path.is_file()]
    if missing:
        raise CampaignError(
            "aggregate oficial exige cases.json e manifest.json; ausentes: " + ", ".join(missing)
        )
    cases = read_json(cases_path)
    manifest = read_json(manifest_path)
    if not isinstance(cases, list) or not cases:
        raise CampaignError("cases.json deve conter uma lista nao vazia")
    validate_schema(manifest, "benchmark-run-manifest.schema.json")
    protocol = manifest["protocol"]
    samples = protocol["measured_samples"]
    if samples % 2:
        raise CampaignError("manifesto symmetric exige protocol.measured_samples par")
    expected_methods = [METHOD_MULTI, METHOD_SEQUENTIAL, METHOD_CUML]
    if protocol["methods"] != expected_methods:
        raise CampaignError(
            f"manifesto possui eixo methods {protocol['methods']!r}; esperado {expected_methods!r}"
        )

    required_case_fields = {
        "id",
        "dataset",
        "n",
        "d",
        "seed",
        "routes",
        "eps",
        "min_samples",
        "k",
        "l",
        "configuration_count",
        "index",
        "tier",
        "dataset_paths",
        "dataset_sha256",
    }
    case_ids: list[str] = []
    expected_method_runs = 0
    expected_warmups = 0
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise CampaignError(f"cases[{case_index}] deve ser objeto")
        missing_fields = sorted(required_case_fields - set(case))
        if missing_fields:
            raise CampaignError(f"cases[{case_index}] omite metadata {missing_fields}")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id:
            raise CampaignError(f"cases[{case_index}].id invalido")
        case_ids.append(case_id)
        routes = case["routes"]
        if (
            not isinstance(routes, list)
            or not routes
            or len(routes) != len(set(routes))
            or "auto" not in routes
            or any(route not in {"annotated", "dense", "auto"} for route in routes)
        ):
            raise CampaignError(f"{case_id}: eixo routes invalido: {routes!r}")
        eps = case["eps"]
        minimums = case["min_samples"]
        if (
            not isinstance(eps, list)
            or not eps
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in eps)
            or any(not math.isfinite(float(value)) or float(value) <= 0 for value in eps)
            or eps != sorted(set(eps))
        ):
            raise CampaignError(f"{case_id}: eixo eps invalido")
        if (
            not isinstance(minimums, list)
            or not minimums
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in minimums)
            or minimums != sorted(set(minimums))
        ):
            raise CampaignError(f"{case_id}: eixo min_samples invalido")
        expected_grid = (len(eps), len(minimums), len(eps) * len(minimums))
        if (case["k"], case["l"], case["configuration_count"]) != expected_grid:
            raise CampaignError(f"{case_id}: k/l/configuration_count divergem da grade")
        if (
            isinstance(case["n"], bool)
            or not isinstance(case["n"], int)
            or case["n"] < 1
            or isinstance(case["d"], bool)
            or not isinstance(case["d"], int)
            or case["d"] < 1
        ):
            raise CampaignError(f"{case_id}: n/d invalidos")
        if case["index"] not in {"int32", "int64"} or case["tier"] not in {
            "pilot",
            "core",
            "stress",
        }:
            raise CampaignError(f"{case_id}: index/tier invalidos")
        dataset_paths = case["dataset_paths"]
        dataset_hashes = case["dataset_sha256"]
        if not isinstance(dataset_paths, dict) or any(
            not isinstance(dataset_paths.get(name), str) or not dataset_paths.get(name)
            for name in ("points", "labels", "meta")
        ):
            raise CampaignError(f"{case_id}: dataset_paths incompleto")
        if not isinstance(dataset_hashes, dict) or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(dataset_hashes.get(name, "")))
            for name in ("points", "labels", "meta")
        ):
            raise CampaignError(f"{case_id}: dataset_sha256 incompleto")
        method_variants = len(routes) + 2
        expected_method_runs += method_variants * samples
        expected_warmups += (
            len(routes) + 2 * case["configuration_count"]
        ) * samples * protocol["warmup"]
    if len(case_ids) != len(set(case_ids)):
        raise CampaignError("cases.json contem IDs duplicados")

    expected_planned = {
        "cases": len(cases),
        "method_runs": expected_method_runs,
        "warmups": expected_warmups,
        "measured_samples": expected_method_runs,
        "valid": expected_method_runs,
        "failed": 0,
        "semantic_rejected": 0,
    }
    if manifest["planned"] != expected_planned:
        raise CampaignError(
            f"manifesto planned diverge dos eixos de cases.json: {manifest['planned']!r} "
            f"!= {expected_planned!r}"
        )
    return cases, manifest, samples


def _validate_record_axes(
    record: dict[str, Any],
    case: dict[str, Any],
    case_index: int,
    manifest: dict[str, Any],
    samples: int,
    campaign_dir: Path,
) -> None:
    """Confere relacoes que JSON Schema isolado nao consegue expressar."""

    sample_id = str(record.get("sample_id"))
    block = record["block_index"]
    if block >= samples:
        raise CampaignError(f"{sample_id}: block_index fora do eixo oficial")
    expected_axis = {
        "sample_kind": "measured",
        "pair_index": block // 2,
        "block_direction": "forward" if block % 2 == 0 else "reverse",
        "repetition": block,
    }
    divergent_axis = [
        field for field, expected in expected_axis.items() if record.get(field) != expected
    ]
    if divergent_axis:
        raise CampaignError(f"{sample_id}: eixos de bloco divergem em {divergent_axis}")

    methods = case_methods(case)
    rotation = (case_index + block // 2) % len(methods)
    forward = methods[rotation:] + methods[:rotation]
    order = forward if block % 2 == 0 else list(reversed(forward))
    order_index = record["order_index"]
    if order_index >= len(order):
        raise CampaignError(f"{sample_id}: order_index fora do eixo oficial")
    expected_method, expected_route = order[order_index]
    route = record["parameters"]["route_requested"]
    if (record["method"], route) != (expected_method, expected_route):
        raise CampaignError(
            f"{sample_id}: ordem nao corresponde ao desenho rotacionado/reverso"
        )
    expected_sample_id = f"{case['id']}.b{block:02d}.p{order_index:02d}.{expected_method}"
    if expected_method == METHOD_MULTI:
        expected_sample_id += f".{expected_route}"
    if sample_id != expected_sample_id:
        raise CampaignError(f"{sample_id}: sample_id canonico esperado {expected_sample_id}")

    snapshot = manifest["snapshot"]
    identity = record["identity"]
    expected_identity = {
        "source_revision": snapshot["source_revision"],
        "revision_kind": snapshot["revision_kind"],
        "source_tree_sha256": snapshot["source_tree_sha256"],
        "source_dirty": snapshot["source_dirty"],
        "binary_sha256": snapshot["binary_sha256"],
        "build_id": snapshot["build_id"],
        "dataset_sha256": case["dataset_sha256"]["points"],
        "dataset_metadata_sha256": case["dataset_sha256"]["meta"],
    }
    identity_mismatches = [
        field for field, expected in expected_identity.items() if identity.get(field) != expected
    ]
    if identity_mismatches:
        raise CampaignError(f"{sample_id}: identidade diverge em {identity_mismatches}")

    protocol = manifest["protocol"]
    parameters = record["parameters"]
    expected_parameters = {
        "dataset": case["dataset"],
        "n": case["n"],
        "d": case["d"],
        "seed": case["seed"],
        "backend": "cuml" if expected_method == METHOD_CUML else protocol["backend"],
        "route_requested": expected_route,
        "index": "implementation-default" if expected_method == METHOD_CUML else case["index"],
        "neigh_per_row": protocol["neigh_per_row"],
        "max_mbytes_per_batch": protocol["max_mbytes_per_batch"],
        "warmup_count": protocol["warmup"],
        "k": case["k"],
        "l": case["l"],
        "configuration_count": case["configuration_count"],
        "tier": case["tier"],
        "eps": case["eps"],
        "min_samples": case["min_samples"],
        "config_order": "eps_major",
        "precision": "float32",
        "metric": "L2",
    }
    parameter_mismatches = [
        field
        for field, expected in expected_parameters.items()
        if parameters.get(field) != expected
    ]
    if parameter_mismatches:
        raise CampaignError(f"{sample_id}: parametros divergem em {parameter_mismatches}")
    if record["timings"]["boundary"] != protocol["timing_boundary"]:
        raise CampaignError(f"{sample_id}: timing boundary divergente")
    configuration_grid = [
        (config, epsilon, minimum)
        for config, (epsilon, minimum) in enumerate(
            (epsilon, minimum)
            for epsilon in case["eps"]
            for minimum in case["min_samples"]
        )
    ]
    configuration_timings = record["timings"]["configuration_fit_ms"]
    configuration_executions = record["execution"]["configuration_executions"]
    if expected_method == METHOD_MULTI:
        if configuration_timings != [] or configuration_executions != []:
            raise CampaignError(f"{sample_id}: Multi nao deve conter fits escalares")
    else:
        observed_grid = [
            (item.get("config_index"), item.get("eps"), item.get("min_samples"))
            for item in configuration_timings
        ]
        if observed_grid != configuration_grid:
            raise CampaignError(f"{sample_id}: grade de tempos escalares incompleta")
        if [item.get("config_index") for item in configuration_executions] != [
            item[0] for item in configuration_grid
        ]:
            raise CampaignError(f"{sample_id}: grade de execucoes escalares incompleta")
        fit_sum = sum(float(item["fit_ms"]) for item in configuration_timings)
        if not math.isclose(
            fit_sum, float(record["timings"]["fit_ms"]), rel_tol=1e-12, abs_tol=1e-9
        ):
            raise CampaignError(f"{sample_id}: fit_ms nao e a soma da grade escalar")
    configuration_count = case["configuration_count"]
    if len(record["result"]["clusters"]) != configuration_count or len(
        record["result"]["noise"]
    ) != configuration_count:
        raise CampaignError(f"{sample_id}: resultado nao cobre toda a grade")
    expected_budget = protocol["max_mbytes_per_batch"] * 1_000_000
    if record["execution"]["requested_budget_bytes"] != expected_budget:
        raise CampaignError(f"{sample_id}: budget solicitado diverge do manifesto")
    validate_artifact_descriptor(
        record["execution"]["runtime_artifact"], campaign_dir, f"{sample_id}.runtime_artifact"
    )


def aggregation_completeness(
    campaign_dir: Path,
    records: list[dict[str, Any]],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Retorna os casos completos depois de validar metadados e todos os eixos raw."""

    cases, manifest, samples = _aggregation_context(campaign_dir)
    case_lookup = {case["id"]: (index, case) for index, case in enumerate(cases)}

    actual_by_case: dict[str, set[tuple[int, str, str]]] = {}
    invalid_record_cases: set[str] = set()
    expected_record_identity = {
        "campaign_id": manifest.get("campaign_id"),
        "run_id": manifest.get("id"),
        "phase": manifest.get("phase"),
        "source_tree_sha256": (manifest.get("snapshot") or {}).get("source_tree_sha256"),
        "binary_sha256": (manifest.get("snapshot") or {}).get("binary_sha256"),
    }
    for record in records:
        case_entry = case_lookup.get(record["case_id"])
        if case_entry is not None:
            case_index, case = case_entry
            _validate_record_axes(record, case, case_index, manifest, samples, campaign_dir)
        observed_identity = {
            "campaign_id": record.get("campaign_id"),
            "run_id": record.get("run_id"),
            "phase": record.get("phase"),
            "source_tree_sha256": (record.get("identity") or {}).get("source_tree_sha256"),
            "binary_sha256": (record.get("identity") or {}).get("binary_sha256"),
        }
        mismatched = [
            field
            for field, expected in expected_record_identity.items()
            if expected is not None and observed_identity.get(field) != expected
        ]
        if mismatched:
            raise CampaignError(
                f"{record.get('sample_id')}: identidade diverge do manifesto em {mismatched}"
            )
        if record.get("status") != "ok" or (record.get("result") or {}).get(
            "validation_status"
        ) != "checked-identical":
            invalid_record_cases.add(str(record["case_id"]))
        actual_by_case.setdefault(record["case_id"], set()).add(
            (
                int(record["block_index"]),
                str(record["method"]),
                str(record["parameters"]["route_requested"]),
            )
        )
    failed_cases = {
        failure.get("case_id")
        for failure in manifest.get("failures", [])
        if isinstance(failure, dict) and failure.get("case_id") not in (None, "campaign")
    }
    complete: set[str] = set()
    excluded: list[dict[str, Any]] = []
    declared_ids: set[str] = set()
    for case in cases:
        case_id = str(case["id"])
        declared_ids.add(case_id)
        methods = (
            [(METHOD_MULTI, route) for route in case["routes"]]
            + [(METHOD_SEQUENTIAL, "not-applicable"), (METHOD_CUML, "not-applicable")]
        )
        expected = {
            (block, method, route)
            for block in range(samples)
            for method, route in methods
        }
        actual = actual_by_case.get(case_id, set())
        missing, extra = expected - actual, actual - expected
        if (
            not missing
            and not extra
            and case_id not in failed_cases
            and case_id not in invalid_record_cases
        ):
            complete.add(case_id)
            continue
        excluded.append(
            {
                "case_id": case_id,
                "reason": (
                    "failure-recorded"
                    if case_id in failed_cases
                    else "invalid-record"
                    if case_id in invalid_record_cases
                    else "incomplete-or-extra"
                ),
                "expected_records": len(expected),
                "observed_records": len(actual),
                "missing": [list(item) for item in sorted(missing)],
                "extra": [list(item) for item in sorted(extra)],
            }
        )
    for case_id in sorted(set(actual_by_case) - declared_ids):
        excluded.append(
            {
                "case_id": case_id,
                "reason": "undeclared-case",
                "expected_records": 0,
                "observed_records": len(actual_by_case[case_id]),
                "missing": [],
                "extra": [list(item) for item in sorted(actual_by_case[case_id])],
            }
        )
    return complete, excluded


def aggregate_records(campaign_dir: Path) -> dict[str, Any]:
    all_records = load_raw_records(campaign_dir)
    seen: set[tuple[str, int, str, str]] = set()
    for record in all_records:
        key = (
            str(record["case_id"]),
            int(record["block_index"]),
            str(record["method"]),
            str(record["parameters"]["route_requested"]),
        )
        if key in seen:
            raise CampaignError(f"registro raw duplicado para {key}")
        seen.add(key)
    complete_case_ids, excluded_cases = aggregation_completeness(campaign_dir, all_records)
    records = [record for record in all_records if record["case_id"] in complete_case_ids]
    by_case_block: dict[tuple[str, int], dict[tuple[str, str], dict[str, Any]]] = {}
    case_info: dict[str, dict[str, Any]] = {}
    for record in records:
        route = record["parameters"]["route_requested"]
        parameters = record["parameters"]
        current = {
            "dataset": parameters["dataset"],
            "n": parameters["n"],
            "d": parameters["d"],
            "k": parameters["k"],
            "l": parameters["l"],
            "configuration_count": parameters["configuration_count"],
            "max_eps": max(parameters["eps"]),
            "tier": parameters["tier"],
            "index": None if parameters["index"] == "implementation-default" else parameters["index"],
        }
        previous = case_info.setdefault(record["case_id"], current)
        for field in ("dataset", "n", "d", "k", "l", "configuration_count", "max_eps", "tier"):
            if previous[field] != current[field]:
                raise CampaignError(f"{record['case_id']}: parametro {field} inconsistente nos raw")
        if current["index"] is not None:
            if previous["index"] not in (None, current["index"]):
                raise CampaignError(f"{record['case_id']}: index inconsistente nos raw")
            previous["index"] = current["index"]
        by_case_block.setdefault((record["case_id"], record["block_index"]), {})[
            (record["method"], route)
        ] = record

    component_values: dict[tuple[str, str], list[float]] = {}
    component_densities: dict[tuple[str, str], list[float]] = {}
    ratio_values: dict[tuple[str, str], list[float]] = {}
    ratio_rows = []
    for (case_id, block), methods in sorted(by_case_block.items()):
        for (method, route), record in methods.items():
            name = f"{method}:{route}"
            component_values.setdefault((case_id, name), []).append(record["timings"]["fit_ms"])
            density = record["execution"].get("density_max_eps")
            if density is not None:
                component_densities.setdefault((case_id, name), []).append(float(density))
        sequential = methods.get((METHOD_SEQUENTIAL, "not-applicable"))
        cuml = methods.get((METHOD_CUML, "not-applicable"))
        multis = {route: record for (method, route), record in methods.items() if method == METHOD_MULTI}
        for route, multi in multis.items():
            if sequential:
                ratio = sequential["timings"]["fit_ms"] / multi["timings"]["fit_ms"]
                ratio_values.setdefault((case_id, f"ganho_multi_puro:{route}"), []).append(ratio)
                ratio_rows.append({"case_id": case_id, "block_index": block, "metric": "ganho_multi_puro", "route": route, "value": ratio})
                configuration_count = (
                    len(multi["parameters"]["eps"])
                    * len(multi["parameters"]["min_samples"])
                )
                efficiency = ratio / configuration_count
                ratio_values.setdefault(
                    (case_id, f"efficiency_per_configuration:{route}"), []
                ).append(efficiency)
                ratio_rows.append(
                    {
                        "case_id": case_id,
                        "block_index": block,
                        "metric": "efficiency_per_configuration",
                        "route": route,
                        "value": efficiency,
                    }
                )
            if cuml and multi["parameters"]["index"] == "int32":
                ratio = cuml["timings"]["fit_ms"] / multi["timings"]["fit_ms"]
                ratio_values.setdefault((case_id, f"speedup_vs_cuml:{route}"), []).append(ratio)
                ratio_rows.append({"case_id": case_id, "block_index": block, "metric": "speedup_vs_cuml", "route": route, "value": ratio})
        if "annotated" in multis and "dense" in multis:
            ratio = multis["dense"]["timings"]["fit_ms"] / multis["annotated"]["timings"]["fit_ms"]
            ratio_values.setdefault((case_id, "annotated_vs_dense"), []).append(ratio)
            ratio_rows.append({"case_id": case_id, "block_index": block, "metric": "annotated_vs_dense", "route": "annotated", "value": ratio})
            if "auto" in multis:
                best = min(multis["annotated"]["timings"]["fit_ms"], multis["dense"]["timings"]["fit_ms"])
                efficiency = best / multis["auto"]["timings"]["fit_ms"]
                ratio_values.setdefault((case_id, "auto_efficiency"), []).append(efficiency)
                ratio_rows.append({"case_id": case_id, "block_index": block, "metric": "auto_efficiency", "route": "auto", "value": efficiency})
                if sequential:
                    best_gain = sequential["timings"]["fit_ms"] / best
                    ratio_values.setdefault((case_id, "ganho_multi_puro:best_forced"), []).append(best_gain)
                    ratio_rows.append({"case_id": case_id, "block_index": block, "metric": "ganho_multi_puro", "route": "best_forced", "value": best_gain})
                    any_multi = multis["annotated"]
                    configuration_count = (
                        len(any_multi["parameters"]["eps"])
                        * len(any_multi["parameters"]["min_samples"])
                    )
                    best_efficiency = best_gain / configuration_count
                    ratio_values.setdefault(
                        (case_id, "efficiency_per_configuration:best_forced"), []
                    ).append(best_efficiency)
                    ratio_rows.append(
                        {
                            "case_id": case_id,
                            "block_index": block,
                            "metric": "efficiency_per_configuration",
                            "route": "best_forced",
                            "value": best_efficiency,
                        }
                    )

    # A ordem forward/reverse existe para contrabalancear deriva temporal. As duas
    # razoes positivas do mesmo pair_index formam uma unica unidade inferencial;
    # combine-as geometricamente antes de qualquer resumo ou bootstrap.
    pair_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in ratio_rows:
        block = int(row["block_index"])
        pair_index = block // 2
        row["pair_index"] = pair_index
        row["block_direction"] = "forward" if block % 2 == 0 else "reverse"
        metric_key = str(row["metric"])
        if metric_key in (
            "ganho_multi_puro",
            "speedup_vs_cuml",
            "efficiency_per_configuration",
        ):
            metric_key = f"{metric_key}:{row['route']}"
        pair_groups.setdefault((str(row["case_id"]), metric_key, pair_index), []).append(row)

    ratio_values = {}
    paired_ratio_rows = []
    for (case_id, metric_key, pair_index), rows in sorted(pair_groups.items()):
        directions = {str(row["block_direction"]) for row in rows}
        if len(rows) != 2 or directions != {"forward", "reverse"}:
            raise CampaignError(
                f"{case_id}/{metric_key}/pair {pair_index}: esperado um bloco forward e um reverse"
            )
        values = [float(row["value"]) for row in rows]
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise CampaignError(
                f"{case_id}/{metric_key}/pair {pair_index}: razao nao positiva ou nao finita"
            )
        pair_value = math.exp(statistics.fmean(math.log(value) for value in values))
        ratio_values.setdefault((case_id, metric_key), []).append(pair_value)
        metric, separator, route = metric_key.partition(":")
        paired_ratio_rows.append(
            {
                "case_id": case_id,
                "pair_index": pair_index,
                "metric": metric,
                "route": route if separator else rows[0]["route"],
                "value": pair_value,
                "forward_value": next(
                    float(row["value"])
                    for row in rows
                    if row["block_direction"] == "forward"
                ),
                "reverse_value": next(
                    float(row["value"])
                    for row in rows
                    if row["block_direction"] == "reverse"
                ),
                "reduction": "geometric-mean",
            }
        )

    components = [
        {
            "case_id": case,
            **case_info[case],
            "component": component,
            "fit_ms": descriptive(values, f"{case}:{component}:fit"),
            "density_max_eps": descriptive(
                component_densities.get((case, component), []),
                f"{case}:{component}:density",
            ),
        }
        for (case, component), values in sorted(component_values.items())
    ]
    ratios = [
        {
            "case_id": case,
            **case_info[case],
            "metric": metric,
            "paired_ratio": descriptive(values, f"{case}:{metric}:ratio"),
        }
        for (case, metric), values in sorted(ratio_values.items())
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_utc(),
        "estimator": "ratio por bloco; media geometrica forward/reverse por pair_index; mediana entre pares",
        "outlier_policy": "nenhuma amostra removida",
        "bootstrap": "percentile bootstrap deterministico da mediana sobre pair_index, 10000 reamostragens; n<5 pares nao conclusivo",
        "timing_boundary": TIMING_BOUNDARY,
        "block_design": "ordem rotacionada em bloco forward e ordem exatamente reversa no bloco seguinte",
        "measurement_limitations": {
            "experimental_setup_h2d_d2h": "nao expostos separadamente pelo binario; end_to_end_ms inclui processo, leitura, transferencias e labels",
            "cuml_effective_batching": "cuml.cluster.DBSCAN recebe o mesmo budget, mas nao expoe lote efetivo",
            "peak_device_memory_bytes": "null: nenhuma coleta nao intrusiva confiavel disponivel",
            "per_batch_nnz": "null: runtime expõe rota por lote e nnz maximo/total agregados",
            "warmup_samples": "executados no mesmo worker e descartados; contagem no manifesto, sem tempo individual",
            "method_timeout_seconds": METHOD_TIMEOUT_SECONDS,
        },
        "metric_definitions": {
            "ganho_multi_puro": "experimental_sequential.fit_ms / multi.fit_ms",
            "speedup_vs_cuml": "cuml_sequential.fit_ms / multi.fit_ms; comparacao de implementacao/API, nao speedup algoritmico puro",
            "annotated_vs_dense": "dense.fit_ms / annotated.fit_ms; >1 favorece annotated",
            "auto_efficiency": "min(annotated.fit_ms,dense.fit_ms) / auto.fit_ms; proximo de 1 e melhor",
            "efficiency_per_configuration": "ganho_multi_puro / (k*l); metrica auxiliar",
            "best_forced": "minimo annotated/dense escolhido no mesmo bloco; diagnostico pos-selecao, nao inferencia principal",
        },
        "raw_records": len(all_records),
        "included_raw_records": len(records),
        "completeness_checked": True,
        "excluded_cases": excluded_cases,
        "pooled_across_cases": False,
        "components": components,
        "ratios": ratios,
        "block_ratio_rows": ratio_rows,
        "paired_ratio_rows": paired_ratio_rows,
    }
    summaries = campaign_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summaries / "summary.json", summary)
    csv_path = summaries / "summary.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    fields = ["case_id", "dataset", "n", "d", "k", "l", "configuration_count", "max_eps", "tier", "index", "kind", "name", "sample_n", "median", "mean", "stddev", "min", "max", "p25", "p75", "iqr", "ci95_low", "ci95_high", "conclusive"]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        statistic_fields = ("median", "mean", "stddev", "min", "max", "p25", "p75", "iqr")
        for item in components:
            stats = item["fit_ms"]
            context = {key: item.get(key) for key in ("case_id", "dataset", "n", "d", "k", "l", "configuration_count", "max_eps", "tier", "index")}
            writer.writerow({**context, "kind": "fit_ms", "name": item["component"], "sample_n": stats.get("n"), **{key: stats.get(key) for key in statistic_fields}, "ci95_low": stats.get("confidence_interval", {}).get("low"), "ci95_high": stats.get("confidence_interval", {}).get("high"), "conclusive": stats.get("confidence_interval", {}).get("conclusive")})
        for item in ratios:
            stats = item["paired_ratio"]
            context = {key: item.get(key) for key in ("case_id", "dataset", "n", "d", "k", "l", "configuration_count", "max_eps", "tier", "index")}
            writer.writerow({**context, "kind": "paired_ratio", "name": item["metric"], "sample_n": stats.get("n"), **{key: stats.get(key) for key in statistic_fields}, "ci95_low": stats.get("confidence_interval", {}).get("low"), "ci95_high": stats.get("confidence_interval", {}).get("high"), "conclusive": stats.get("confidence_interval", {}).get("conclusive")})
    os.replace(temporary, csv_path)
    return summary


def finalize_logs(campaign_dir: Path, stdout_log: Path, stderr_log: Path) -> dict[str, Any]:
    """Anexa hashes dos logs depois que o processo principal fechou as redirecoes."""

    campaign_dir = campaign_dir.resolve()
    manifest_path = campaign_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") == "running":
        raise CampaignError("manifesto ainda esta running; logs nao podem ser finalizados")
    manifest["artifacts"]["stdout_log"] = artifact_descriptor(stdout_log, campaign_dir)
    manifest["artifacts"]["stderr_log"] = artifact_descriptor(stderr_log, campaign_dir)
    if manifest["artifacts"]["stdout_log"] is None or manifest["artifacts"]["stderr_log"] is None:
        raise CampaignError("logs stdout/stderr ausentes")
    validate_schema(manifest, "benchmark-run-manifest.schema.json")
    atomic_write_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "stdout_log": manifest["artifacts"]["stdout_log"],
        "stderr_log": manifest["artifacts"]["stderr_log"],
    }


def run_campaign(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    if spec["phase"] == "core" and (not args.allow_core or args.pilot_manifest is None):
        raise CampaignError(
            "CORE bloqueada: exige --allow-core e --pilot-manifest de um PILOT completo"
        )
    datasets = validate_all_inputs(spec, args.data_dir)
    cases = resolved_cases(spec, datasets)
    identity = binary_identity(args.binary)
    source_sha = source_tree_sha256()
    if identity["build"]["git_sha"] != source_sha[:40]:
        raise CampaignError(
            f"binario pertence a {identity['build']['git_sha']}, arvore atual e {source_sha[:40]}"
        )
    pilot_manifest = (
        validate_pilot_manifest(args.pilot_manifest, source_sha)
        if spec["phase"] == "core"
        else None
    )
    gate = validate_gate(args.validation_matrix, identity)
    lock = check_lockfile(args.lockfile)
    gpu_info = read_json(args.gpu_info)
    environment = normalized_environment(gpu_info)
    if environment["gpu_model"] != "NVIDIA A100-SXM4-80GB" or environment["compute_capability"] != "8.0":
        raise CampaignError(f"GPU oficial inesperada: {environment['gpu_model']} CC {environment['compute_capability']}")
    if environment["vram_bytes"] < 80_000 * 1024 * 1024:
        raise CampaignError(f"VRAM insuficiente para classe A100-80GB: {environment['vram_bytes']}")

    campaign_dir = args.campaign_dir.resolve()
    records_dir = campaign_dir / "raw" / "records"
    raw_jsonl = campaign_dir / "raw" / f"{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl"
    reserved_outputs = (
        campaign_dir / "manifest.json",
        campaign_dir / "cases.json",
        campaign_dir / "inputs",
        campaign_dir / "schemas",
        campaign_dir / "datasets" / "hashes.json",
        campaign_dir / "raw",
        campaign_dir / "labels",
        campaign_dir / "summaries",
    )
    occupied = [
        str(path)
        for path in reserved_outputs
        if path.is_file() or (path.is_dir() and any(path.iterdir()))
    ]
    if occupied:
        raise CampaignError(
            "campanha ja iniciada; recusei sobrescrever: " + ", ".join(occupied)
        )
    for directory in (
        records_dir,
        campaign_dir / "labels",
        campaign_dir / "summaries",
        campaign_dir / "datasets" / "metadata",
        campaign_dir / "inputs",
        campaign_dir / "schemas",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    runtime_dir = campaign_dir / "raw" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(campaign_dir / "cases.json", cases)

    campaign_spec_copy = campaign_dir / "inputs" / "campaign-spec.json"
    lockfile_copy = campaign_dir / "inputs" / "requirements.lock.txt"
    validation_matrix_copy = campaign_dir / "inputs" / "validation-matrix.json"
    atomic_write_bytes(campaign_spec_copy, args.spec.read_bytes())
    atomic_write_bytes(lockfile_copy, args.lockfile.read_bytes())
    atomic_write_bytes(validation_matrix_copy, args.validation_matrix.read_bytes())
    schema_copies = {}
    for name in (
        "benchmark-campaign.schema.json",
        "benchmark-sample.schema.json",
        "benchmark-run-manifest.schema.json",
    ):
        destination = campaign_dir / "schemas" / name
        atomic_write_bytes(destination, (SCHEMAS / name).read_bytes())
        schema_copies[name] = destination
    pilot_manifest_copy = None
    if pilot_manifest is not None:
        pilot_manifest_copy = campaign_dir / "inputs" / "pilot-manifest.json"
        atomic_write_bytes(pilot_manifest_copy, args.pilot_manifest.read_bytes())

    dataset_hashes = []
    for dataset in datasets.values():
        metadata_source = Path(dataset["paths"]["meta"])
        metadata_copy = campaign_dir / "datasets" / "metadata" / metadata_source.name
        atomic_write_bytes(metadata_copy, metadata_source.read_bytes())
        metadata_artifact = artifact_descriptor(metadata_copy, campaign_dir)
        dataset_hashes.append({**dataset, "metadata_artifact": metadata_artifact})
    dataset_hashes_path = campaign_dir / "datasets" / "hashes.json"
    atomic_write_json(dataset_hashes_path, dataset_hashes)
    lock = {
        **lock,
        "path": "inputs/requirements.lock.txt",
        "sha256": sha256_file(lockfile_copy),
    }
    enriched_environment = {
        **gpu_info,
        "benchmark_environment": environment,
        "python": platform.python_version(),
        "packages": {
            "numpy": package_version("numpy"),
            "cupy": package_version("cupy-cuda12x", "cupy"),
            "cuml": environment["cuml_version"],
            "cuvs": environment["cuvs_version"],
            "raft": environment["raft_version"],
            "rmm": environment["rmm_version"],
        },
        "lockfile": lock,
        "captured_at": now_utc(),
    }
    atomic_write_json(campaign_dir / "environment.json", enriched_environment)

    plan = plan_counts(spec)
    run_id = f"{spec['phase']}-job-{os.environ.get('SLURM_JOB_ID', 'local')}-{identity['binary_sha256'][:12]}"
    created = now_utc()
    empty_counts = {"cases": 0, "method_runs": 0, "warmups": 0, "measured_samples": 0, "valid": 0, "failed": 0, "semantic_rejected": 0}
    planned_counts = {
        "cases": plan["cases"],
        "method_runs": plan["raw_measured_records"],
        "warmups": plan["underlying_warmup_dbscan_fits"],
        "measured_samples": plan["raw_measured_records"],
        "valid": plan["raw_measured_records"],
        "failed": 0,
        "semantic_rejected": 0,
    }
    gate_job = gate.get("environment", {}).get("slurm_job_id") or os.environ.get("SLURM_JOB_ID") or spec["baseline_validation"]["job_id"]
    manifest = {
        "$schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "id": run_id,
        "campaign_id": spec["id"],
        "phase": spec["phase"],
        "purpose": spec["purpose"],
        "status": "running",
        "created_at": created,
        "completed_at": None,
        "campaign_spec": artifact_descriptor(campaign_spec_copy, campaign_dir),
        "filters": {
            "case_tiers": sorted({case["tier"] for case in cases}),
            "primary_inference_tiers": ["pilot" if spec["phase"] == "pilot" else "core"],
            "excluded_from_primary_inference": (
                ["stress"] if any(case["tier"] == "stress" for case in cases) else []
            ),
            "pooled_across_cases": False,
        },
        "snapshot": {
            "source_revision": identity["build"]["git_sha"],
            "revision_kind": identity["build"]["revision_kind"],
            "source_tree_sha256": source_sha,
            "source_dirty": identity["build"].get("git_dirty"),
            "binary_sha256": identity["binary_sha256"],
            "build_id": identity["build"]["build_id"],
        },
        "protocol": {key: spec["protocol"][key] for key in ("backend", "index", "neigh_per_row", "max_mbytes_per_batch", "warmup", "measured_samples", "block_design", "methods", "timing_boundary")},
        "validation_gate": {
            "job_id": int(gate_job),
            "source_revision": identity["build"]["git_sha"],
            "validation_approved": True,
            "checked_at": dt.datetime.fromtimestamp(args.validation_matrix.stat().st_mtime, dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "planned": planned_counts,
        "observed": dict(empty_counts),
        "artifacts": {
            key: None
            for key in (
                "environment_json",
                "cases_json",
                "raw_jsonl",
                "summary_json",
                "summary_csv",
                "stdout_log",
                "stderr_log",
                "validation_matrix",
                "lockfile",
                "dataset_hashes",
                "campaign_schema",
                "sample_schema",
                "run_manifest_schema",
                "pilot_manifest",
            )
        },
        "failures": [],
    }
    atomic_write_json(campaign_dir / "manifest.json", manifest)

    method_runs = valid = failed = semantic_rejected = completed_cases = 0
    observed_warmup_fits = 0
    gpu_identifier = gpu_info.get("uuid") or gpu_info.get("pci_bus_id")
    first_labels: dict[tuple[str, str, str], np.ndarray] = {}
    working = campaign_dir / ".working"
    working.mkdir(exist_ok=False)
    try:
        for case_index, case in enumerate(cases):
            case_failed = False
            methods = case_methods(case)
            for block_index in range(spec["protocol"]["measured_samples"]):
                rotation = (case_index + block_index // 2) % len(methods)
                forward = methods[rotation:] + methods[:rotation]
                order = forward if block_index % 2 == 0 else list(reversed(forward))
                block_results = []
                block_arrays: dict[str, np.ndarray] = {}
                for order_index, (method, route) in enumerate(order):
                    sample_stub = f"{case['id']}.b{block_index:02d}.p{order_index:02d}.{method}.{route}"
                    output = working / f"{sample_stub}.worker.json"
                    labels_path = working / f"{sample_stub}.labels.i32"
                    command = worker_command(output=output, labels=labels_path, method=method, route=route, case=case, spec=spec, binary=args.binary)
                    telemetry_before = query_sample_telemetry(gpu_identifier)
                    completed = run_worker_process(command)
                    telemetry_after = query_sample_telemetry(gpu_identifier)
                    method_runs += 1
                    if completed.returncode or not output.is_file():
                        failure_path = campaign_dir / "labels" / f"{sample_stub}.worker-failure.json"
                        failure = {
                            "command": command,
                            "returncode": completed.returncode,
                            "stdout": completed.stdout,
                            "stderr": completed.stderr,
                            "telemetry_before": telemetry_before,
                            "telemetry_after": telemetry_after,
                            "worker_output": read_json(output) if output.is_file() else None,
                        }
                        atomic_write_json(failure_path, failure)
                        failure_type = (
                            "worker-timeout" if completed.returncode == 124 else "worker-failure"
                        )
                        manifest["failures"].append({"case_id": case["id"], "sample_id": sample_stub, "type": failure_type, "message": f"worker {method}/{route} falhou", "artifact": artifact_descriptor(failure_path, campaign_dir)})
                        # Os métodos anteriores deste bloco também ficam sem par e não
                        # podem entrar no agregado; classifique todo o prefixo executado.
                        failed += len(block_results) + 1
                        case_failed = True
                        break
                    worker = read_json(output)
                    if worker.get("worker_failed"):
                        raise CampaignError(f"worker retornou falha com exit 0: {worker}")
                    worker["telemetry_before"] = telemetry_before
                    worker["telemetry_after"] = telemetry_after
                    runtime_path = runtime_dir / f"{sample_stub}.json"
                    atomic_write_json(runtime_path, worker)
                    runtime_artifact = artifact_descriptor(runtime_path, campaign_dir)
                    assert runtime_artifact is not None
                    warmup_multiplier = 1 if method == METHOD_MULTI else case["configuration_count"]
                    observed_warmup_fits += warmup_multiplier * int(spec["protocol"]["warmup"])
                    arrays, _, _ = labels_summary(labels_path, case["n"], case["configuration_count"])
                    key_name = f"{method}.{route}"
                    determinism_key = (case["id"], method, route)
                    if determinism_key in first_labels and not labels_identical(arrays, first_labels[determinism_key]):
                        artifact = preserve_failure_labels(campaign_dir, case["id"], block_index, {"first": first_labels[determinism_key], "current": arrays}, {"type": "nondeterminism", "method": method, "route": route})
                        manifest["failures"].append({"case_id": case["id"], "sample_id": sample_stub, "type": "nondeterminism", "message": f"{method}/{route} divergiu da primeira amostra", "artifact": artifact_descriptor(artifact, campaign_dir)})
                        semantic_rejected += len(block_results) + 1
                        case_failed = True
                        break
                    first_labels.setdefault(determinism_key, arrays.copy())
                    block_arrays[key_name] = arrays
                    record = sample_record(spec=spec, run_id=run_id, case=case, method=method, route=route, block_index=block_index, order_index=order_index, repetition=block_index, identity=identity, source_sha=source_sha, environment=environment, worker=worker, runtime_artifact=runtime_artifact)
                    block_results.append((record, labels_path, key_name))
                if case_failed:
                    if block_arrays:
                        preserve_failure_labels(campaign_dir, case["id"], block_index, block_arrays, {"type": "incomplete-block"})
                    break
                reference_key = "multi.auto"
                if reference_key not in block_arrays:
                    raise CampaignError(f"{case['id']}: bloco sem multi.auto")
                reference = block_arrays[reference_key]
                divergent = [name for name, labels in block_arrays.items() if not labels_identical(labels, reference)]
                if divergent:
                    artifact = preserve_failure_labels(campaign_dir, case["id"], block_index, block_arrays, {"type": "partition-divergence", "reference": reference_key, "divergent": divergent})
                    manifest["failures"].append({"case_id": case["id"], "sample_id": None, "type": "semantic-rejected", "message": f"particoes divergentes: {divergent}", "artifact": artifact_descriptor(artifact, campaign_dir)})
                    semantic_rejected += len(block_results)
                    case_failed = True
                    break
                for record, labels_path, _ in block_results:
                    record["result"]["validation_status"] = "checked-identical"
                    validate_schema(record, "benchmark-sample.schema.json")
                    atomic_write_json(records_dir / f"{record['sample_id']}.json", record)
                    valid += 1
                    labels_path.unlink(missing_ok=True)
                rebuild_jsonl(records_dir, raw_jsonl)
                manifest["observed"].update({"method_runs": method_runs, "warmups": observed_warmup_fits, "measured_samples": valid + failed + semantic_rejected, "valid": valid, "failed": failed, "semantic_rejected": semantic_rejected})
                atomic_write_json(campaign_dir / "manifest.json", manifest)
            if not case_failed:
                completed_cases += 1
            else:
                # Requisito: pare somente o caso defeituoso; preserve e continue os demais.
                manifest["observed"].update(
                    {
                        "cases": completed_cases,
                        "method_runs": method_runs,
                        "warmups": observed_warmup_fits,
                        "measured_samples": valid + failed + semantic_rejected,
                        "valid": valid,
                        "failed": failed,
                        "semantic_rejected": semantic_rejected,
                    }
                )
                atomic_write_json(campaign_dir / "manifest.json", manifest)
                continue
    finally:
        shutil.rmtree(working, ignore_errors=True)

    rebuild_jsonl(records_dir, raw_jsonl)
    if sha256_file(args.binary) != identity["binary_sha256"]:
        raise CampaignError("binario mudou durante a campanha")
    if source_tree_sha256() != source_sha:
        raise CampaignError("arvore-fonte mudou durante a campanha")
    check_lockfile(lockfile_copy)
    # Releitura final impede aceitar uma campanha se points/meta forem trocados no meio.
    final_datasets = validate_all_inputs(spec, args.data_dir)
    for key, initial in datasets.items():
        if final_datasets[key]["sha256"] != initial["sha256"]:
            raise CampaignError(f"dataset {key} mudou durante a campanha")
    manifest["observed"] = {
        "cases": completed_cases,
        "method_runs": method_runs,
        "warmups": observed_warmup_fits,
        "measured_samples": valid + failed + semantic_rejected,
        "valid": valid,
        "failed": failed,
        "semantic_rejected": semantic_rejected,
    }
    # A agregacao le este manifesto para excluir casos incompletos por inteiro.
    atomic_write_json(campaign_dir / "manifest.json", manifest)
    aggregate_records(campaign_dir)
    manifest["completed_at"] = now_utc()
    complete = (
        completed_cases == plan["cases"]
        and method_runs == plan["raw_measured_records"]
        and observed_warmup_fits == plan["underlying_warmup_dbscan_fits"]
        and valid == plan["raw_measured_records"]
        and failed == 0
        and semantic_rejected == 0
    )
    manifest["status"] = "completed" if complete else "partial"
    manifest["artifacts"] = {
        "environment_json": artifact_descriptor(campaign_dir / "environment.json", campaign_dir),
        "cases_json": artifact_descriptor(campaign_dir / "cases.json", campaign_dir),
        "raw_jsonl": artifact_descriptor(raw_jsonl, campaign_dir),
        "summary_json": artifact_descriptor(campaign_dir / "summaries" / "summary.json", campaign_dir),
        "summary_csv": artifact_descriptor(campaign_dir / "summaries" / "summary.csv", campaign_dir),
        # O shell ainda mantem estes arquivos abertos enquanto o manifesto e fechado; um
        # hash calculado aqui ficaria obsoleto depois do ultimo print. Os logs continuam
        # preservados, mas sao deliberadamente null no manifesto ate um finalizador externo.
        "stdout_log": None,
        "stderr_log": None,
        "validation_matrix": artifact_descriptor(validation_matrix_copy, campaign_dir),
        "lockfile": artifact_descriptor(lockfile_copy, campaign_dir),
        "dataset_hashes": artifact_descriptor(dataset_hashes_path, campaign_dir),
        "campaign_schema": artifact_descriptor(
            schema_copies["benchmark-campaign.schema.json"], campaign_dir
        ),
        "sample_schema": artifact_descriptor(
            schema_copies["benchmark-sample.schema.json"], campaign_dir
        ),
        "run_manifest_schema": artifact_descriptor(
            schema_copies["benchmark-run-manifest.schema.json"], campaign_dir
        ),
        "pilot_manifest": artifact_descriptor(pilot_manifest_copy, campaign_dir),
    }
    validate_schema(manifest, "benchmark-run-manifest.schema.json")
    atomic_write_json(campaign_dir / "manifest.json", manifest)
    print(json.dumps({"campaign_dir": str(campaign_dir), "status": manifest["status"], "observed": manifest["observed"]}, ensure_ascii=False))
    return 0 if complete else 2


def parse_float_csv(raw: str) -> list[float]:
    try:
        values = [float(item) for item in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("valores devem ser finitos e positivos")
    return values


def parse_int_csv(raw: str) -> list[int]:
    try:
        values = [int(item) for item in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("valores devem ser inteiros positivos")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare", "validate-inputs"):
        child = sub.add_parser(command)
        child.add_argument("--spec", type=Path, required=True)
        if command != "plan":
            child.add_argument("--data-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--campaign-dir", type=Path, required=True)
    run.add_argument("--gpu-info", type=Path, required=True)
    run.add_argument("--validation-matrix", type=Path, required=True)
    run.add_argument("--lockfile", type=Path, required=True)
    run.add_argument(
        "--allow-core",
        action="store_true",
        help="libera CORE somente junto de --pilot-manifest aprovado",
    )
    run.add_argument(
        "--pilot-manifest",
        type=Path,
        help="manifest.json completo do PILOT, obrigatorio para a CORE",
    )
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--campaign-dir", type=Path, required=True)
    finalize = sub.add_parser("finalize-logs")
    finalize.add_argument("--campaign-dir", type=Path, required=True)
    finalize.add_argument("--stdout-log", type=Path, required=True)
    finalize.add_argument("--stderr-log", type=Path, required=True)
    worker = sub.add_parser("_worker")
    worker.add_argument("--method", choices=(METHOD_MULTI, METHOD_SEQUENTIAL, METHOD_CUML), required=True)
    worker.add_argument("--route", choices=("auto", "annotated", "dense", "not-applicable"), required=True)
    worker.add_argument("--binary", type=Path, required=True)
    worker.add_argument("--input", type=Path, required=True)
    worker.add_argument("--labels", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--n", type=int, required=True)
    worker.add_argument("--d", type=int, required=True)
    worker.add_argument("--eps", type=parse_float_csv, required=True)
    worker.add_argument("--min-samples", type=parse_int_csv, required=True)
    worker.add_argument("--backend", choices=("cuvs", "codes"), required=True)
    worker.add_argument("--index", choices=("int32", "int64"), required=True)
    worker.add_argument("--neigh-per-row", type=int, required=True)
    worker.add_argument("--max-mbytes-per-batch", type=int, required=True)
    worker.add_argument("--warmup", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "plan":
            print(json.dumps(plan_counts(load_spec(args.spec)), indent=2, ensure_ascii=False))
            return 0
        if args.command == "prepare":
            results = prepare_datasets(load_spec(args.spec), args.data_dir)
            print(json.dumps({"prepared": len(results), "datasets": [{"dataset": item["dataset"], "n": item["n"], "sha256": item["sha256"]} for item in results]}, indent=2, ensure_ascii=False))
            return 0
        if args.command == "validate-inputs":
            spec = load_spec(args.spec)
            datasets = validate_all_inputs(spec, args.data_dir)
            cases = resolved_cases(spec, datasets)
            print(json.dumps({"valid": True, "datasets": len(datasets), "cases": len(cases)}, ensure_ascii=False))
            return 0
        if args.command == "aggregate":
            print(json.dumps(aggregate_records(args.campaign_dir), indent=2, ensure_ascii=False))
            return 0
        if args.command == "finalize-logs":
            print(
                json.dumps(
                    finalize_logs(args.campaign_dir, args.stdout_log, args.stderr_log),
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "_worker":
            return worker_main(args)
        return run_campaign(args)
    except CampaignError as exc:
        if args.command == "run":
            manifest_path = args.campaign_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = read_json(manifest_path)
                    if manifest.get("status") == "running":
                        manifest["status"] = "failed"
                        manifest["completed_at"] = now_utc()
                        manifest.setdefault("failures", []).append(
                            {
                                "case_id": "campaign",
                                "sample_id": None,
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "artifact": None,
                            }
                        )
                        validate_schema(manifest, "benchmark-run-manifest.schema.json")
                        atomic_write_json(manifest_path, manifest)
                except Exception as manifest_error:
                    print(
                        f"aviso: nao foi possivel fechar manifesto de falha: {manifest_error}",
                        file=sys.stderr,
                    )
        print(f"erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
