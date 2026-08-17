import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tools import benchmark_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
PILOT_SPEC = ROOT / "scripts" / "campaigns" / "pilot.json"
CORE_SPEC = ROOT / "scripts" / "campaigns" / "core.json"
ZERO_SHA256 = "0" * 64
EMPTY_JSON_SHA256 = hashlib.sha256(b"{}\n").hexdigest()


@pytest.mark.parametrize(
    ("spec_path", "expected"),
    [
        (
            PILOT_SPEC,
            {
                "campaign_id": "dbscanmulti-pilot-v1",
                "phase": "pilot",
                "cases": 7,
                "multi_route_variants_per_block": 13,
                "method_variants_per_block": 27,
                "measured_samples_per_method": 2,
                "symmetric_block_pairs": 1,
                "raw_measured_records": 54,
                "underlying_measured_dbscan_fits": 118,
                "underlying_warmup_dbscan_fits": 118,
                "underlying_total_dbscan_fits": 236,
            },
        ),
        (
            CORE_SPEC,
            {
                "campaign_id": "dbscanmulti-core-v1",
                "phase": "core",
                "cases": 35,
                "multi_route_variants_per_block": 85,
                "method_variants_per_block": 155,
                "measured_samples_per_method": 10,
                "symmetric_block_pairs": 5,
                "raw_measured_records": 1550,
                "underlying_measured_dbscan_fits": 5790,
                "underlying_warmup_dbscan_fits": 5790,
                "underlying_total_dbscan_fits": 11580,
            },
        ),
    ],
)
def test_official_campaign_plan_counts_are_exact(spec_path, expected):
    assert campaign.plan_counts(campaign.load_spec(spec_path)) == expected


def test_minpts_pool_is_deterministic_order_independent_and_preserves_extremes():
    expected = [8, 11, 15, 20, 27, 37, 50, 68]

    assert campaign.minpts_pool([8, 17, 34, 68], 8) == expected
    assert campaign.minpts_pool([68, 34, 17, 8, 17, 8], 8) == expected
    assert campaign.minpts_pool(iter([34, 8, 68, 17]), 8) == expected
    assert campaign.minpts_pool([8, 68], 1) == [8]

    with pytest.raises(campaign.CampaignError, match="inteiros positivos"):
        campaign.minpts_pool([0, 8, 68], 8)
    with pytest.raises(campaign.CampaignError, match="nao contem 8 inteiros"):
        campaign.minpts_pool([8, 12], 8)


def _protocol():
    return deepcopy(campaign.load_spec(PILOT_SPEC)["protocol"])


def _write_dataset(data_dir, *, dataset="tiny_2d", n=3, d=2):
    paths = campaign.dataset_paths(data_dir, dataset, n)
    points = np.arange(n * d, dtype=np.float32)
    labels = np.asarray([0, 0, 1], dtype=np.int32)
    paths["points"].write_bytes(points.tobytes(order="C"))
    paths["labels"].write_bytes(labels.tobytes(order="C"))
    protocol = _protocol()
    metadata = {
        "protocolo_dataset": protocol["dataset_protocol"],
        "dataset": dataset,
        "n": n,
        "d": d,
        "seed": protocol["seed"],
        "dtype": "float32",
        "layout": "row-major",
        "config_order": "eps_major",
        "quantis_eps": protocol["eps_quantiles"],
        "eps": [float(index + 1) / 100 for index in range(8)],
        "min_samples": [8, 17, 34, 68],
        "arquivos": {
            "points": paths["points"].name,
            "labels_verdadeiros": paths["labels"].name,
        },
        "sha256": {
            "points": campaign.sha256_file(paths["points"]),
            "labels_verdadeiros": campaign.sha256_file(paths["labels"]),
            "gerador": campaign.sha256_file(ROOT / "tools" / "gerar_datasets.py"),
        },
    }
    paths["meta"].write_text(json.dumps(metadata), encoding="utf-8")
    return paths, metadata


def test_validate_dataset_accepts_exact_protocol_and_returns_content_hashes(tmp_path):
    paths, _ = _write_dataset(tmp_path)

    result = campaign.validate_dataset(tmp_path, "tiny_2d", 3, _protocol())

    assert result["d"] == 2
    assert result["eps_pool"] == [float(index + 1) / 100 for index in range(8)]
    assert result["minpts_pool"] == [8, 11, 15, 20, 27, 37, 50, 68]
    assert result["sha256"] == {
        "points": campaign.sha256_file(paths["points"]),
        "labels": campaign.sha256_file(paths["labels"]),
        "meta": campaign.sha256_file(paths["meta"]),
    }


@pytest.mark.parametrize(
    ("artifact", "message"),
    [("points", "SHA-256 points diverge"), ("labels", "SHA-256 labels diverge")],
)
def test_validate_dataset_rejects_same_size_content_corruption(tmp_path, artifact, message):
    paths, _ = _write_dataset(tmp_path)
    original = paths[artifact].read_bytes()
    paths[artifact].write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])

    with pytest.raises(campaign.CampaignError, match=message):
        campaign.validate_dataset(tmp_path, "tiny_2d", 3, _protocol())


def test_validate_dataset_rejects_wrong_quantiles_and_point_shape(tmp_path):
    paths, metadata = _write_dataset(tmp_path)
    wrong_quantiles = deepcopy(metadata)
    wrong_quantiles["quantis_eps"] = list(reversed(wrong_quantiles["quantis_eps"]))
    paths["meta"].write_text(json.dumps(wrong_quantiles), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="quantis_eps"):
        campaign.validate_dataset(tmp_path, "tiny_2d", 3, _protocol())

    paths, metadata = _write_dataset(tmp_path)
    truncated = paths["points"].read_bytes()[:-4]
    paths["points"].write_bytes(truncated)
    metadata["sha256"]["points"] = campaign.sha256_file(paths["points"])
    paths["meta"].write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="tamanho .* invalido"):
        campaign.validate_dataset(tmp_path, "tiny_2d", 3, _protocol())


@pytest.mark.parametrize("invalid", [-2, 3])
def test_labels_summary_rejects_values_outside_dbscan_contract(tmp_path, invalid):
    path = tmp_path / "labels.i32"
    np.asarray([0, -1, invalid], dtype=np.int32).tofile(path)
    with pytest.raises(campaign.CampaignError, match=r"\[-1, N-1\]"):
        campaign.labels_summary(path, 3, 1)


def _sample(*, method, route, block, fit_ms, order_index=0):
    is_multi = method == campaign.METHOD_MULTI
    backend = "cuml" if method == campaign.METHOD_CUML else "cuvs"
    route_observed = route if is_multi and route != "auto" else "annotated"
    if not is_multi:
        route_observed = "not-applicable"
    scalar_grid = [(0.01, 8), (0.01, 16), (0.02, 8), (0.02, 16)]
    configuration_fit_ms = (
        []
        if is_multi
        else [
            {
                "config_index": index,
                "eps": eps,
                "min_samples": minimum,
                "fit_ms": fit_ms * fraction,
            }
            for index, ((eps, minimum), fraction) in enumerate(
                zip(scalar_grid, (0.1, 0.2, 0.3, 0.4))
            )
        ]
    )
    configuration_executions = (
        []
        if is_multi
        else [
            {
                "config_index": index,
                "effective_budget_bytes": None,
                "batch_size": None,
                "batches": None,
                "attempts": None,
                "batch_corrections": None,
                "route_observed": "not-applicable",
                "max_nnz": None,
                "total_nnz_max_eps": None,
                "density_max_eps": None,
            }
            for index in range(4)
        ]
    )
    return {
        "$schema": "../../schemas/benchmark-sample.schema.json",
        "schema_version": "1.0.0",
        "campaign_id": "aggregation-test",
        "run_id": "cpu-test",
        "case_id": "multi_both",
        "sample_id": (
            f"multi_both.b{block:02d}.p{order_index:02d}.{method}"
            + (f".{route}" if method == campaign.METHOD_MULTI else "")
        ),
        "phase": "pilot",
        "method": method,
        "sample_kind": "measured",
        "status": "ok",
        "block_index": block,
        "pair_index": block // 2,
        "block_direction": "forward" if block % 2 == 0 else "reverse",
        "order_index": order_index,
        "repetition": block,
        "recorded_at": "2026-08-17T12:00:00Z",
        "identity": {
            "source_revision": "02a77d03e738d23392f14a294e9ae208028e8e12",
            "revision_kind": "source-tree-sha256",
            "source_tree_sha256": ZERO_SHA256,
            "source_dirty": False,
            "binary_sha256": ZERO_SHA256,
            "build_id": "backend-cuvs-arch-sm_80",
            "dataset_sha256": ZERO_SHA256,
            "dataset_metadata_sha256": ZERO_SHA256,
        },
        "parameters": {
            "dataset": "moons_16d",
            "n": 64000,
            "d": 16,
            "seed": 42,
            "backend": backend,
            "route_requested": route,
            "index": "implementation-default" if method == campaign.METHOD_CUML else "int32",
            "neigh_per_row": 0,
            "max_mbytes_per_batch": 56000,
            "warmup_count": 1,
            "k": 2,
            "l": 2,
            "configuration_count": 4,
            "tier": "pilot",
            "eps": [0.01, 0.02],
            "min_samples": [8, 16],
            "config_order": "eps_major",
            "precision": "float32",
            "metric": "L2",
        },
        "environment": {
            "gpu_model": "NVIDIA A100-SXM4-80GB",
            "gpu_uuid": "GPU-cpu-test",
            "compute_capability": "8.0",
            "vram_bytes": 85_899_345_920,
            "driver_version": "test",
            "cuda_runtime_version": 12080,
            "cuda_toolkit_version": "12.8",
            "cuml_version": "26.2.0",
            "cuvs_version": "26.2.0",
            "raft_version": "26.2.0",
            "rmm_version": "26.2.0",
            "hostname": "cpu-test",
            "slurm_job_id": "test",
            "cuda_visible_devices": "0",
        },
        "timings": {
            "boundary": campaign.TIMING_BOUNDARY,
            "setup_ms": 1.0,
            "h2d_ms": 1.0,
            "internal_setup_ms": None,
            "fit_ms": fit_ms,
            "d2h_ms": 1.0,
            "end_to_end_ms": fit_ms + 3.0,
            "configuration_fit_ms": configuration_fit_ms,
        },
        "execution": {
            "requested_budget_bytes": 56_000_000_000,
            "effective_budget_bytes": 56_000_000_000 if is_multi else None,
            "batch_size": 64000 if is_multi else None,
            "batches": 1 if is_multi else None,
            "attempts": 1 if is_multi else None,
            "batch_corrections": 0 if is_multi else None,
            "route_observed": route_observed,
            "routes_per_batch": (
                [{"batch": 0, "route": route_observed, "nnz_max_eps": 64000}]
                if is_multi
                else []
            ),
            "configuration_executions": configuration_executions,
            "max_nnz": 64000 if is_multi else None,
            "total_nnz_max_eps": 64000 if is_multi else None,
            "density_max_eps": 1 / 64000 if is_multi else None,
            "peak_device_memory_bytes": None,
            "runtime_artifact": {
                "path": "raw/runtime/cpu-test.json",
                "sha256": EMPTY_JSON_SHA256,
            },
            "telemetry_before": None,
            "telemetry_after": None,
        },
        "result": {
            "clusters": [2, 2, 2, 2],
            "noise": [0, 0, 0, 0],
            "validation_status": "checked-identical",
        },
        "error": None,
    }


def _write_aggregation_context(base, routes):
    runtime_path = base / "raw" / "runtime" / "cpu-test.json"
    campaign.atomic_write_json(runtime_path, {})
    case = {
        "id": "multi_both",
        "dataset": "moons_16d",
        "n": 64000,
        "d": 16,
        "seed": 42,
        "routes": list(routes),
        "eps": [0.01, 0.02],
        "min_samples": [8, 16],
        "k": 2,
        "l": 2,
        "configuration_count": 4,
        "index": "int32",
        "tier": "pilot",
        "dataset_paths": {
            "points": "data/moons_16d.f32",
            "labels": "data/moons_16d.labels.i32",
            "meta": "data/moons_16d.json",
        },
        "dataset_sha256": {"points": ZERO_SHA256, "labels": ZERO_SHA256, "meta": ZERO_SHA256},
    }
    campaign.atomic_write_json(base / "cases.json", [case])
    spec_path = base / "campaign.json"
    campaign.atomic_write_json(spec_path, {"fixture": True})
    samples = 2
    method_runs = (len(routes) + 2) * samples
    warmups = (len(routes) + 2 * case["configuration_count"]) * samples
    counts = {
        "cases": 1,
        "method_runs": method_runs,
        "warmups": warmups,
        "measured_samples": method_runs,
        "valid": method_runs,
        "failed": 0,
        "semantic_rejected": 0,
    }
    manifest = {
        "$schema": "schemas/benchmark-run-manifest.schema.json",
        "schema_version": campaign.SCHEMA_VERSION,
        "id": "cpu-test",
        "campaign_id": "aggregation-test",
        "phase": "pilot",
        "purpose": "fixture",
        "status": "running",
        "created_at": "2026-08-17T12:00:00Z",
        "completed_at": None,
        "campaign_spec": campaign.artifact_descriptor(spec_path, base),
        "filters": {
            "case_tiers": ["pilot"],
            "primary_inference_tiers": ["pilot"],
            "excluded_from_primary_inference": [],
            "pooled_across_cases": False,
        },
        "snapshot": {
            "source_revision": "02a77d03e738d23392f14a294e9ae208028e8e12",
            "revision_kind": "source-tree-sha256",
            "source_tree_sha256": ZERO_SHA256,
            "source_dirty": False,
            "binary_sha256": ZERO_SHA256,
            "build_id": "backend-cuvs-arch-sm_80",
        },
        "protocol": {
            "backend": "cuvs",
            "index": "int32",
            "neigh_per_row": 0,
            "max_mbytes_per_batch": 56000,
            "warmup": 1,
            "measured_samples": samples,
            "block_design": "symmetric",
            "methods": [
                campaign.METHOD_MULTI,
                campaign.METHOD_SEQUENTIAL,
                campaign.METHOD_CUML,
            ],
            "timing_boundary": campaign.TIMING_BOUNDARY,
        },
        "validation_gate": {
            "job_id": 1,
            "source_revision": "02a77d03e738d23392f14a294e9ae208028e8e12",
            "validation_approved": True,
            "checked_at": "2026-08-17T11:00:00Z",
        },
        "planned": counts,
        "observed": {key: 0 for key in counts},
        "artifacts": {
            name: None
            for name in (
                "environment_json", "cases_json", "raw_jsonl", "summary_json",
                "summary_csv", "stdout_log", "stderr_log", "validation_matrix",
                "lockfile", "dataset_hashes", "campaign_schema", "sample_schema",
                "run_manifest_schema", "pilot_manifest",
            )
        },
        "failures": [],
    }
    campaign.atomic_write_json(base / "manifest.json", manifest)
    return case


def _ratio(summary, metric):
    return next(item["paired_ratio"] for item in summary["ratios"] if item["metric"] == metric)


def test_binary_worker_timeout_is_a_protocol_failure(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise campaign.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(campaign.subprocess, "run", timeout)
    with pytest.raises(campaign.CampaignError, match="excedeu timeout"):
        campaign.run_binary_once(
            binary=tmp_path / "dbscan_multi",
            input_path=tmp_path / "points.f32",
            labels_path=tmp_path / "labels.i32",
            n=10,
            d=2,
            eps=[0.1],
            minimums=[4],
            backend="cuvs",
            index="int32",
            neigh_per_row=0,
            budget=56000,
            warmup=1,
            route="auto",
            timeout_seconds=0.01,
        )


def test_outer_worker_timeout_returns_124_and_reaps_process(monkeypatch):
    monkeypatch.setattr(campaign, "METHOD_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(campaign, "WORKER_SHUTDOWN_GRACE_SECONDS", 0.01)
    completed = campaign.run_worker_process(
        [campaign.sys.executable, "-c", "import time; time.sleep(60)"]
    )
    assert completed.returncode == 124
    assert "excedeu timeout" in completed.stderr


def test_paired_aggregation_ci_and_atomic_json_csv_outputs(tmp_path):
    records_dir = tmp_path / "raw" / "records"
    records_dir.mkdir(parents=True)
    routes = ("annotated", "dense", "auto")
    _write_aggregation_context(tmp_path, routes)
    methods = (
        [(campaign.METHOD_MULTI, route) for route in routes]
        + [
            (campaign.METHOD_SEQUENTIAL, "not-applicable"),
            (campaign.METHOD_CUML, "not-applicable"),
        ]
    )
    timings = {
        0: {"annotated": 10.0, "dense": 12.0, "auto": 11.0, "sequential": 20.0, "cuml": 30.0},
        1: {"annotated": 20.0, "dense": 16.0, "auto": 18.0, "sequential": 30.0, "cuml": 36.0},
    }
    for block, values in timings.items():
        order = methods if block % 2 == 0 else list(reversed(methods))
        order_indexes = {method_route: index for index, method_route in enumerate(order)}
        records = [
            _sample(
                method=campaign.METHOD_MULTI,
                route=route,
                block=block,
                fit_ms=values[route],
                order_index=order_indexes[(campaign.METHOD_MULTI, route)],
            )
            for route in routes
        ]
        records.extend(
            [
                _sample(
                    method=campaign.METHOD_SEQUENTIAL,
                    route="not-applicable",
                    block=block,
                    fit_ms=values["sequential"],
                    order_index=order_indexes[
                        (campaign.METHOD_SEQUENTIAL, "not-applicable")
                    ],
                ),
                _sample(
                    method=campaign.METHOD_CUML,
                    route="not-applicable",
                    block=block,
                    fit_ms=values["cuml"],
                    order_index=order_indexes[(campaign.METHOD_CUML, "not-applicable")],
                ),
            ]
        )
        for record in records:
            campaign.atomic_write_json(records_dir / f"{record['sample_id']}.json", record)

    summary = campaign.aggregate_records(tmp_path)

    # Block ratios are [20/10, 30/20]. Forward/reverse form one pair, reduced
    # by geometric mean before the descriptive statistics and bootstrap.
    pure_annotated = _ratio(summary, "ganho_multi_puro:annotated")
    assert pure_annotated["n"] == 1
    assert pure_annotated["median"] == pytest.approx(math.sqrt(3.0))
    assert pure_annotated["median"] != pytest.approx(25.0 / 15.0)
    assert pure_annotated["confidence_interval"]["conclusive"] is False
    assert pure_annotated["confidence_interval"]["low"] == pytest.approx(math.sqrt(3.0))
    assert pure_annotated["confidence_interval"]["high"] == pytest.approx(math.sqrt(3.0))

    assert _ratio(summary, "speedup_vs_cuml:annotated")["median"] == pytest.approx(math.sqrt(5.4))
    assert _ratio(summary, "annotated_vs_dense")["median"] == pytest.approx(math.sqrt(0.96))
    assert _ratio(summary, "ganho_multi_puro:best_forced")["median"] == pytest.approx(math.sqrt(3.75))
    assert _ratio(summary, "efficiency_per_configuration:annotated")["median"] == pytest.approx(math.sqrt(3.0) / 4)
    assert summary["paired_ratio_rows"][0]["reduction"] == "geometric-mean"
    assert summary["measurement_limitations"]["method_timeout_seconds"] == 900
    assert {row["block_direction"] for row in summary["block_ratio_rows"]} == {
        "forward",
        "reverse",
    }

    summary_path = tmp_path / "summaries" / "summary.json"
    csv_path = tmp_path / "summaries" / "summary.csv"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    csv_ratio = next(
        row
        for row in rows
        if row["kind"] == "paired_ratio" and row["name"] == "ganho_multi_puro:annotated"
    )
    assert int(csv_ratio["n"]) == 64000
    assert int(csv_ratio["sample_n"]) == 1
    assert float(csv_ratio["median"]) == pytest.approx(math.sqrt(3.0))
    assert csv_ratio["conclusive"] == "False"
    assert not list((tmp_path / "summaries").glob(".*.tmp-*"))

    first_ci = campaign.bootstrap_ci([2.0, 1.5], "stable-key", iterations=1000)
    second_ci = campaign.bootstrap_ci([2.0, 1.5], "stable-key", iterations=1000)
    assert first_ci == second_ci
    assert first_ci["seed"] == int.from_bytes(
        hashlib.sha256(b"stable-key").digest()[:8], "big"
    )


def test_aggregation_rejects_duplicate_case_block_method_route(tmp_path):
    records_dir = tmp_path / "raw" / "records"
    records_dir.mkdir(parents=True)
    first = _sample(method=campaign.METHOD_MULTI, route="auto", block=0, fit_ms=10.0)
    second = deepcopy(first)
    second["sample_id"] = first["sample_id"] + ".duplicate"
    campaign.atomic_write_json(records_dir / "first.json", first)
    campaign.atomic_write_json(records_dir / "second.json", second)

    with pytest.raises(campaign.CampaignError, match="registro raw duplicado"):
        campaign.aggregate_records(tmp_path)


def test_aggregation_excludes_entire_incomplete_case(tmp_path):
    records_dir = tmp_path / "raw" / "records"
    records_dir.mkdir(parents=True)
    _write_aggregation_context(tmp_path, ["auto"])
    record = _sample(method=campaign.METHOD_MULTI, route="auto", block=0, fit_ms=10.0)
    campaign.atomic_write_json(records_dir / "one.json", record)

    summary = campaign.aggregate_records(tmp_path)

    assert summary["completeness_checked"] is True
    assert summary["raw_records"] == 1
    assert summary["included_raw_records"] == 0
    assert summary["components"] == []
    assert summary["ratios"] == []
    assert summary["excluded_cases"][0]["case_id"] == "multi_both"
    assert summary["excluded_cases"][0]["expected_records"] == 6


def _completed_pilot_manifest(base, source_sha):
    spec = campaign.load_spec(PILOT_SPEC)
    spec_copy = base / "inputs" / "campaign-spec.json"
    spec_copy.parent.mkdir(parents=True)
    spec_copy.write_bytes(PILOT_SPEC.read_bytes())
    artifact_names = (
        "environment_json",
        "cases_json",
        "raw_jsonl",
        "summary_json",
        "summary_csv",
        "validation_matrix",
        "lockfile",
        "dataset_hashes",
        "campaign_schema",
        "sample_schema",
        "run_manifest_schema",
    )
    artifacts = {}
    for name in artifact_names:
        path = base / "evidence" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")
        artifacts[name] = campaign.artifact_descriptor(path, base)
    artifacts.update({"stdout_log": None, "stderr_log": None, "pilot_manifest": None})
    plan = campaign.plan_counts(spec)
    protocol = spec["protocol"]
    return {
        "$schema": "schemas/benchmark-run-manifest.schema.json",
        "schema_version": campaign.SCHEMA_VERSION,
        "id": "pilot-job-test",
        "campaign_id": spec["id"],
        "phase": "pilot",
        "purpose": spec["purpose"],
        "status": "completed",
        "created_at": "2026-08-17T12:00:00Z",
        "completed_at": "2026-08-17T13:00:00Z",
        "campaign_spec": campaign.artifact_descriptor(spec_copy, base),
        "filters": {
            "case_tiers": ["pilot"],
            "primary_inference_tiers": ["pilot"],
            "excluded_from_primary_inference": [],
            "pooled_across_cases": False,
        },
        "snapshot": {
            "source_revision": source_sha[:40],
            "revision_kind": "source-tree-sha256",
            "source_tree_sha256": source_sha,
            "source_dirty": None,
            "binary_sha256": ZERO_SHA256,
            "build_id": "pilot-test",
        },
        "protocol": {
            key: protocol[key]
            for key in (
                "backend",
                "index",
                "neigh_per_row",
                "max_mbytes_per_batch",
                "warmup",
                "measured_samples",
                "block_design",
                "methods",
                "timing_boundary",
            )
        },
        "validation_gate": {
            "job_id": 5000,
            "source_revision": source_sha[:40],
            "validation_approved": True,
            "checked_at": "2026-08-17T11:00:00Z",
        },
        "planned": {
            "cases": plan["cases"],
            "method_runs": plan["raw_measured_records"],
            "warmups": plan["underlying_warmup_dbscan_fits"],
            "measured_samples": plan["raw_measured_records"],
            "valid": plan["raw_measured_records"],
            "failed": 0,
            "semantic_rejected": 0,
        },
        "observed": {
            "cases": plan["cases"],
            "method_runs": plan["raw_measured_records"],
            "warmups": plan["underlying_warmup_dbscan_fits"],
            "measured_samples": plan["raw_measured_records"],
            "valid": plan["raw_measured_records"],
            "failed": 0,
            "semantic_rejected": 0,
        },
        "artifacts": artifacts,
        "failures": [],
    }


def test_core_promotion_requires_complete_same_snapshot_official_pilot(tmp_path):
    source_sha = "a" * 64
    manifest = _completed_pilot_manifest(tmp_path, source_sha)
    manifest_path = tmp_path / "manifest.json"
    campaign.atomic_write_json(manifest_path, manifest)

    with pytest.raises(campaign.CampaignError, match="stdout_log"):
        campaign.validate_pilot_manifest(manifest_path, source_sha)

    wrong_snapshot = deepcopy(manifest)
    wrong_snapshot["snapshot"]["source_tree_sha256"] = "b" * 64
    campaign.atomic_write_json(manifest_path, wrong_snapshot)
    with pytest.raises(campaign.CampaignError, match="outro snapshot"):
        campaign.validate_pilot_manifest(manifest_path, source_sha)

    partial = deepcopy(manifest)
    partial["status"] = "partial"
    campaign.atomic_write_json(manifest_path, partial)
    with pytest.raises(campaign.CampaignError, match="status=completed"):
        campaign.validate_pilot_manifest(manifest_path, source_sha)
