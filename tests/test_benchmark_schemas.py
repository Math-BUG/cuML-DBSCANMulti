import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
CAMPAIGNS = ROOT / "scripts" / "campaigns"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(name):
    schema = _read(SCHEMAS / name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _assert_valid(validator, instance):
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    assert not errors, "\n".join(error.message for error in errors)


def _case_signatures(spec):
    return {
        case["id"]: (
            case["dataset"],
            case["n"],
            case["eps_indices"],
            case["minpts_indices"],
            case["routes"],
        )
        for case in spec["cases"]
    }


def test_all_benchmark_schemas_are_valid_draft_2020_12():
    for name in (
        "benchmark-campaign.schema.json",
        "benchmark-sample.schema.json",
        "benchmark-run-manifest.schema.json",
    ):
        schema = _read(SCHEMAS / name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("name", ["pilot.json", "core.json"])
def test_campaign_specs_validate_without_any_output_artifacts(name):
    _assert_valid(_validator("benchmark-campaign.schema.json"), _read(CAMPAIGNS / name))


def test_campaign_protocol_is_fixed_and_scientifically_comparable():
    expected_common = {
        "backend": "cuvs",
        "index": "int32",
        "neigh_per_row": 0,
        "max_mbytes_per_batch": 56000,
        "warmup": 1,
        "block_design": "symmetric",
        "eps_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "minpts_pool_size": 8,
        "minpts_pool_method": "geometric-rounded-deduplicated-filled-v1",
        "seed": 42,
        "dataset_protocol": "knn-sample-rank-v2",
        "methods": ["multi", "experimental_sequential", "cuml_sequential"],
        "timing_boundary": "device-resident-input-to-device-labels",
    }
    pilot = _read(CAMPAIGNS / "pilot.json")
    core = _read(CAMPAIGNS / "core.json")
    for spec in (pilot, core):
        for key, value in expected_common.items():
            assert spec["protocol"][key] == value
        assert spec["baseline_validation"] == {
            "source_revision": "02a77d03e738d23392f14a294e9ae208028e8e12",
            "job_id": 4996,
            "validation_approved": True,
        }
    assert pilot["protocol"]["measured_samples"] == 2
    assert core["protocol"]["measured_samples"] == 10


def test_pilot_case_inventory_is_exact():
    spec = _read(CAMPAIGNS / "pilot.json")
    all_routes = ["annotated", "dense", "auto"]
    expected = {
        "scalar_sparse": ("filaments_16d", 64000, [0], [3], ["auto"]),
        "scalar_intermediate": ("moons_16d", 64000, [3], [3], ["auto"]),
        "scalar_dense": ("dense_blobs_16d", 64000, [7], [3], ["auto"]),
        "multi_minpts_l4": ("moons_16d", 64000, [3], [0, 2, 4, 7], ["auto"]),
        "multi_eps_sparse_k4": ("filaments_16d", 64000, [0, 1, 2, 3], [3], all_routes),
        "multi_eps_dense_k4": ("dense_blobs_16d", 64000, [4, 5, 6, 7], [3], all_routes),
        "multi_both_2x4": ("moons_16d", 64000, [1, 6], [0, 2, 4, 7], all_routes),
    }
    assert _case_signatures(spec) == expected
    assert sum(len(case["routes"]) for case in spec["cases"]) == 13
    assert 13 + 2 * len(spec["cases"]) == 27


def test_core_case_inventory_is_exact():
    spec = _read(CAMPAIGNS / "core.json")
    auto = ["auto"]
    routes = ["annotated", "dense", "auto"]
    all_eps = list(range(8))
    all_mp = list(range(8))
    expected = {
        "ref_scalar_eps_i0": ("moons_16d", 100000, [0], [3], auto),
        "ref_scalar_eps_i3": ("moons_16d", 100000, [3], [3], auto),
        "ref_scalar_eps_i7": ("moons_16d", 100000, [7], [3], auto),
        "ref_multi_minpts_l2": ("moons_16d", 100000, [3], [0, 7], auto),
        "ref_multi_minpts_l4": ("moons_16d", 100000, [3], [0, 2, 4, 7], auto),
        "ref_multi_minpts_l8": ("moons_16d", 100000, [3], all_mp, auto),
        "ref_multi_eps_k2": ("moons_16d", 100000, [1, 6], [3], routes),
        "ref_multi_eps_k4": ("moons_16d", 100000, [0, 2, 5, 7], [3], routes),
        "ref_multi_eps_k8": ("moons_16d", 100000, all_eps, [3], routes),
        "ref_multi_both_2x2": ("moons_16d", 100000, [1, 6], [0, 7], routes),
        "ref_multi_both_2x4": ("moons_16d", 100000, [1, 6], [0, 2, 4, 7], routes),
        "ref_multi_both_4x2": ("moons_16d", 100000, [0, 2, 5, 7], [0, 7], routes),
        "ref_multi_both_4x4": ("moons_16d", 100000, [0, 2, 5, 7], [0, 2, 4, 7], routes),
        "ref_multi_both_4x8": ("moons_16d", 100000, [0, 2, 5, 7], all_mp, routes),
        "ref_multi_both_8x4": ("moons_16d", 100000, all_eps, [0, 2, 4, 7], routes),
        "index64_multi_minpts_l4": ("moons_16d", 100000, [3], [0, 2, 4, 7], auto),
        "index64_multi_both_2x4_auto": ("moons_16d", 100000, [1, 6], [0, 2, 4, 7], auto),
        "density_filaments_multi_eps_k4": ("filaments_16d", 64000, [0, 1, 2, 3], [3], routes),
        "density_filaments_multi_both_2x4": ("filaments_16d", 64000, [1, 6], [0, 2, 4, 7], routes),
        "density_dense_blobs_multi_eps_k4": ("dense_blobs_16d", 64000, [4, 5, 6, 7], [3], routes),
        "density_dense_blobs_multi_both_2x4": ("dense_blobs_16d", 64000, [1, 6], [0, 2, 4, 7], routes),
        "density_uniform_multi_eps_k4": ("uniform_64d", 64000, [2, 3, 4, 5], [3], routes),
        "density_uniform_multi_both_2x4": ("uniform_64d", 64000, [1, 6], [0, 2, 4, 7], routes),
        "nscale_moons_n64000_multi_minpts_l4": ("moons_16d", 64000, [3], [0, 2, 4, 7], auto),
        "nscale_moons_n64000_multi_eps_k4": ("moons_16d", 64000, [0, 2, 5, 7], [3], routes),
        "nscale_moons_n64000_multi_both_2x4": ("moons_16d", 64000, [1, 6], [0, 2, 4, 7], routes),
        "nscale_moons_n200000_multi_minpts_l4": ("moons_16d", 200000, [3], [0, 2, 4, 7], auto),
        "nscale_moons_n200000_multi_eps_k4": ("moons_16d", 200000, [0, 2, 5, 7], [3], routes),
        "nscale_moons_n200000_multi_both_2x4": ("moons_16d", 200000, [1, 6], [0, 2, 4, 7], routes),
        "dscale_dense_blobs_2d_multi_eps_k4": ("dense_blobs_2d", 64000, [0, 2, 5, 7], [3], routes),
        "dscale_dense_blobs_2d_multi_both_2x4": ("dense_blobs_2d", 64000, [1, 6], [0, 2, 4, 7], routes),
        "dscale_dense_blobs_32d_multi_eps_k4": ("dense_blobs_32d", 64000, [0, 2, 5, 7], [3], routes),
        "dscale_dense_blobs_32d_multi_both_2x4": ("dense_blobs_32d", 64000, [1, 6], [0, 2, 4, 7], routes),
        "dscale_dense_blobs_64d_multi_eps_k4": ("dense_blobs_64d", 64000, [0, 2, 5, 7], [3], routes),
        "dscale_dense_blobs_64d_multi_both_2x4": ("dense_blobs_64d", 64000, [1, 6], [0, 2, 4, 7], routes),
    }
    assert _case_signatures(spec) == expected
    assert len(spec["cases"]) == 35
    assert sum(len(case["routes"]) for case in spec["cases"]) == 85
    assert 85 + 2 * len(spec["cases"]) == 155

    diagnostic_cases = {
        case["id"]: case
        for case in spec["cases"]
        if case.get("index") == "int64"
    }
    assert set(diagnostic_cases) == {
        "index64_multi_minpts_l4",
        "index64_multi_both_2x4_auto",
    }
    assert all(case["routes"] == ["auto"] for case in diagnostic_cases.values())
    assert all("index-overhead" in case["tags"] for case in diagnostic_cases.values())
    assert all(case.get("tier", "core") == "core" for case in diagnostic_cases.values())
    assert {
        case["id"] for case in spec["cases"] if case.get("tier") == "stress"
    } == {
        "density_uniform_multi_eps_k4",
        "density_uniform_multi_both_2x4",
        "nscale_moons_n200000_multi_minpts_l4",
        "nscale_moons_n200000_multi_eps_k4",
        "nscale_moons_n200000_multi_both_2x4",
        "dscale_dense_blobs_64d_multi_eps_k4",
        "dscale_dense_blobs_64d_multi_both_2x4",
    }


def _valid_sample():
    zeros64 = "0" * 64
    return {
        "$schema": "../../schemas/benchmark-sample.schema.json",
        "schema_version": "1.0.0",
        "campaign_id": "dbscanmulti-pilot-v1",
        "run_id": "pilot-job-5000",
        "case_id": "scalar_sparse",
        "sample_id": "scalar_sparse-b0-multi-r0",
        "phase": "pilot",
        "method": "multi",
        "sample_kind": "measured",
        "status": "ok",
        "block_index": 0,
        "pair_index": 0,
        "block_direction": "forward",
        "order_index": 0,
        "repetition": 0,
        "recorded_at": "2026-08-17T12:00:00Z",
        "identity": {
            "source_revision": "02a77d03e738d23392f14a294e9ae208028e8e12",
            "revision_kind": "source-tree-sha256",
            "source_tree_sha256": zeros64,
            "source_dirty": None,
            "binary_sha256": zeros64,
            "build_id": "backend-cuvs-arch-sm_80",
            "dataset_sha256": zeros64,
            "dataset_metadata_sha256": zeros64,
        },
        "parameters": {
            "dataset": "filaments_16d",
            "n": 64000,
            "d": 16,
            "seed": 42,
            "backend": "cuvs",
            "route_requested": "auto",
            "index": "int32",
            "neigh_per_row": 0,
            "max_mbytes_per_batch": 56000,
            "warmup_count": 1,
            "k": 1,
            "l": 1,
            "configuration_count": 1,
            "tier": "pilot",
            "eps": [0.01],
            "min_samples": [16],
            "config_order": "eps_major",
            "precision": "float32",
            "metric": "L2",
        },
        "environment": {
            "gpu_model": "NVIDIA A100",
            "gpu_uuid": "GPU-example",
            "compute_capability": "8.0",
            "vram_bytes": 81153000000,
            "driver_version": "example",
            "cuda_runtime_version": 12080,
            "cuda_toolkit_version": "12.8",
            "cuml_version": "26.2.0",
            "cuvs_version": "26.2.0",
            "raft_version": "26.2.0",
            "rmm_version": "26.2.0",
            "hostname": "gpu-node",
            "slurm_job_id": "5000",
            "cuda_visible_devices": "0",
        },
        "timings": {
            "boundary": "device-resident-input-to-device-labels",
            "setup_ms": 1.0,
            "h2d_ms": 1.0,
            "internal_setup_ms": None,
            "fit_ms": 10.0,
            "d2h_ms": 1.0,
            "end_to_end_ms": 13.0,
            "configuration_fit_ms": [],
        },
        "execution": {
            "requested_budget_bytes": 56000000000,
            "effective_budget_bytes": 56000000000,
            "batch_size": 64000,
            "batches": 1,
            "attempts": 1,
            "batch_corrections": 0,
            "route_observed": "not-applicable",
            "routes_per_batch": [],
            "configuration_executions": [],
            "max_nnz": 64000,
            "total_nnz_max_eps": 64000,
            "density_max_eps": 1 / 64000,
            "peak_device_memory_bytes": None,
            "runtime_artifact": {"path": "raw/runtime/sample.json", "sha256": zeros64},
            "telemetry_before": None,
            "telemetry_after": None,
        },
        "result": {"clusters": [2], "noise": [0], "validation_status": "approved-snapshot"},
        "error": None,
    }


def test_sample_contract_requires_measured_fit_and_structured_failure():
    validator = _validator("benchmark-sample.schema.json")
    sample = _valid_sample()
    _assert_valid(validator, sample)

    missing_fit = copy.deepcopy(sample)
    missing_fit["timings"]["fit_ms"] = None
    assert list(validator.iter_errors(missing_fit))

    failed_without_error = copy.deepcopy(sample)
    failed_without_error["status"] = "failed"
    assert list(validator.iter_errors(failed_without_error))


def test_planned_run_manifest_does_not_require_output_artifacts():
    zeros64 = "0" * 64
    manifest = {
        "$schema": "schemas/benchmark-run-manifest.schema.json",
        "schema_version": "1.0.0",
        "id": "pilot-job-5000",
        "campaign_id": "dbscanmulti-pilot-v1",
        "phase": "pilot",
        "purpose": "Validar o fluxo oficial antes da CORE.",
        "status": "planned",
        "created_at": "2026-08-17T12:00:00Z",
        "completed_at": None,
        "campaign_spec": {"path": "scripts/campaigns/pilot.json", "sha256": zeros64},
        "filters": {
            "case_tiers": ["pilot"],
            "primary_inference_tiers": ["pilot"],
            "excluded_from_primary_inference": [],
            "pooled_across_cases": False,
        },
        "snapshot": {
            "source_revision": "02a77d03e738d23392f14a294e9ae208028e8e12",
            "revision_kind": "source-tree-sha256",
            "source_tree_sha256": zeros64,
            "source_dirty": None,
            "binary_sha256": zeros64,
            "build_id": "backend-cuvs-arch-sm_80",
        },
        "protocol": {
            "backend": "cuvs",
            "index": "int32",
            "neigh_per_row": 0,
            "max_mbytes_per_batch": 56000,
            "warmup": 1,
            "measured_samples": 2,
            "block_design": "symmetric",
            "methods": ["multi", "experimental_sequential", "cuml_sequential"],
            "timing_boundary": "device-resident-input-to-device-labels",
        },
        "validation_gate": {
            "job_id": 4996,
            "source_revision": "02a77d03e738d23392f14a294e9ae208028e8e12",
            "validation_approved": True,
            "checked_at": None,
        },
        "planned": {"cases": 7, "method_runs": 54, "warmups": 118, "measured_samples": 54, "valid": 54, "failed": 0, "semantic_rejected": 0},
        "observed": {"cases": 0, "method_runs": 0, "warmups": 0, "measured_samples": 0, "valid": 0, "failed": 0, "semantic_rejected": 0},
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
    validator = _validator("benchmark-run-manifest.schema.json")
    _assert_valid(validator, manifest)

    completed_without_outputs = copy.deepcopy(manifest)
    completed_without_outputs["status"] = "completed"
    completed_without_outputs["completed_at"] = "2026-08-17T13:00:00Z"
    assert list(validator.iter_errors(completed_without_outputs))
