import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.bench_vs_cuml import rodar_binario, validar_contrato_runtime


ROOT = Path(__file__).resolve().parents[1]


def runtime_result(*, requested_route="auto", batch_routes=None):
    routes = list(batch_routes or ["annotated", "dense"])
    annotated = routes.count("annotated")
    dense = routes.count("dense")
    if annotated and dense:
        observed = "mixed"
    elif annotated:
        observed = "annotated"
    elif dense:
        observed = "dense"
    else:
        observed = "not-applicable"
    return {
        "backend": "cuvs",
        "index": "int32",
        "requested_route": requested_route,
        "configuration_count": 2,
        "eps_count": 2,
        "min_samples_count": 1,
        "eps": [0.25, 0.5],
        "min_samples": [4],
        "config_order": "eps_major",
        "n": 9,
        "d": 2,
        "execution": {
            "batches": len(routes),
            "batch_routes": routes,
            "annotated_batches": annotated,
            "dense_batches": dense,
            "route_observed": observed,
        },
        "build": {
            "git_sha": "ce11a80",
            "build_id": "backend-cuvs-arch-80-git-ce11a80",
            "configured_backend": "cuvs",
            "compiled_backends": ["cuvs", "codes"],
        },
    }


def validate(result, route="auto"):
    return validar_contrato_runtime(
        result,
        backend="cuvs",
        index="int32",
        route=route,
        n=9,
        d=2,
        eps=[0.25, 0.5],
        min_samples=[4],
    )


def test_runtime_contract_accepts_per_batch_route_telemetry():
    result = runtime_result()
    assert validate(result) is result


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda result: result["execution"].update(batch_routes=["annotated"]),
         "batch_routes tem 1 itens"),
        (lambda result: result["execution"].update(annotated_batches=0),
         "annotated_batches diverge"),
        (lambda result: result["execution"].update(route_observed="annotated"),
         "route_observed diverge"),
    ],
)
def test_runtime_contract_rejects_inconsistent_route_telemetry(mutation, message):
    result = runtime_result()
    mutation(result)
    with pytest.raises(SystemExit, match=message):
        validate(result)


def test_runtime_contract_rejects_route_that_was_not_forced_for_every_batch():
    result = runtime_result(requested_route="annotated")
    with pytest.raises(SystemExit, match="rota forçada 'annotated'"):
        validate(result, route="annotated")


def test_legacy_harness_forwards_route_to_binary(monkeypatch):
    result = runtime_result(requested_route="annotated", batch_routes=["annotated", "annotated"])
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout=json.dumps(result) + "\n", stderr="")

    monkeypatch.setattr("tools.bench_vs_cuml.subprocess.run", fake_run)
    received, labels = rodar_binario(
        "fake-binary",
        "points.f32",
        9,
        2,
        [0.25, 0.5],
        [4],
        1,
        0,
        0,
        "cuvs",
        "int32",
        0,
        route="annotated",
    )

    route_index = captured["command"].index("--route")
    assert captured["command"][route_index + 1] == "annotated"
    assert received == result
    assert labels is None


def test_codes_route_is_reported_as_not_applicable():
    result = runtime_result(batch_routes=["not-applicable", "not-applicable"])
    result["backend"] = "codes"
    result["build"] = {
        "git_sha": "ce11a80",
        "build_id": "backend-codes-arch-80-git-ce11a80",
        "configured_backend": "codes",
        "compiled_backends": ["codes"],
    }
    assert validar_contrato_runtime(
        result,
        backend="codes",
        index="int32",
        route="auto",
        n=9,
        d=2,
        eps=[0.25, 0.5],
        min_samples=[4],
    ) is result


def test_route_telemetry_preserves_warmup_capacity_inside_measured_fit():
    runner = (ROOT / "src" / "multi" / "runner_multi.cuh").read_text(encoding="utf-8")
    assert "stats.reset_preserving_capacity();" in runner
    assert "stats = ExecutionStats{}" not in runner
    reset = runner.split("void reset_preserving_capacity() noexcept", 1)[1].split("};", 1)[0]
    assert "batch_routes.clear();" in reset
    assert "shrink_to_fit" not in reset
    assert ".swap(" not in reset
