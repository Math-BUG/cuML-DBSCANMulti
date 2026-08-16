import unittest
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.dbscan_validation import build_epsilon_graph
import tools.run_validation_matrix as validation_runner
from tools.run_validation_matrix import adversarial_cases


class ValidationMatrixCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case["name"]: case for case in adversarial_cases(2)}

    def test_ambiguous_border_case_really_contains_ambiguity(self):
        case = self.cases["ambiguous_border_2d"]
        structure = build_epsilon_graph(case["points"], 0.51).structure(5)
        ambiguous = np.flatnonzero(structure.ambiguous_border_mask)
        self.assertEqual(ambiguous.tolist(), [6])

    def test_nextafter_epsilons_remain_distinct(self):
        eps = sorted(self.cases["epsilon_nextafter_1d"]["eps"])
        self.assertEqual(len(set(eps)), 3)
        self.assertLess(eps[0], eps[1])
        self.assertLess(eps[1], eps[2])

    def test_high_dimension_exercises_shared_memory_fallback(self):
        self.assertEqual(self.cases["high_dimension_8193"]["points"].shape, (24, 8193))

    def test_forced_batch_case_is_within_oracle_limit(self):
        case = self.cases["forced_batches_2d"]
        self.assertTrue(case["must_batch"])
        self.assertEqual(case["points"].shape, (1000, 2))


if __name__ == "__main__":
    unittest.main()


def _tiny_case():
    return {
        "name": "smoke",
        "points": np.asarray([[0.0], [0.5]], dtype=np.float32),
        "eps": [1.0],
        "min_samples": [2],
    }


def _runtime_metadata():
    return {
        "execution": {
            "batches": 1,
            "annotated_batches": 0,
            "dense_batches": 0,
        }
    }


def _contract_metadata(*, n=2, d=1, eps=None, min_samples=None):
    eps = [1.0] if eps is None else list(eps)
    min_samples = [2] if min_samples is None else list(min_samples)
    return {
        "backend": "codes",
        "index": "int32",
        "requested_route": "auto",
        "configuration_count": len(eps) * len(min_samples),
        "eps_count": len(eps),
        "min_samples_count": len(min_samples),
        "eps": eps,
        "min_samples": min_samples,
        "config_order": "eps_major",
        "n": n,
        "d": d,
        "build": {
            "git_sha": "abcdef123456",
            "build_id": "backend-codes-test",
            "configured_backend": "codes",
            "compiled_backends": ["codes"],
        },
        "execution": {
            "batches": 1,
            "annotated_batches": 0,
            "dense_batches": 0,
        },
    }


@pytest.mark.parametrize(
    "override, expected_message",
    [
        ({"n": 3}, "shape runtime"),
        ({"eps": [0.5]}, "eps efetivo"),
        ({"config_order": "min_samples_major"}, "config_order"),
    ],
)
def test_run_binary_rejects_runtime_shape_or_grid(
    tmp_path, monkeypatch, override, expected_message
):
    points = np.asarray([[0.0], [0.5]], dtype=np.float32)
    points_path = tmp_path / "points.f32"
    output_path = tmp_path / "labels.i32"
    points.tofile(points_path)
    np.zeros((1, 2), dtype=np.int32).tofile(output_path)
    metadata = _contract_metadata()
    metadata.update(override)
    monkeypatch.setattr(
        validation_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(metadata) + "\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match=expected_message):
        validation_runner._run_binary(
            tmp_path / "dbscan_multi",
            points_path,
            output_path,
            points,
            [1.0],
            [2],
            backend="codes",
            index="int32",
        )


def test_run_binary_forwards_success_stderr(tmp_path, monkeypatch, capsys):
    points = np.asarray([[0.0], [0.5]], dtype=np.float32)
    points_path = tmp_path / "points.f32"
    output_path = tmp_path / "labels.i32"
    points.tofile(points_path)
    np.zeros((1, 2), dtype=np.int32).tofile(output_path)
    monkeypatch.setattr(
        validation_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_contract_metadata()) + "\n",
            stderr="warning from CUDA runtime\n",
        ),
    )

    labels, _ = validation_runner._run_binary(
        tmp_path / "dbscan_multi",
        points_path,
        output_path,
        points,
        [1.0],
        [2],
        backend="codes",
        index="int32",
    )
    assert labels.shape == (1, 2)
    assert "warning from CUDA runtime" in capsys.readouterr().err


def test_main_reaches_run_binary_with_the_declared_signature(tmp_path, monkeypatch, capsys):
    binary = tmp_path / "dbscan_multi"
    binary.write_bytes(b"mock")
    output = tmp_path / "matrix.json"
    calls = []

    def fake_run_binary(
        binary_path,
        points_path,
        output_path,
        points,
        eps,
        min_samples,
        *,
        backend,
        index,
        route="auto",
        max_mbytes=0,
    ):
        calls.append((binary_path, points_path, output_path, backend, index, route, max_mbytes))
        labels = np.zeros((len(eps) * len(min_samples), points.shape[0]), dtype=np.int32)
        return labels, _runtime_metadata()

    monkeypatch.setattr(validation_runner, "adversarial_cases", lambda _seeds: [_tiny_case()])
    monkeypatch.setattr(validation_runner, "_run_binary", fake_run_binary)
    monkeypatch.setattr(
        validation_runner,
        "_run_cuml",
        lambda points, eps, min_samples: np.zeros(
            (len(eps) * len(min_samples), points.shape[0]), dtype=np.int32
        ),
    )

    assert validation_runner.main(
        [
            "--binary",
            str(binary),
            "--random-seeds",
            "0",
            "--repetitions",
            "2",
            "--out",
            str(output),
        ]
    ) == 0
    capsys.readouterr()
    assert calls
    assert all(len(call) == 7 for call in calls)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["validation_approved"] is True
    evidence = result["cases"][0]["labels_evidence"]
    evidence_path = tmp_path / (output.parent / evidence["path"]).name
    assert evidence_path.is_file()
    assert validation_runner.sha256_file(evidence_path) == evidence["sha256"]
    with np.load(evidence_path, allow_pickle=False) as bundle:
        assert bundle["points"].tolist() == [[0.0], [0.5]]
        assert "labels_cuvs_auto_i32_rep0" in bundle
        assert "labels_cuml_rep1" in bundle


def test_nondeterminism_artifact_preserves_first_divergent_repetition(
    tmp_path, monkeypatch, capsys
):
    binary = tmp_path / "dbscan_multi"
    binary.write_bytes(b"mock")
    output = tmp_path / "matrix.json"
    failures = tmp_path / "failures"
    call_counts = {}

    def fake_run_binary(
        binary_path,
        points_path,
        output_path,
        points,
        eps,
        min_samples,
        *,
        backend,
        index,
        route="auto",
        max_mbytes=0,
    ):
        key = (backend, index, route, max_mbytes)
        repetition = call_counts.get(key, 0)
        call_counts[key] = repetition + 1
        labels = np.zeros((len(eps) * len(min_samples), points.shape[0]), dtype=np.int32)
        if key == ("cuvs", "int32", "auto", 0) and repetition == 1:
            labels[0, 1] = 1
        return labels, _runtime_metadata()

    monkeypatch.setattr(validation_runner, "adversarial_cases", lambda _seeds: [_tiny_case()])
    monkeypatch.setattr(validation_runner, "_run_binary", fake_run_binary)
    monkeypatch.setattr(
        validation_runner,
        "_run_cuml",
        lambda points, eps, min_samples: np.zeros(
            (len(eps) * len(min_samples), points.shape[0]), dtype=np.int32
        ),
    )

    assert validation_runner.main(
        [
            "--binary",
            str(binary),
            "--random-seeds",
            "0",
            "--repetitions",
            "2",
            "--falhas-dir",
            str(failures),
            "--out",
            str(output),
        ]
    ) == 2
    capsys.readouterr()

    manifests = list(failures.glob("*/failure.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert "cuvs_auto_i32_rep0" in manifest["labels"]
    assert "cuvs_auto_i32_rep1" in manifest["labels"]
    assert manifest["primeiro_ponto"]["indice"] == 1
    assert manifest["contexto"]["config"] == 0
    determinism = manifest["contexto"]["determinism_failures"]
    assert determinism[0]["source"] == "cuvs_auto_i32"
    assert determinism[0]["different_repetition"] == 1
    assert determinism[0]["point"] == 1

    with np.load(
        manifests[0].parent / manifest["labels"]["cuvs_auto_i32_rep0"]["bundle"],
        allow_pickle=False,
    ) as bundle:
        assert bundle["labels_cuvs_auto_i32_rep0"].tolist() == [0, 0]
        assert bundle["labels_cuvs_auto_i32_rep1"].tolist() == [0, 1]
