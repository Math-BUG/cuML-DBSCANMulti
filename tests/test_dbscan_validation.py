from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.bench_vs_cuml import (
    _csv_floats as bench_csv_floats,
    carregar_config,
    coletar_ambiente,
    ler_labels_config_major,
    rodar_binario,
    sha256_arquivo,
    validacao_exit_code,
    validar_cientificamente,
    validar_contrato_runtime,
    validar_hash_input_meta,
)
from tools.dbscan_validation import (
    build_epsilon_graph,
    canonicalize_labels,
    compare_semantically,
    partition_metrics,
    sha256_array,
    write_failure_artifact,
)
from tools.gerar_datasets import DATASET_PROTOCOL
from tools.validate_dbscan_matrix import (
    _csv_floats as matrix_csv_floats,
    _csv_ints as matrix_csv_ints,
    main as matrix_main,
    validate_matrix,
)


def ambiguous_border_case():
    # Eight core points form two disconnected dense components.  Point 8 sees one core
    # point from each component, but has only three neighbours including itself, so with
    # min_samples=4 it is a genuine ambiguous border point.
    points = np.asarray(
        [
            [-0.9, 0.0],
            [-1.1, 0.0],
            [-1.1, 0.1],
            [-1.1, -0.1],
            [0.9, 0.0],
            [1.1, 0.0],
            [1.1, 0.1],
            [1.1, -0.1],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    border_on_left = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int32)
    border_on_right = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    return points, border_on_left, border_on_right


def labels_from_structure(structure):
    labels = np.full(structure.core_mask.shape, -1, dtype=np.int32)
    labels[structure.core_mask] = structure.core_component[structure.core_mask]
    for index in np.flatnonzero(~structure.core_mask & ~structure.expected_noise_mask):
        labels[index] = structure.adjacent_core_components[int(index)][0]
    return labels


def test_canonical_partition_ignores_cluster_id_permutation():
    left = np.asarray([7, 7, -1, 3, 3], dtype=np.int32)
    right = np.asarray([11, 11, -1, 4, 4], dtype=np.int32)

    assert np.array_equal(canonicalize_labels(left), canonicalize_labels(right))
    metrics = partition_metrics(left, right)
    assert metrics["particao_identica"] is True
    assert metrics["ari"] == 1.0


@pytest.mark.parametrize("raw", ["", ",1", "1,", "1,,2", "nan", "inf", "-inf"])
def test_csv_float_parsers_reject_empty_and_nonfinite_items(raw):
    with pytest.raises((SystemExit, argparse.ArgumentTypeError)):
        bench_csv_floats(raw)
    with pytest.raises(argparse.ArgumentTypeError):
        matrix_csv_floats(raw)


@pytest.mark.parametrize("raw", ["", ",1", "1,", "1,,2", "0", "-1"])
def test_matrix_csv_int_parser_rejects_empty_and_nonpositive_items(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        matrix_csv_ints(raw)


def runtime_json(**overrides):
    result = {
        "backend": "codes",
        "index": "int32",
        "configuration_count": 4,
        "eps_count": 2,
        "min_samples_count": 2,
        "eps": [0.25, 0.5],
        "min_samples": [4, 8],
        "config_order": "eps_major",
        "n": 9,
        "d": 2,
        "build": {
            "git_sha": "ce11a80",
            "build_id": "backend-codes_arch-80_git-ce11a80",
            "configured_backend": "codes",
            "compiled_backends": ["codes"],
        },
    }
    result.update(overrides)
    return result


def validate_runtime(result, **kwargs):
    defaults = {
        "backend": "codes",
        "n": 9,
        "d": 2,
        "eps": [0.25, 0.5],
        "min_samples": [4, 8],
    }
    defaults.update(kwargs)
    return validar_contrato_runtime(result, **defaults)


def test_runtime_contract_requires_identified_build_and_exact_backend_grid():
    valid = runtime_json()
    assert validate_runtime(valid) is valid

    with pytest.raises(SystemExit, match="backend efetivo"):
        validate_runtime(runtime_json(backend="cuvs"))
    with pytest.raises(SystemExit, match="configuration_count"):
        validate_runtime(runtime_json(configuration_count=3))
    with pytest.raises(SystemExit, match="eps efetivo"):
        validate_runtime(runtime_json(eps=[0.25, 0.6]))
    with pytest.raises(SystemExit, match="min_samples efetivo"):
        validate_runtime(runtime_json(min_samples=[4, 9]))
    with pytest.raises(SystemExit, match="index efetivo"):
        validate_runtime(valid, index="int64")
    with pytest.raises(SystemExit, match="revisão de fonte/build_id"):
        validate_runtime(runtime_json(build={"git_sha": "unknown", "build_id": "unknown"}))

    cuvs_from_codes_build = runtime_json(
        backend="cuvs",
        build={
            "git_sha": "ce11a80",
            "build_id": "backend-codes_arch-80_git-ce11a80",
            "configured_backend": "codes",
            "compiled_backends": ["codes"],
        },
    )
    with pytest.raises(SystemExit, match="runtime alegou backend cuvs"):
        validate_runtime(cuvs_from_codes_build, backend="cuvs")

    # O build cuVS contém os dois caminhos e pode executar codes legitimamente.
    codes_from_cuvs_build = runtime_json(
        build={
            "git_sha": "ce11a80",
            "build_id": "backend-cuvs_arch-80_git-ce11a80",
            "configured_backend": "cuvs",
            "compiled_backends": ["cuvs", "codes"],
        }
    )
    assert validate_runtime(codes_from_cuvs_build) is codes_from_cuvs_build

    duplicate_backends = runtime_json(
        build={
            "git_sha": "ce11a80",
            "build_id": "duplicate",
            "configured_backend": "codes",
            "compiled_backends": ["codes", "codes"],
        }
    )
    with pytest.raises(SystemExit, match="contém duplicatas"):
        validate_runtime(duplicate_backends)

    codes_claiming_cuvs = runtime_json(
        build={
            "git_sha": "ce11a80",
            "build_id": "invalid-codes-build",
            "configured_backend": "codes",
            "compiled_backends": ["codes", "cuvs"],
        }
    )
    with pytest.raises(SystemExit, match="não pode declarar cuvs compilado"):
        validate_runtime(codes_claiming_cuvs)


def test_runtime_contract_allows_legacy_build_only_with_explicit_opt_in():
    legacy = runtime_json(build=None)
    with pytest.raises(SystemExit, match="permitir-binario-legado"):
        validate_runtime(legacy)
    assert validate_runtime(legacy, permitir_binario_legado=True) is legacy


def test_binary_success_stderr_is_forwarded(monkeypatch, capsys):
    result = runtime_json()
    result["cuda"] = {"runtime_version": 12080, "driver_version": 12090}
    monkeypatch.setattr(
        "tools.bench_vs_cuml.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(result) + "\n",
            stderr="allocator warning kept for Slurm\n",
        ),
    )

    received, labels = rodar_binario(
        "fake-binary",
        "points.f32",
        9,
        2,
        [0.25, 0.5],
        [4, 8],
        1,
        0,
        0,
        "codes",
        "int32",
        0,
    )
    assert received == result
    assert labels is None
    assert "allocator warning kept for Slurm" in capsys.readouterr().err


def test_environment_provenance_is_json_serializable(monkeypatch):
    monkeypatch.setattr(
        "tools.bench_vs_cuml._registro_pacote",
        lambda *names: {"distribution": names[0], "version": f"version-{names[0]}"},
    )
    monkeypatch.setattr("tools.bench_vs_cuml.platform.node", lambda: "gpu-node-07")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURMD_NODENAME", "gpu-node-07")
    monkeypatch.setenv("SLURM_NODELIST", "gpu-node-[07]")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    environment = coletar_ambiente(
        {"runtime_version": 12080, "driver_version": 12090},
        gpu=b"NVIDIA Test GPU",
        cuml_version="26.08.0",
        cupy_version="14.0.0",
    )
    round_trip = json.loads(json.dumps(environment))
    assert round_trip["hostname"] == "gpu-node-07"
    assert round_trip["gpu"] == "NVIDIA Test GPU"
    assert round_trip["numpy"] == "version-numpy"
    assert round_trip["sklearn"] == "version-scikit-learn"
    assert round_trip["libraft"] == "version-libraft-cu12"
    assert round_trip["librmm"] == "version-librmm-cu12"
    assert round_trip["libcuvs"] == "version-libcuvs-cu12"
    assert round_trip["slurm_job_id"] == "12345"
    assert round_trip["cuda_visible_devices"] == "2"
    assert round_trip["cuda_runtime_version"] == 12080
    assert round_trip["cuda_driver_version"] == 12090


def test_file_sha256_is_content_based(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")
    assert sha256_arquivo(artifact) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
    assert validar_hash_input_meta(sha256_arquivo(artifact), sha256_arquivo(artifact)) is None
    with pytest.raises(SystemExit, match="diverge de meta.sha256.points"):
        validar_hash_input_meta(sha256_arquivo(artifact), "0" * 64)


def test_config_major_label_reader_checks_size_before_reshape(tmp_path):
    labels_path = tmp_path / "labels.i32"
    np.asarray([0, 0, 1], dtype=np.int32).tofile(labels_path)
    with pytest.raises(SystemExit, match="3 labels int32; esperado 4"):
        ler_labels_config_major(labels_path, n=2, configuration_count=2)

    np.asarray([0, 0, 1, 1], dtype=np.int32).tofile(labels_path)
    labels = ler_labels_config_major(labels_path, n=2, configuration_count=2)
    assert labels.tolist() == [[0, 0], [1, 1]]


def test_meta_requires_points_hash_unless_legacy_is_explicit(tmp_path):
    points = tmp_path / "points.f32"
    np.asarray([[0.0], [1.0]], dtype=np.float32).tofile(points)
    meta_path = tmp_path / "dataset.json"
    base_meta = {
        "protocolo_dataset": DATASET_PROTOCOL,
        "dataset": "tiny",
        "n": 2,
        "d": 1,
        "eps": [1.0],
        "min_samples": [2],
        "arquivos": {"points": points.name},
    }
    args = SimpleNamespace(
        meta=str(meta_path),
        input=None,
        n=None,
        d=None,
        eps=None,
        min_samples=None,
        permitir_binario_legado=False,
    )
    meta_path.write_text(json.dumps(base_meta), encoding="utf-8")
    with pytest.raises(SystemExit, match="meta não contém sha256.points"):
        carregar_config(args)

    args.permitir_binario_legado = True
    assert carregar_config(args)[-1] is None

    expected = sha256_arquivo(points)
    base_meta["sha256"] = {"points": expected}
    meta_path.write_text(json.dumps(base_meta), encoding="utf-8")
    args.permitir_binario_legado = False
    assert carregar_config(args)[-1] == expected


def test_oracle_accepts_both_assignments_of_a_genuine_ambiguous_border():
    points, border_on_left, border_on_right = ambiguous_border_case()
    structure = build_epsilon_graph(
        points, 1.0, max_n=20, squared_distance_atol=0.0
    ).structure(4)

    assert structure.core_mask.tolist() == [True] * 8 + [False]
    assert structure.component_count == 2
    assert structure.ambiguous_border_mask.tolist() == [False] * 8 + [True]
    assert structure.validate_labels(border_on_left)["valido"] is True
    assert structure.validate_labels(border_on_right)["valido"] is True

    comparison = compare_semantically(border_on_left, border_on_right, structure)
    assert comparison["particao_identica"] is False
    assert comparison["valida"] is True
    assert comparison["status"] == "equivalente_por_borda_ambigua"
    assert comparison["primeiro_ponto_divergente"] == 8
    assert comparison["n_divergencias_fora_de_borda_ambigua"] == 0


def test_oracle_rejects_artificial_core_mutation_and_noise_mutation():
    points, valid, _ = ambiguous_border_case()
    structure = build_epsilon_graph(points, 1.0, max_n=20).structure(4)

    mutated_core = valid.copy()
    mutated_core[0] = -1
    core_result = structure.validate_labels(mutated_core, "mutated")
    assert core_result["valido"] is False
    assert core_result["primeiro_ponto_invalido"] == 0
    assert core_result["violacoes"][0]["codigo"] == "componente_core_fragmentado"

    mutated_border = valid.copy()
    mutated_border[8] = -1
    border_result = structure.validate_labels(mutated_border, "mutated")
    assert border_result["valido"] is False
    assert any(v["codigo"] == "borda_rotulada_como_ruido" for v in border_result["violacoes"])


def test_closed_epsilon_boundary_and_min_samples_threshold_are_exact():
    points_on_boundary = np.asarray([[0.0], [1.0]], dtype=np.float32)
    on_boundary = build_epsilon_graph(
        points_on_boundary, 1.0, max_n=10, squared_distance_atol=0.0
    ).structure(2)
    assert on_boundary.neighbour_counts.tolist() == [2, 2]
    assert on_boundary.core_mask.tolist() == [True, True]
    assert on_boundary.validate_labels(np.asarray([0, 0], dtype=np.int32))["valido"]

    just_outside = np.nextafter(np.float32(1.0), np.float32(2.0))
    points_outside = np.asarray([[0.0], [just_outside]], dtype=np.float32)
    outside = build_epsilon_graph(
        points_outside, 1.0, max_n=10, squared_distance_atol=0.0
    ).structure(2)
    assert outside.neighbour_counts.tolist() == [1, 1]
    assert outside.core_mask.tolist() == [False, False]
    assert outside.validate_labels(np.asarray([-1, -1], dtype=np.int32))["valido"]


def test_all_noise_and_disconnected_core_components_are_enforced():
    points = np.asarray([[0.0], [10.0], [20.0]], dtype=np.float32)
    structure = build_epsilon_graph(points, 0.5, max_n=10).structure(2)
    assert structure.validate_labels(np.asarray([-1, -1, -1], dtype=np.int32))["valido"]
    invalid = structure.validate_labels(np.asarray([0, -1, -1], dtype=np.int32))
    assert invalid["valido"] is False
    assert invalid["violacoes"][0]["codigo"] == "ruido_rotulado_como_cluster"


def test_multi_eps_multi_minpts_and_rectangular_multi_both_matrix():
    points, _, _ = ambiguous_border_case()
    eps_values = [0.25, 1.0, 2.5]
    min_samples_values = [2, 3, 4, 6]
    labels = []
    for eps in eps_values:
        graph = build_epsilon_graph(points, eps, max_n=20)
        for min_samples in min_samples_values:
            labels.append(labels_from_structure(graph.structure(min_samples)))
    labels = np.asarray(labels, dtype=np.int32)

    result = validate_matrix(
        points,
        eps_values,
        min_samples_values,
        {"codes": labels.copy(), "cuml": labels.copy(), "cuvs": labels.copy()},
        max_n=20,
    )
    assert result["validacao_aprovada"] is True
    assert len(result["configuracoes"]) == 12
    assert [item["config"] for item in result["configuracoes"]] == list(range(12))

    # eps=1.0, min_samples=4 is config 1 * 4 + 2.  Mutating a core point proves
    # that a single bad cell in the rectangular matrix blocks the full validation.
    mutated = labels.copy()
    mutated[6, 0] = -1
    invalid = validate_matrix(
        points,
        eps_values,
        min_samples_values,
        {"codes": mutated, "cuml": labels.copy(), "cuvs": labels.copy()},
        max_n=20,
    )
    assert invalid["validacao_aprovada"] is False
    assert invalid["configuracoes"][6]["aprovada"] is False


def test_benchmark_policy_returns_nonzero_for_unverified_or_invalid_divergence():
    points, valid, other_valid_border = ambiguous_border_case()

    accepted = validar_cientificamente(
        points,
        valid,
        other_valid_border,
        1.0,
        4,
        modo="auto",
        max_n=20,
    )
    assert accepted["valida"] is True

    mutation = valid.copy()
    mutation[0] = -1
    rejected = validar_cientificamente(
        points, mutation, valid, 1.0, 4, modo="auto", max_n=20
    )
    assert rejected["valida"] is False
    assert validacao_exit_code([{"validacao": rejected}]) == 2

    too_large = validar_cientificamente(
        points, valid, other_valid_border, 1.0, 4, modo="auto", max_n=2
    )
    assert too_large["valida"] is False
    assert too_large["status"] == "divergencia_sem_oraculo"

    malformed = np.asarray([-2] + valid[1:].tolist(), dtype=np.int32)
    malformed_result = validar_cientificamente(
        points, malformed, malformed.copy(), 1.0, 4, modo="auto", max_n=20
    )
    assert malformed_result["particao_identica"] is True
    assert malformed_result["valida"] is False
    assert malformed_result["status"] == "formato_de_rotulo_invalido"


def test_failure_artifact_is_self_contained_and_replayable(tmp_path, capsys):
    points, valid, _ = ambiguous_border_case()
    mutated = valid.copy()
    mutated[0] = -1
    structure = build_epsilon_graph(points, 1.0, max_n=20).structure(4)
    validation = compare_semantically(mutated, valid, structure)

    manifest_path = write_failure_artifact(
        tmp_path,
        dataset_name="ambiguous-border",
        points=points,
        labels={"nosso": mutated, "cuml": valid},
        eps=1.0,
        min_samples=4,
        validation=validation,
        source_path="synthetic.f32",
        context={"backend": "codes", "config": 0},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    stored_points = np.load(
        manifest_path.parent / manifest["dataset"]["array_file"], allow_pickle=False
    )
    labels_bundle = np.load(
        manifest_path.parent / manifest["labels"]["nosso"]["bundle"],
        allow_pickle=False,
    )
    assert np.array_equal(stored_points, points)
    assert np.array_equal(labels_bundle["labels_nosso"], mutated)
    assert np.array_equal(labels_bundle["labels_cuml"], valid)
    assert manifest["primeiro_ponto"]["indice"] == 0
    assert manifest["primeiro_ponto"]["rotulos"] == {"nosso": -1, "cuml": 0}
    write_failure_artifact(
        tmp_path,
        dataset_name="ambiguous-border",
        points=points,
        labels={"nosso": mutated, "cuml": valid},
        eps=1.0,
        min_samples=5,
        validation=validation,
    )
    assert len(list(tmp_path.glob("*/failure.json"))) == 2
    assert len(list((tmp_path / "_datasets").glob("*.npy"))) == 1

    assert matrix_main(["--artifact", str(manifest_path), "--oraculo-max-n", "20"]) == 2
    capsys.readouterr()

    # A v1 artifact embedded X and labels in one case.npz; replay remains compatible.
    legacy_directory = tmp_path / "legacy_v1"
    legacy_directory.mkdir()
    np.savez(
        legacy_directory / "case.npz",
        points=points,
        labels_nosso=mutated,
        labels_cuml=valid,
    )
    legacy_manifest = {
        "schema_version": 1,
        "tipo": "dbscan_validation_failure",
        "dataset": {
            "nome": "legacy",
            "sha256": sha256_array(points),
            "bundle": "case.npz",
        },
        "parametros": {"eps": 1.0, "min_samples": 4},
        "labels": {
            "nosso": {
                "array_no_bundle": "labels_nosso",
                "sha256": sha256_array(mutated),
            },
            "cuml": {
                "array_no_bundle": "labels_cuml",
                "sha256": sha256_array(valid),
            },
        },
    }
    legacy_path = legacy_directory / "failure.json"
    legacy_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    assert matrix_main(["--artifact", str(legacy_path), "--oraculo-max-n", "20"]) == 2
    capsys.readouterr()


def test_offline_matrix_cli_exits_nonzero_and_writes_artifact(tmp_path, capsys):
    points, valid, _ = ambiguous_border_case()
    input_path = tmp_path / "points.f32"
    points.tofile(input_path)
    valid_path = tmp_path / "valid.i32"
    valid.tofile(valid_path)
    mutated = valid.copy()
    mutated[0] = -1
    mutated_path = tmp_path / "mutated.i32"
    mutated.tofile(mutated_path)
    failures = tmp_path / "failures"

    base_args = [
        "--input",
        str(input_path),
        "--n",
        str(points.shape[0]),
        "--d",
        str(points.shape[1]),
        "--eps",
        "1.0",
        "--min-samples",
        "4",
        "--oraculo-max-n",
        "20",
    ]
    # A codes-only build can still be checked against cuML without a cuVS label file.
    assert matrix_main(
        base_args
        + ["--labels-codes", str(valid_path), "--labels-cuml", str(valid_path)]
    ) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit, match="exigir-tres-fontes"):
        matrix_main(
            base_args
            + [
                "--labels-codes",
                str(valid_path),
                "--labels-cuml",
                str(valid_path),
                "--exigir-tres-fontes",
            ]
        )

    exit_code = matrix_main(
        base_args
        + [
            "--labels-cuvs",
            str(valid_path),
            "--labels-codes",
            str(mutated_path),
            "--labels-cuml",
            str(valid_path),
            "--falhas-dir",
            str(failures),
        ]
    )
    capsys.readouterr()
    assert exit_code == 2
    manifests = list(failures.glob("*/failure.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert set(manifest["labels"]) == {"codes", "cuml", "cuvs"}
    assert manifest["primeiro_ponto"]["indice"] == 0
