"""Validate saved cuVS/codes/cuML DBSCAN labels without requiring a GPU.

Raw matrix mode expects config-major int32 files, with configuration order
``config = eps_index * len(min_samples) + min_samples_index``::

    python tools/validate_dbscan_matrix.py \
      --input data.f32 --n 100 --d 2 --eps 0.2,0.3 --min-samples 4,8 \
      --labels-cuvs cuvs.i32 --labels-codes codes.i32 --labels-cuml cuml.i32

Failure artifacts emitted by ``bench_vs_cuml.py`` can be replayed directly::

    python tools/validate_dbscan_matrix.py --artifact path/to/failure.json
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

try:
    from dbscan_validation import (
        build_epsilon_graph,
        compare_semantically,
        sha256_array,
        write_failure_artifact,
    )
except ImportError:  # pragma: no cover - imported as tools.validate_dbscan_matrix
    from tools.dbscan_validation import (
        build_epsilon_graph,
        compare_semantically,
        sha256_array,
        write_failure_artifact,
    )


def _csv_floats(value: str) -> list[float]:
    items = value.split(",")
    if not items or any(not item.strip() for item in items):
        raise argparse.ArgumentTypeError("a lista contém item vazio")
    try:
        result = sorted({float(item) for item in items})
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"float inválido: {error}") from error
    if not result or any(not np.isfinite(item) or item <= 0 for item in result):
        raise argparse.ArgumentTypeError("eps exige valores finitos e positivos")
    return result


def _csv_ints(value: str) -> list[int]:
    items = value.split(",")
    if not items or any(not item.strip() for item in items):
        raise argparse.ArgumentTypeError("a lista contém item vazio")
    try:
        result = sorted({int(item) for item in items})
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"inteiro inválido: {error}") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("min-samples exige inteiros positivos")
    return result


def _read_array(path: str | Path, dtype: np.dtype, expected_size: int, name: str) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype)
    if values.size != expected_size:
        raise SystemExit(
            f"erro: {name} em '{path}' possui {values.size} valores; esperado {expected_size}"
        )
    return values


def _load_artifact(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("tipo") != "dbscan_validation_failure":
        raise SystemExit(f"erro: '{path}' não é um artefato de validação DBSCAN")
    schema_version = int(manifest.get("schema_version", 1))
    if schema_version == 1:
        # v1 embedded the complete dataset and every label vector in case.npz.
        bundle_path = path.parent / manifest["dataset"]["bundle"]
        with np.load(bundle_path, allow_pickle=False) as bundle:
            points = np.asarray(bundle["points"])
            labels = {
                name: np.asarray(bundle[metadata["array_no_bundle"]])
                for name, metadata in manifest["labels"].items()
            }
    elif schema_version == 2:
        dataset_path = path.parent / manifest["dataset"]["array_file"]
        points = np.asarray(np.load(dataset_path, allow_pickle=False))
        labels = {}
        for name, metadata in manifest["labels"].items():
            labels_path = path.parent / metadata["bundle"]
            with np.load(labels_path, allow_pickle=False) as bundle:
                labels[name] = np.asarray(bundle[metadata["array_no_bundle"]])
    else:
        raise SystemExit(
            f"erro: schema_version={schema_version} não é suportado por este replay"
        )

    if sha256_array(points) != manifest["dataset"]["sha256"]:
        raise SystemExit("erro: hash do dataset deduplicado diverge do manifesto")
    for name, values in labels.items():
        if sha256_array(values) != manifest["labels"][name]["sha256"]:
            raise SystemExit(f"erro: hash dos labels {name} diverge do manifesto")
    parameters = manifest["parametros"]
    return (
        manifest["dataset"]["nome"],
        points,
        [float(parameters["eps"])],
        [int(parameters["min_samples"])],
        {name: values.reshape(1, points.shape[0]) for name, values in labels.items()},
        str(path),
    )


def _load_raw(args):
    missing = [
        option
        for option, value in (
            ("--input", args.input),
            ("--n", args.n),
            ("--d", args.d),
            ("--eps", args.eps),
            ("--min-samples", args.min_samples),
        )
        if value is None
    ]
    if missing:
        raise SystemExit("erro: modo matriz requer " + ", ".join(missing))
    if args.n <= 0 or args.d <= 0:
        raise SystemExit("erro: --n e --d devem ser positivos")

    try:
        eps = _csv_floats(args.eps)
        min_samples = _csv_ints(args.min_samples)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise SystemExit(f"erro: grade inválida: {error}") from error
    label_paths = {
        name: path
        for name, path in {
            "cuvs": args.labels_cuvs,
            "codes": args.labels_codes,
            "cuml": args.labels_cuml,
        }.items()
        if path is not None
    }
    if len(label_paths) < 2:
        raise SystemExit(
            "erro: informe labels de pelo menos duas fontes entre cuVS, codes e cuML"
        )
    if args.exigir_tres_fontes and len(label_paths) != 3:
        raise SystemExit(
            "erro: --exigir-tres-fontes requer --labels-cuvs, --labels-codes e --labels-cuml"
        )

    points = _read_array(args.input, np.float32, args.n * args.d, "dataset").reshape(
        args.n, args.d
    )
    configuration_count = len(eps) * len(min_samples)
    labels = {
        name: _read_array(
            path,
            np.int32,
            configuration_count * args.n,
            f"labels {name}",
        ).reshape(configuration_count, args.n)
        for name, path in label_paths.items()
    }
    return Path(args.input).stem, points, eps, min_samples, labels, str(args.input)


def validate_matrix(
    points: np.ndarray,
    eps_values: list[float],
    min_samples_values: list[int],
    labels_by_backend: dict[str, np.ndarray],
    *,
    max_n: int = 5000,
) -> dict:
    """Validate all configurations/backends and all backend pairs."""

    if len(labels_by_backend) < 2:
        raise ValueError("at least two label sources are required")
    n = points.shape[0]
    expected_shape = (len(eps_values) * len(min_samples_values), n)
    for backend, labels in labels_by_backend.items():
        if np.asarray(labels).shape != expected_shape:
            raise ValueError(
                f"labels {backend!r} have shape {np.asarray(labels).shape}; "
                f"expected {expected_shape}"
            )

    configurations = []
    width = len(min_samples_values)
    for eps_index, eps in enumerate(eps_values):
        graph = build_epsilon_graph(points, eps, max_n=max_n)
        for min_index, min_samples in enumerate(min_samples_values):
            config = eps_index * width + min_index
            structure = graph.structure(min_samples)
            semantic = {
                backend: structure.validate_labels(values[config], backend)
                for backend, values in labels_by_backend.items()
            }
            pair_results = {}
            for left, right in combinations(sorted(labels_by_backend), 2):
                pair_results[f"{left}_vs_{right}"] = compare_semantically(
                    labels_by_backend[left][config],
                    labels_by_backend[right][config],
                    structure,
                    candidate_name=left,
                    reference_name=right,
                )
            approved = all(item["valido"] for item in semantic.values()) and all(
                item["valida"] for item in pair_results.values()
            )
            configurations.append(
                {
                    "config": config,
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                    "aprovada": bool(approved),
                    "semantica": semantic,
                    "pares": pair_results,
                }
            )
    return {
        "schema_version": 1,
        "config_order": "eps_major",
        "n": n,
        "d": int(points.shape[1]),
        "eps": [float(value) for value in eps_values],
        "min_samples": [int(value) for value in min_samples_values],
        "backends": sorted(labels_by_backend),
        "configuracoes": configurations,
        "validacao_aprovada": all(item["aprovada"] for item in configurations),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, help="failure.json a reproduzir")
    parser.add_argument("--input")
    parser.add_argument("--n", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--eps")
    parser.add_argument("--min-samples")
    parser.add_argument("--labels-cuvs")
    parser.add_argument("--labels-codes")
    parser.add_argument("--labels-cuml")
    parser.add_argument(
        "--exigir-tres-fontes",
        action="store_true",
        help="gate da campanha principal: exige cuVS, codes e cuML simultaneamente",
    )
    parser.add_argument("--oraculo-max-n", type=int, default=5000)
    parser.add_argument("--falhas-dir", default="validation_failures")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    if args.oraculo_max_n < 0:
        parser.error("--oraculo-max-n deve ser >= 0")
    if args.artifact:
        raw_options = (
            args.input,
            args.n,
            args.d,
            args.eps,
            args.min_samples,
            args.labels_cuvs,
            args.labels_codes,
            args.labels_cuml,
        )
        if any(value is not None for value in raw_options):
            parser.error("--artifact não pode ser combinado com argumentos do modo matriz")
        dataset_name, points, eps, min_samples, labels, source_path = _load_artifact(
            args.artifact
        )
        if args.exigir_tres_fontes and len(labels) != 3:
            raise SystemExit(
                "erro: o artefato não contém cuVS, codes e cuML como exigido"
            )
    else:
        dataset_name, points, eps, min_samples, labels, source_path = _load_raw(args)

    result = validate_matrix(
        points,
        eps,
        min_samples,
        labels,
        max_n=args.oraculo_max_n,
    )
    result["dataset"] = dataset_name
    result["source_path"] = source_path

    artifacts = []
    if not result["validacao_aprovada"] and not args.artifact:
        for configuration in result["configuracoes"]:
            if configuration["aprovada"]:
                continue
            config = int(configuration["config"])
            artifact = write_failure_artifact(
                args.falhas_dir,
                dataset_name=dataset_name,
                points=points,
                labels={name: values[config] for name, values in labels.items()},
                eps=configuration["eps"],
                min_samples=configuration["min_samples"],
                validation={
                    "valida": False,
                    "status": "matriz_invalida",
                    "primeiro_ponto_divergente": next(
                        (
                            pair["primeiro_ponto_divergente"]
                            for pair in configuration["pares"].values()
                            if not pair["valida"]
                        ),
                        None,
                    ),
                    "detalhes": configuration,
                },
                source_path=source_path,
                context={"modo": "matriz_offline", "backends": sorted(labels)},
            )
            artifacts.append(str(artifact))
    result["artefatos_falha"] = artifacts

    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["validacao_aprovada"] else 2


if __name__ == "__main__":
    sys.exit(main())
