"""Independent, CPU-only validation helpers for DBSCAN label vectors.

The CUDA implementation and cuML are deliberately not used here.  For datasets
small enough for an exact scientific check, this module builds the epsilon graph
with NumPy in float64 and validates the DBSCAN definition directly:

* a point is core iff its closed epsilon-neighbourhood has at least min_samples;
* clusters of core points are exactly the connected components of the core graph;
* a non-core point is noise iff it has no core neighbour;
* a border point may use any adjacent core component (the only valid ambiguity).

The graph is dense by design.  Callers must set a maximum N appropriate for the
machine instead of silently turning a large benchmark into an approximate check.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

import numpy as np


ARTIFACT_SCHEMA_VERSION = 2


class OracleLimitError(ValueError):
    """Raised when an exact dense epsilon graph exceeds the configured limit."""


def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Canonicalize cluster IDs by first occurrence while preserving noise (-1)."""

    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got shape {values.shape}")
    result = np.full(values.shape, -1, dtype=np.int64)
    mapping: dict[int, int] = {}
    for index, raw_value in enumerate(values):
        value = int(raw_value)
        if value < 0:
            continue
        if value not in mapping:
            mapping[value] = len(mapping)
        result[index] = mapping[value]
    return result


def partition_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Return label-permutation-invariant comparison metrics."""

    from sklearn.metrics import adjusted_rand_score

    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError(
            "candidate and reference must be one-dimensional and have the same shape"
        )

    def cluster_count(values: np.ndarray) -> int:
        return int(np.unique(values[values >= 0]).size)

    candidate_canonical = canonicalize_labels(candidate)
    reference_canonical = canonicalize_labels(reference)
    different = np.flatnonzero(candidate_canonical != reference_canonical)
    return {
        "ari": float(adjusted_rand_score(reference, candidate)),
        "concordancia_ruido": float(np.mean((candidate < 0) == (reference < 0))),
        "particao_identica": bool(different.size == 0),
        "primeiro_ponto_particao_diferente": (
            int(different[0]) if different.size else None
        ),
        "n_clusters_nosso": cluster_count(candidate),
        "n_clusters_cuml": cluster_count(reference),
        "n_ruido_nosso": int(np.count_nonzero(candidate < 0)),
        "n_ruido_cuml": int(np.count_nonzero(reference < 0)),
    }


@dataclass(frozen=True)
class DBSCANStructure:
    """Exact DBSCAN structure for one (epsilon, min_samples) configuration."""

    eps: float
    min_samples: int
    neighbour_counts: np.ndarray
    core_mask: np.ndarray
    core_component: np.ndarray
    adjacent_core_components: tuple[tuple[int, ...], ...]
    ambiguous_border_mask: np.ndarray
    expected_noise_mask: np.ndarray

    @property
    def component_count(self) -> int:
        components = self.core_component[self.core_component >= 0]
        return int(components.max() + 1) if components.size else 0

    def validate_labels(self, labels: np.ndarray, source: str = "labels") -> dict[str, Any]:
        """Validate a label vector against the mathematical DBSCAN structure."""

        values = np.asarray(labels)
        violations: list[dict[str, Any]] = []

        def add_violation(code: str, index: int | None, message: str) -> None:
            # A short bounded list keeps benchmark JSON useful even after a broad mutation.
            if len(violations) < 20:
                violations.append({"codigo": code, "ponto": index, "mensagem": message})

        n = self.core_mask.size
        if values.ndim != 1 or values.size != n:
            add_violation(
                "shape_invalido",
                None,
                f"{source}: shape {values.shape}; esperado ({n},)",
            )
            return self._validation_result(source, violations)
        if not np.issubdtype(values.dtype, np.integer):
            add_violation(
                "dtype_invalido", None, f"{source}: dtype {values.dtype}; esperado inteiro"
            )
            return self._validation_result(source, violations)

        values = values.astype(np.int64, copy=False)
        invalid_negative = np.flatnonzero(values < -1)
        if invalid_negative.size:
            index = int(invalid_negative[0])
            add_violation(
                "rotulo_negativo_invalido",
                index,
                f"{source}: rotulo {int(values[index])}; somente -1 representa ruido",
            )

        component_labels: dict[int, int] = {}
        label_owner: dict[int, int] = {}
        for component in range(self.component_count):
            members = np.flatnonzero(self.core_component == component)
            member_labels = np.unique(values[members])
            if member_labels.size != 1 or int(member_labels[0]) < 0:
                index = int(members[0])
                add_violation(
                    "componente_core_fragmentado",
                    index,
                    f"{source}: componente core {component} possui rotulos "
                    f"{member_labels.tolist()}",
                )
                continue
            label = int(member_labels[0])
            component_labels[component] = label
            if label in label_owner:
                index = int(members[0])
                add_violation(
                    "componentes_core_fundidos",
                    index,
                    f"{source}: componentes core {label_owner[label]} e {component} "
                    f"reutilizam o rotulo {label}",
                )
            else:
                label_owner[label] = component

        for index in np.flatnonzero(~self.core_mask):
            point = int(index)
            adjacent = self.adjacent_core_components[point]
            label = int(values[point])
            if not adjacent:
                if label != -1:
                    add_violation(
                        "ruido_rotulado_como_cluster",
                        point,
                        f"{source}: ponto sem core vizinho recebeu rotulo {label}",
                    )
                continue

            if label == -1:
                add_violation(
                    "borda_rotulada_como_ruido",
                    point,
                    f"{source}: ponto de borda possui componentes core adjacentes "
                    f"{list(adjacent)}",
                )
                continue

            allowed_labels = {
                component_labels[c] for c in adjacent if c in component_labels
            }
            if allowed_labels and label not in allowed_labels:
                add_violation(
                    "borda_em_componente_nao_adjacente",
                    point,
                    f"{source}: rotulo {label}; rotulos adjacentes validos "
                    f"{sorted(allowed_labels)}",
                )

        return self._validation_result(source, violations)

    def component_assignment(self, labels: np.ndarray) -> np.ndarray:
        """Map valid labels to intrinsic core-component IDs (-1 for mathematical noise)."""

        validation = self.validate_labels(labels)
        if not validation["valido"]:
            raise ValueError("component_assignment requires a semantically valid label vector")

        values = np.asarray(labels).astype(np.int64, copy=False)
        assignment = np.full(values.shape, -1, dtype=np.int64)
        label_to_component: dict[int, int] = {}
        for component in range(self.component_count):
            members = np.flatnonzero(self.core_component == component)
            label_to_component[int(values[members[0]])] = component

        assignment[self.core_mask] = self.core_component[self.core_mask]
        for index in np.flatnonzero(~self.core_mask & ~self.expected_noise_mask):
            assignment[index] = label_to_component[int(values[index])]
        return assignment

    def _validation_result(
        self, source: str, violations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        first = next((v["ponto"] for v in violations if v["ponto"] is not None), None)
        return {
            "fonte": source,
            "valido": not violations,
            "primeiro_ponto_invalido": first,
            "violacoes": violations,
            "n_core": int(np.count_nonzero(self.core_mask)),
            "n_componentes_core": self.component_count,
            "n_ruido_matematico": int(np.count_nonzero(self.expected_noise_mask)),
            "n_bordas_ambiguas": int(np.count_nonzero(self.ambiguous_border_mask)),
        }


@dataclass(frozen=True)
class DBSCANEpsilonGraph:
    """Dense closed epsilon-neighbourhood graph reusable across min_samples values."""

    eps: float
    adjacency: np.ndarray

    @property
    def neighbour_counts(self) -> np.ndarray:
        return np.count_nonzero(self.adjacency, axis=1).astype(np.int64, copy=False)

    def structure(self, min_samples: int) -> DBSCANStructure:
        if int(min_samples) <= 0:
            raise ValueError("min_samples must be positive")

        n = self.adjacency.shape[0]
        counts = self.neighbour_counts
        core = counts >= int(min_samples)

        parent = np.arange(n, dtype=np.int64)

        def find(value: int) -> int:
            root = value
            while int(parent[root]) != root:
                root = int(parent[root])
            while int(parent[value]) != value:
                next_value = int(parent[value])
                parent[value] = root
                value = next_value
            return root

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            # Stable roots make artifacts and tests deterministic.
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        core_indices = np.flatnonzero(core)
        for row in core_indices:
            previous_neighbours = np.flatnonzero(self.adjacency[row, :row] & core[:row])
            for neighbour in previous_neighbours:
                union(int(row), int(neighbour))

        component = np.full(n, -1, dtype=np.int64)
        root_to_component: dict[int, int] = {}
        for index in core_indices:
            root = find(int(index))
            if root not in root_to_component:
                root_to_component[root] = len(root_to_component)
            component[index] = root_to_component[root]

        adjacent_components: list[tuple[int, ...]] = [tuple() for _ in range(n)]
        ambiguous = np.zeros(n, dtype=bool)
        expected_noise = np.zeros(n, dtype=bool)
        for index in np.flatnonzero(~core):
            neighbours = np.flatnonzero(self.adjacency[index] & core)
            components = tuple(sorted({int(component[j]) for j in neighbours}))
            adjacent_components[int(index)] = components
            ambiguous[index] = len(components) > 1
            expected_noise[index] = len(components) == 0

        return DBSCANStructure(
            eps=self.eps,
            min_samples=int(min_samples),
            neighbour_counts=counts,
            core_mask=core,
            core_component=component,
            adjacent_core_components=tuple(adjacent_components),
            ambiguous_border_mask=ambiguous,
            expected_noise_mask=expected_noise,
        )


def build_epsilon_graph(
    points: np.ndarray,
    eps: float,
    *,
    max_n: int = 5000,
    block_size: int = 256,
    squared_distance_atol: float | None = None,
) -> DBSCANEpsilonGraph:
    """Build an exact dense epsilon graph on CPU using float64 arithmetic."""

    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError(f"points must be two-dimensional, got shape {points.shape}")
    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("points must have at least one row and one column")
    if not np.isfinite(points).all():
        raise ValueError("points contain NaN or infinity")
    if not np.isfinite(eps) or float(eps) <= 0:
        raise ValueError("eps must be finite and positive")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    n, d = points.shape
    if int(max_n) > 0 and n > int(max_n):
        raise OracleLimitError(
            f"exact semantic oracle requires N <= {int(max_n)}, got N={n}; "
            "increase --oraculo-max-n explicitly or use a reduced validation dataset"
        )

    values = np.asarray(points, dtype=np.float64, order="C")
    norms = np.einsum("ij,ij->i", values, values)
    eps_squared = float(eps) ** 2
    if squared_distance_atol is None:
        scale = max(1.0, float(np.max(np.abs(values))), abs(float(eps)))
        squared_distance_atol = 64.0 * np.finfo(np.float64).eps * (
            d * scale * scale + eps_squared
        )
    if squared_distance_atol < 0 or not np.isfinite(squared_distance_atol):
        raise ValueError("squared_distance_atol must be finite and non-negative")

    adjacency = np.empty((n, n), dtype=bool)
    for start in range(0, n, int(block_size)):
        stop = min(start + int(block_size), n)
        distances_squared = (
            norms[start:stop, None]
            + norms[None, :]
            - 2.0 * values[start:stop] @ values.T
        )
        np.maximum(distances_squared, 0.0, out=distances_squared)
        adjacency[start:stop] = distances_squared <= (
            eps_squared + float(squared_distance_atol)
        )

    # Euclidean distance is symmetric.  OR only resolves last-bit BLAS differences at the
    # boundary, which are already constrained by the explicit float64 tolerance above.
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, True)
    return DBSCANEpsilonGraph(eps=float(eps), adjacency=adjacency)


def compare_semantically(
    candidate: np.ndarray,
    reference: np.ndarray,
    structure: DBSCANStructure,
    *,
    candidate_name: str = "nosso",
    reference_name: str = "cuml",
) -> dict[str, Any]:
    """Compare two label vectors, accepting only genuine ambiguous-border differences."""

    metrics = partition_metrics(candidate, reference)
    candidate_validation = structure.validate_labels(candidate, candidate_name)
    reference_validation = structure.validate_labels(reference, reference_name)
    result: dict[str, Any] = {
        **metrics,
        "oraculo_executado": True,
        "nivel_validacao": "semantico_independente",
        "fontes": {"candidato": candidate_name, "referencia": reference_name},
        "semantica_por_fonte": {
            candidate_name: candidate_validation,
            reference_name: reference_validation,
        },
        "n_clusters_candidato": metrics["n_clusters_nosso"],
        "n_clusters_referencia": metrics["n_clusters_cuml"],
        "n_ruido_candidato": metrics["n_ruido_nosso"],
        "n_ruido_referencia": metrics["n_ruido_cuml"],
        # Nomes mantidos para compatibilidade com o JSON histórico do benchmark.
        "nosso_semantico": candidate_validation,
        "cuml_semantico": reference_validation,
    }

    semantic_valid = candidate_validation["valido"] and reference_validation["valido"]
    differing_assignment = np.empty(0, dtype=np.int64)
    non_ambiguous = np.empty(0, dtype=np.int64)
    if semantic_valid:
        candidate_assignment = structure.component_assignment(candidate)
        reference_assignment = structure.component_assignment(reference)
        differing_assignment = np.flatnonzero(candidate_assignment != reference_assignment)
        non_ambiguous = differing_assignment[
            ~structure.ambiguous_border_mask[differing_assignment]
        ]

    valid = bool(semantic_valid and non_ambiguous.size == 0)
    if not valid:
        status = "divergencia_invalida"
    elif metrics["particao_identica"]:
        status = "particao_identica_e_semanticamente_valida"
    elif differing_assignment.size:
        status = "equivalente_por_borda_ambigua"
    else:
        # Different canonical vectors can result from the different border point changing
        # the first-seen cluster ID, while intrinsic component assignments still coincide.
        status = "equivalente_semanticamente"

    first_candidates = [
        candidate_validation["primeiro_ponto_invalido"],
        reference_validation["primeiro_ponto_invalido"],
        int(differing_assignment[0]) if differing_assignment.size else None,
        metrics["primeiro_ponto_particao_diferente"],
    ]
    first = next((int(value) for value in first_candidates if value is not None), None)
    result.update(
        {
            "valida": valid,
            "status": status,
            "primeiro_ponto_divergente": first,
            "n_atribuicoes_componentes_diferentes": int(differing_assignment.size),
            "n_divergencias_fora_de_borda_ambigua": int(non_ambiguous.size),
        }
    )
    return result


def partition_only_validation(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    reason: str,
) -> dict[str, Any]:
    """Conservatively validate without the semantic oracle.

    Exact partitions pass because cluster IDs are arbitrary.  Any non-exact partition is
    rejected: without the independent graph it cannot be attributed safely to an ambiguous
    border point.
    """

    candidate_array = np.asarray(candidate)
    reference_array = np.asarray(reference)
    metrics = partition_metrics(candidate_array, reference_array)
    format_violations = []
    first_invalid = None
    for name, values in (("nosso", candidate_array), ("cuml", reference_array)):
        if not np.issubdtype(values.dtype, np.integer):
            format_violations.append(f"{name}: dtype {values.dtype}; esperado inteiro")
            continue
        invalid = np.flatnonzero(values < -1)
        if invalid.size:
            index = int(invalid[0])
            if first_invalid is None:
                first_invalid = index
            format_violations.append(
                f"{name}: rotulo {int(values[index])} no ponto {index}; "
                "somente -1 representa ruido"
            )
    valid = bool(metrics["particao_identica"] and not format_violations)
    if format_violations:
        status = "formato_de_rotulo_invalido"
    elif valid:
        status = "particao_identica"
    else:
        status = "divergencia_sem_oraculo"
    return {
        **metrics,
        "oraculo_executado": False,
        "nivel_validacao": "particao_canonica",
        "motivo_oraculo_nao_executado": reason,
        "formato_labels_valido": not format_violations,
        "violacoes_formato_labels": format_violations,
        "valida": valid,
        "status": status,
        "primeiro_ponto_divergente": (
            first_invalid
            if first_invalid is not None
            else metrics["primeiro_ponto_particao_diferente"]
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    """Public content hash used by artifact writers and independent replay tools."""

    return _sha256_array(values)


def write_failure_artifact(
    directory: str | Path,
    *,
    dataset_name: str,
    points: np.ndarray,
    labels: Mapping[str, np.ndarray],
    eps: float,
    min_samples: int,
    validation: Mapping[str, Any],
    source_path: str | Path | None = None,
    context: Mapping[str, Any] | None = None,
) -> Path:
    """Write a replayable failure manifest with deduplicated X and per-case labels."""

    points = np.asarray(points)
    normalized_labels = {name: np.asarray(values) for name, values in labels.items()}
    n = points.shape[0]
    for name, values in normalized_labels.items():
        if values.shape != (n,):
            raise ValueError(f"labels {name!r} have shape {values.shape}; expected ({n},)")

    dataset_hash = _sha256_array(points)
    identity = hashlib.sha256()
    identity.update(dataset_hash.encode("ascii"))
    identity.update(repr((float(eps), int(min_samples))).encode("ascii"))
    for name in sorted(normalized_labels):
        identity.update(name.encode("utf-8"))
        identity.update(_sha256_array(normalized_labels[name]).encode("ascii"))

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name).strip("._") or "dataset"
    case_name = (
        f"{safe_name}_eps-{float(eps):.10g}_minpts-{int(min_samples)}_"
        f"{identity.hexdigest()[:12]}"
    )
    artifact_root = Path(directory)
    case_directory = artifact_root / case_name
    case_directory.mkdir(parents=True, exist_ok=True)

    # Store X once per content hash.  A rectangular multi-grid may fail in many cells;
    # embedding the same N*D array in every case used to multiply storage by k*l.
    datasets_directory = artifact_root / "_datasets"
    datasets_directory.mkdir(parents=True, exist_ok=True)
    dataset_path = datasets_directory / f"points-{dataset_hash}.npy"
    if not dataset_path.exists():
        temporary_path = datasets_directory / (
            f".{dataset_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary_path.open("wb") as stream:
                np.save(stream, np.asarray(points), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic publication. Concurrent writers produce the same hash/content, so the
            # last replace is harmless and no manifest can observe a partial .npy file.
            os.replace(temporary_path, dataset_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    labels_bundle_path = case_directory / "labels.npz"
    np.savez(
        labels_bundle_path,
        **{f"labels_{name}": values for name, values in normalized_labels.items()},
    )

    first = validation.get("primeiro_ponto_divergente")
    if first is None:
        for side in ("nosso_semantico", "cuml_semantico"):
            semantic = validation.get(side)
            if isinstance(semantic, Mapping) and semantic.get("primeiro_ponto_invalido") is not None:
                first = int(semantic["primeiro_ponto_invalido"])
                break

    first_point = None
    if first is not None and 0 <= int(first) < n:
        index = int(first)
        first_point = {
            "indice": index,
            "coordenadas": np.asarray(points[index]).tolist(),
            "rotulos": {name: int(values[index]) for name, values in normalized_labels.items()},
        }

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "tipo": "dbscan_validation_failure",
        "dataset": {
            "nome": dataset_name,
            "source_path": str(source_path) if source_path is not None else None,
            "sha256": dataset_hash,
            "shape": list(points.shape),
            "dtype": str(points.dtype),
            "array_file": (Path("..") / "_datasets" / dataset_path.name).as_posix(),
        },
        "parametros": {"eps": float(eps), "min_samples": int(min_samples)},
        "labels": {
            name: {
                "array_no_bundle": f"labels_{name}",
                "bundle": labels_bundle_path.name,
                "sha256": _sha256_array(values),
                "dtype": str(values.dtype),
            }
            for name, values in normalized_labels.items()
        },
        "primeiro_ponto": first_point,
        "validacao": _json_safe(validation),
        "contexto": _json_safe(context or {}),
        "reproducao": (
            "python tools/validate_dbscan_matrix.py --artifact "
            + str((case_directory / "failure.json").resolve())
        ),
    }
    manifest_path = case_directory / "failure.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path
