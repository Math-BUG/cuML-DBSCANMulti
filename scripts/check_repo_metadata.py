#!/usr/bin/env python3
"""Valida metadados, proveniência e invariantes CPU do repositório.

O modo normal verifica consistência sem fingir que bloqueios humanos foram resolvidos.
`--publication` transforma esses bloqueios em erro e só deve passar numa release citável.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # pragma: no cover - CI instala a dependência; fallback é para o cluster
    Draft202012Validator = None
    FormatChecker = None


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "third_party" / "cuml"


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.relative_to(ROOT)}: JSON inválido: {exc}") from exc


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def validate_instance(instance_path: Path, schema_path: Path, errors: list[str]) -> object:
    try:
        schema = read_json(schema_path)
        instance = read_json(instance_path)
        if Draft202012Validator is not None:
            Draft202012Validator.check_schema(schema)
            validator = Draft202012Validator(schema, format_checker=FormatChecker())
            for error in sorted(
                validator.iter_errors(instance),
                key=lambda item: tuple(str(part) for part in item.absolute_path),
            ):
                location = ".".join(str(part) for part in error.path) or "<raiz>"
                errors.append(f"{instance_path.relative_to(ROOT)}:{location}: {error.message}")
        elif isinstance(schema, dict) and isinstance(instance, dict):
            # Fallback deliberadamente pequeno: mantém o gate operacional no cluster sem
            # instalar nada. O CI sempre executa a validação JSON Schema completa.
            missing = set(schema.get("required", [])) - set(instance)
            if missing:
                errors.append(
                    f"{instance_path.relative_to(ROOT)}: campos obrigatórios ausentes: {sorted(missing)}"
                )
            expected = schema.get("properties", {}).get("schema_version", {}).get("const")
            if expected is not None and instance.get("schema_version") != expected:
                errors.append(
                    f"{instance_path.relative_to(ROOT)}: schema_version deve ser {expected}"
                )
        return instance
    except Exception as exc:  # schema inválido também deve produzir diagnóstico único
        errors.append(str(exc))
        return {}


def check_schema_files(errors: list[str]) -> None:
    """Valida todos os contratos mesmo quando ainda não existem outputs da campanha."""

    ids: dict[str, Path] = {}
    for path in sorted((ROOT / "schemas").glob("*.schema.json")):
        try:
            schema = read_json(path)
            if not isinstance(schema, dict):
                errors.append(f"{path.relative_to(ROOT)}: schema deve ser um objeto JSON")
                continue
            if Draft202012Validator is not None:
                Draft202012Validator.check_schema(schema)
            elif schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
                errors.append(
                    f"{path.relative_to(ROOT)}: $schema deve declarar draft 2020-12"
                )
            schema_id = schema.get("$id")
            if not isinstance(schema_id, str) or not schema_id:
                errors.append(f"{path.relative_to(ROOT)}: $id ausente")
            elif schema_id in ids:
                errors.append(
                    f"{path.relative_to(ROOT)}: $id duplicado com "
                    f"{ids[schema_id].relative_to(ROOT)}"
                )
            else:
                ids[schema_id] = path
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: schema inválido: {exc}")


def check_campaign_cross_fields(spec: dict, path: Path, errors: list[str]) -> None:
    """Invariantes derivadas que JSON Schema não expressa sem extensões não portáveis."""

    if not isinstance(spec, dict):
        return
    protocol = spec.get("protocol", {})
    cases = spec.get("cases", [])
    if not isinstance(protocol, dict) or not isinstance(cases, list):
        return

    eps_count = len(protocol.get("eps_quantiles", []))
    minpts_count = protocol.get("minpts_pool_size")
    seen: set[str] = set()
    family_tags = {"scalar", "multi-minpts", "multi-eps", "multi-both"}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        prefix = f"{path.relative_to(ROOT)}:cases.{index}"
        case_id = case.get("id")
        if case_id in seen:
            errors.append(f"{prefix}: id duplicado: {case_id!r}")
        elif isinstance(case_id, str):
            seen.add(case_id)

        eps_indices = case.get("eps_indices", [])
        minpts_indices = case.get("minpts_indices", [])
        if isinstance(eps_indices, list):
            if eps_indices != sorted(eps_indices):
                errors.append(f"{prefix}: eps_indices deve estar em ordem crescente")
            if any(not isinstance(item, int) or item < 0 or item >= eps_count for item in eps_indices):
                errors.append(f"{prefix}: eps_indices excede a grade de {eps_count} quantis")
        if isinstance(minpts_indices, list) and isinstance(minpts_count, int):
            if minpts_indices != sorted(minpts_indices):
                errors.append(f"{prefix}: minpts_indices deve estar em ordem crescente")
            if any(
                not isinstance(item, int) or item < 0 or item >= minpts_count
                for item in minpts_indices
            ):
                errors.append(
                    f"{prefix}: minpts_indices excede o pool de {minpts_count} valores"
                )

        tags = set(case.get("tags", [])) if isinstance(case.get("tags"), list) else set()
        selected_families = tags & family_tags
        if len(selected_families) != 1:
            errors.append(f"{prefix}: deve haver exatamente uma tag de família")
            continue
        family = next(iter(selected_families))
        expected_routes = (
            ["auto"]
            if "index-overhead" in tags
            else (
                ["annotated", "dense", "auto"]
                if family in {"multi-eps", "multi-both"}
                else ["auto"]
            )
        )
        if case.get("routes") != expected_routes:
            errors.append(
                f"{prefix}: routes de {family} deve ser {expected_routes!r}"
            )


def check_vendor(errors: list[str]) -> None:
    manifest_path = VENDOR_ROOT / "VENDORED.json"
    manifest = read_json(manifest_path)
    entries = manifest.get("files", []) if isinstance(manifest, dict) else []
    declared = {entry.get("path"): entry.get("git_blob_sha1") for entry in entries}
    actual = {
        path.relative_to(VENDOR_ROOT).as_posix()
        for path in VENDOR_ROOT.rglob("*")
        if path.is_file() and path.name not in {"VENDORED.md", "VENDORED.json"}
    }
    if set(declared) != actual:
        missing = sorted(actual - set(declared))
        extra = sorted(set(declared) - actual)
        errors.append(f"VENDORED.json diverge da árvore; ausentes={missing}, extras={extra}")
    for relative, expected in declared.items():
        path = VENDOR_ROOT / relative
        if not path.is_file():
            continue
        observed = git_blob_sha1(path)
        if not re.fullmatch(r"[0-9a-f]{40}", str(expected)):
            errors.append(f"VENDORED.json:{relative}: SHA-1 não tem 40 hexadecimais")
        elif observed != expected:
            errors.append(f"blob alterado: third_party/cuml/{relative}: {observed} != {expected}")


def check_file_licenses(project_status: dict, errors: list[str]) -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for path in sorted((ROOT / "src").rglob("*")):
        if not path.is_file() or path.suffix not in {".cu", ".cuh", ".h", ".hpp"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "SPDX-License-Identifier:" not in text[:1200]:
            errors.append(f"{path.relative_to(ROOT)}: cabeçalho SPDX ausente")

    derived = [
        ROOT / "src/multi/corepoints_multi.cuh",
        ROOT / "src/multi/runner_multi.cuh",
        ROOT / "src/multi/vertexdeg_cuvs.cuh",
    ]
    for path in derived:
        text = path.read_text(encoding="utf-8")
        if "NVIDIA CORPORATION" not in text[:1500] or "Modifications Copyright" not in text[:1500]:
            errors.append(f"{path.relative_to(ROOT)}: avisos do derivado incompletos")
        if path.relative_to(ROOT).as_posix() not in notice:
            errors.append(f"{path.relative_to(ROOT)}: derivado ausente do NOTICE")

    license_status = project_status.get("project_license", {}).get("status")
    root_license = project_status.get("project_license", {}).get("root_license_file")
    if license_status == "resolved":
        if not root_license or not (ROOT / root_license).is_file():
            errors.append("project_license resolvida sem arquivo de licença raiz existente")
    elif root_license is not None:
        errors.append("project_license não resolvida deve manter root_license_file como null")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for key in ("cff-version", "message", "title", "authors", "repository-code"):
        if not re.search(rf"^{re.escape(key)}\s*:", citation, flags=re.MULTILINE):
            errors.append(f"CITATION.cff: campo obrigatório/local ausente: {key}")
    if license_status != "resolved" and re.search(r"^license\s*:", citation, flags=re.MULTILINE):
        errors.append("CITATION.cff declara licença global enquanto project_license está aberta")


def check_markdown_links(errors: list[str]) -> None:
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    documents.extend(sorted((ROOT / "docs").glob("*.md")))
    documents.append(VENDOR_ROOT / "VENDORED.md")
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            target = target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^(?:https?://|mailto:)", target):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{document.relative_to(ROOT)}: link local inexistente: {target}")


def check_requirements(errors: list[str]) -> None:
    text = (ROOT / "requirements-cuml.txt").read_text(encoding="utf-8")
    for package in ("cuml-cu12", "libraft-cu12", "librmm-cu12", "libcuvs-cu12"):
        if not re.search(rf"^{re.escape(package)}==26\.2\.\*$", text, flags=re.MULTILINE):
            errors.append(f"requirements-cuml.txt: {package} deve permanecer na série 26.2.*")
    if not re.search(r"^rapids-logger(?:[<=> ]|$)", text, flags=re.MULTILINE):
        errors.append("requirements-cuml.txt: rapids-logger é exigido pelos headers do RAFT")

    cpu_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("numpy", "scikit-learn", "pytest"):
        if not re.search(rf"^{re.escape(package)}(?:[<=> #]|$)", cpu_text, flags=re.MULTILINE):
            errors.append(f"requirements.txt: dependência direta ausente: {package}")
    for package in ("pandas", "matplotlib"):
        if re.search(rf"^{re.escape(package)}(?:[<=> #]|$)", cpu_text, flags=re.MULTILINE):
            errors.append(f"requirements.txt: dependência direta sem uso: {package}")


def check_manifest_cross_fields(manifest: dict, errors: list[str]) -> None:
    experiment_id = manifest.get("experiment_id", "<sem-id>")
    validation = manifest.get("validation", {})
    if validation.get("validation_passed") and (
        validation.get("semantic_valid_configurations")
        != validation.get("configuration_count")
    ):
        errors.append(
            f"{experiment_id}: semantic_valid_configurations deve cobrir "
            "configuration_count quando validation_passed=true"
        )

    protocol = manifest.get("protocol", {})
    requested = protocol.get("index_requested")
    effective = protocol.get("index_effective")
    if requested in {"int32", "int64"} and effective != requested:
        errors.append(
            f"{experiment_id}: index_effective={effective!r} diverge do índice "
            f"forçado {requested!r}"
        )

    budget = protocol.get("batch_budget_protocol", {})
    max_mbytes = protocol.get("max_mbytes_per_batch")
    controlled = budget.get("controlled_equal_request")
    if controlled:
        expected_bytes = max_mbytes * 1_000_000 if isinstance(max_mbytes, int) else None
        if not isinstance(max_mbytes, int) or max_mbytes <= 0:
            errors.append(
                f"{experiment_id}: orçamento controlado exige max_mbytes_per_batch > 0"
            )
        if budget.get("requested_mbytes_each") != max_mbytes:
            errors.append(
                f"{experiment_id}: requested_mbytes_each diverge de max_mbytes_per_batch"
            )
        for field in ("multi_requested_bytes", "cuml_requested_bytes"):
            if budget.get(field) != expected_bytes:
                errors.append(
                    f"{experiment_id}: {field} diverge do orçamento controlado"
                )
    elif isinstance(max_mbytes, int) and max_mbytes > 0:
        errors.append(
            f"{experiment_id}: max_mbytes_per_batch explícito deve marcar "
            "controlled_equal_request=true"
        )

    repository = manifest.get("repository", {})
    binary = manifest.get("binary", {})
    commit = repository.get("commit")
    build_sha = binary.get("git_sha")
    if (
        isinstance(commit, str)
        and isinstance(build_sha, str)
        and commit != "0" * 40
        and not commit.startswith(build_sha)
    ):
        errors.append(
            f"{experiment_id}: binary.git_sha não corresponde a repository.commit"
        )
    if (
        isinstance(repository.get("dirty"), bool)
        and isinstance(binary.get("git_dirty"), bool)
        and repository["dirty"] != binary["git_dirty"]
    ):
        errors.append(
            f"{experiment_id}: estado dirty do binário diverge do manifesto"
        )


def check_publication(project_status: dict, manifests: list[dict], errors: list[str]) -> None:
    if not project_status.get("publication_ready"):
        errors.append("proveniência/licença ainda marcam publication_ready=false")
    if project_status.get("blockers"):
        errors.append("provenance/project-status.json ainda contém bloqueios")
    lock_path = ROOT / "requirements.lock.txt"
    if not lock_path.is_file():
        errors.append("requirements.lock.txt ausente")
    else:
        lock_lines = [
            line.strip()
            for line in lock_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        locked: dict[str, str] = {}
        malformed = []
        for line in lock_lines:
            if line.count("==") != 1:
                malformed.append(line)
                continue
            name, version = line.split("==", 1)
            canonical_name = re.sub(r"[-_.]+", "-", name).lower()
            if not canonical_name or not version or any(char.isspace() for char in version):
                malformed.append(line)
                continue
            locked[canonical_name] = version
        if not lock_lines:
            errors.append("requirements.lock.txt vazio")
        if malformed:
            errors.append(
                "requirements.lock.txt contém entradas sem pin name==version: "
                f"{malformed[:3]}"
            )
        required_direct = {
            "numpy",
            "scikit-learn",
            "pytest",
            "cupy-cuda12x",
            "cuml-cu12",
            "libraft-cu12",
            "librmm-cu12",
            "libcuvs-cu12",
        }
        missing_direct = sorted(required_direct - set(locked))
        if missing_direct:
            errors.append(
                "requirements.lock.txt não congela todas as dependências diretas: "
                f"{missing_direct}"
            )
        for package in ("cuml-cu12", "libraft-cu12", "librmm-cu12", "libcuvs-cu12"):
            version = locked.get(package)
            if version is not None and not version.startswith("26.2."):
                errors.append(
                    f"requirements.lock.txt: {package}={version} diverge da série 26.2.x"
                )
    candidates = [
        manifest
        for manifest in manifests
        if not str(manifest.get("experiment_id", "")).startswith("example-")
        and manifest.get("status") == "publication-ready"
    ]
    if not candidates:
        errors.append("nenhum manifesto publication-ready não-exemplo")
    for manifest in candidates:
        validation = manifest.get("validation", {})
        if validation.get("semantic_valid_configurations") != validation.get(
            "configuration_count"
        ):
            errors.append(
                f"{manifest.get('experiment_id')}: o oráculo semântico deve cobrir toda a grade"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publication",
        action="store_true",
        help="falha também nos bloqueios externos necessários para uma release citável",
    )
    args = parser.parse_args()
    errors: list[str] = []

    check_schema_files(errors)

    provenance_path = ROOT / "provenance" / "project-status.json"
    project_status = validate_instance(
        provenance_path, ROOT / "schemas" / "provenance-status.schema.json", errors
    )
    manifests = []
    for path in sorted((ROOT / "results" / "manifests").glob("*.json")):
        manifest = validate_instance(
            path, ROOT / "schemas" / "experiment-manifest.schema.json", errors
        )
        manifests.append(manifest)
        check_manifest_cross_fields(manifest, errors)

    campaign_schema = ROOT / "schemas" / "benchmark-campaign.schema.json"
    for name in ("pilot.json", "core.json"):
        path = ROOT / "scripts" / "campaigns" / name
        if not path.is_file():
            errors.append(f"{path.relative_to(ROOT)}: especificação obrigatória ausente")
            continue
        spec = validate_instance(path, campaign_schema, errors)
        check_campaign_cross_fields(
            spec if isinstance(spec, dict) else {}, path, errors
        )

    check_vendor(errors)
    check_file_licenses(project_status if isinstance(project_status, dict) else {}, errors)
    check_markdown_links(errors)
    check_requirements(errors)
    if args.publication:
        check_publication(
            project_status if isinstance(project_status, dict) else {},
            [item for item in manifests if isinstance(item, dict)],
            errors,
        )

    if errors:
        for error in errors:
            print(f"ERRO: {error}", file=sys.stderr)
        return 1
    if Draft202012Validator is None:
        print(
            "AVISO: jsonschema ausente; validação estrutural reduzida. "
            "Instale requirements-ci.txt para o gate completo."
        )
    mode = "publicação" if args.publication else "consistência"
    print(f"OK: metadados, schemas, proveniência e licenças ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
