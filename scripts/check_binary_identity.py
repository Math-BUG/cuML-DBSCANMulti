#!/usr/bin/env python3
"""Atesta hash, manifesto e --build-info de um binario antes de executa-lo."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


UNIDENTIFIED = {"", "unknown", "unavailable", "n/a", "none", "null"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"erro: manifesto de build ausente: {path}")
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cuvs", "codes"))
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument(
        "--require-identified",
        action="store_true",
        help="rejeita revisão de fonte/build_id/flags com valores desconhecidos",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binario nao encontrado: {binary}")

    digest_before = sha256(binary)
    completed = subprocess.run(
        [str(binary), "--build-info"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(
            f"erro: --build-info retornou {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
        build = payload["build"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(f"erro: --build-info invalido: {exc}: {completed.stdout!r}") from exc

    digest_after = sha256(binary)
    if digest_before != digest_after:
        raise SystemExit("erro: binario mudou durante a atestacao")

    errors: list[str] = []
    if build.get("configured_backend") != args.backend:
        errors.append(
            f"configured_backend={build.get('configured_backend')!r}, esperado {args.backend!r}"
        )
    if build.get("cuda_arch") != args.cuda_arch:
        errors.append(f"cuda_arch={build.get('cuda_arch')!r}, esperado {args.cuda_arch!r}")
    for field in ("git_sha", "build_id", "flags"):
        value = build.get(field)
        unidentified = not isinstance(value, str) or not value.strip()
        if args.require_identified:
            unidentified = unidentified or value.strip().lower() in UNIDENTIFIED
        if unidentified:
            errors.append(f"build.{field} ausente ou nao identificado: {value!r}")
    revision_kind = build.get("revision_kind")
    if revision_kind not in ("git", "source-tree-sha256", "provided"):
        errors.append(f"revision_kind invalido: {revision_kind!r}")
    if revision_kind == "source-tree-sha256":
        revision = build.get("git_sha", "")
        if not isinstance(revision, str) or len(revision) != 40:
            errors.append("hash da arvore-fonte deve ter 40 digitos hexadecimais")
    if build.get("git_dirty") not in (True, False, None):
        errors.append(f"git_dirty invalido: {build.get('git_dirty')!r}")

    expected_compiled = {"codes"} if args.backend == "codes" else {"cuvs", "codes"}
    compiled = build.get("compiled_backends")
    if (
        not isinstance(compiled, list)
        or len(compiled) != len(expected_compiled)
        or set(compiled) != expected_compiled
    ):
        errors.append(f"compiled_backends={compiled!r}, esperado {sorted(expected_compiled)!r}")

    flags = build.get("flags", "")
    for fragment in (f"backend={args.backend}", f"arch={args.cuda_arch}", "std=", "link_raft="):
        if fragment not in flags:
            errors.append(f"flags nao contem {fragment!r}: {flags!r}")

    manifest_path = binary.parent / "build-config.txt"
    manifest = parse_manifest(manifest_path)
    expected_manifest = {
        "backend": args.backend,
        "cuda_arch": args.cuda_arch,
        "build_id": build.get("build_id"),
        "revision_kind": revision_kind,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            errors.append(f"manifesto {key}={manifest.get(key)!r}, esperado {expected!r}")

    path_fragments = (f"backend-{args.backend}", f"arch-{args.cuda_arch}")
    for fragment in path_fragments:
        if fragment not in binary.parent.name:
            errors.append(f"diretorio do artefato nao contem {fragment!r}: {binary.parent}")

    if errors:
        raise SystemExit("erro: identidade do binario invalida:\n  - " + "\n  - ".join(errors))

    print(
        json.dumps(
            {
                "binary": str(binary),
                "sha256": digest_after,
                "manifest": str(manifest_path.resolve()),
                "build": build,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
