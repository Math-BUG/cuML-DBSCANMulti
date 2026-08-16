#!/usr/bin/env python3
"""Emite um SHA-256 determinístico dos arquivos que definem a execução do projeto."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("Makefile", "requirements.txt", "requirements-cuml.txt", "requirements-ci.txt")
SOURCE_DIRS = ("src", "tools", "scripts", "schemas", "third_party/cuml")


def source_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for directory in SOURCE_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        files.extend(
            path
            for path in base.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> int:
    digest = hashlib.sha256()
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    # 40 hex mantém compatibilidade com consumidores que esperavam um SHA de commit.
    print(digest.hexdigest()[:40])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
