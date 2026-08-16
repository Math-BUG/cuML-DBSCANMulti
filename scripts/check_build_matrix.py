#!/usr/bin/env python3
"""Valida a identidade dos artefatos e o grafo do Makefile sem exigir CUDA."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ("base", "cuvs", "sm_80", "c++17", "0"),
    ("backend", "codes", "sm_80", "c++17", "0"),
    ("arch", "cuvs", "sm_90", "c++17", "0"),
    ("std", "cuvs", "sm_80", "c++20", "0"),
    ("raft", "cuvs", "sm_80", "c++17", "1"),
)


def make_args(make: str, backend: str, arch: str, std: str, raft: str) -> list[str]:
    return [
        make,
        "--no-print-directory",
        f"PYTHON={sys.executable}",
        f"BACKEND={backend}",
        f"CUDA_ARCH={arch}",
        f"STD={std}",
        f"LINK_RAFT={raft}",
    ]


def run(command: list[str], *, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout, end="", file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(
            f"falha ({completed.returncode}) ao executar: {' '.join(command)}"
        )
    if not quiet and completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed


def target_for(make: str, backend: str, arch: str, std: str, raft: str) -> str:
    completed = run(
        [*make_args(make, backend, arch, std, raft), "-s", "print-target"],
        quiet=True,
    )
    if completed.stderr.strip():
        raise SystemExit(f"print-target escreveu em stderr: {completed.stderr!r}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SystemExit(f"print-target deveria emitir uma linha; recebeu: {lines!r}")
    return lines[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", default="make", help="executavel GNU Make")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="tambem expande o grafo/receitas com make -B -n",
    )
    args = parser.parse_args()

    targets: dict[str, str] = {}
    for label, backend, arch, std, raft in CONFIGS:
        targets[label] = target_for(args.make, backend, arch, std, raft)

    if len(set(targets.values())) != len(targets):
        duplicates = {path for path in targets.values() if list(targets.values()).count(path) > 1}
        raise SystemExit(f"configuracoes colidiram no mesmo artefato: {sorted(duplicates)}")

    expected = {
        "base": ("backend-cuvs", "arch-sm_80", "std-c++17", "raft-0"),
        "backend": ("backend-codes",),
        "arch": ("arch-sm_90",),
        "std": ("std-c++20",),
        "raft": ("raft-1",),
    }
    for label, fragments in expected.items():
        missing = [fragment for fragment in fragments if fragment not in targets[label]]
        if missing:
            raise SystemExit(
                f"alvo {label!r} nao codifica {missing}: {targets[label]}"
            )

    print("ok: artefatos distintos por BACKEND/CUDA_ARCH/STD/LINK_RAFT")
    for label, target in targets.items():
        print(f"  {label:8s} {target}")

    if args.dry_run:
        for _, backend, arch, std, raft in CONFIGS:
            run(
                [
                    *make_args(args.make, backend, arch, std, raft),
                    "-B",
                    "-n",
                    "all",
                ],
                quiet=True,
            )
        print("ok: dry-run da matriz de build")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
