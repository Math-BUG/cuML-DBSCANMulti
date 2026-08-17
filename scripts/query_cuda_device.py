#!/usr/bin/env python3
"""Consulta o dispositivo CUDA logico 0, respeitando CUDA_VISIBLE_DEVICES.

Sem ``--field``, emite um objeto JSON. Com ``--field NOME``, emite somente o
valor solicitado, o que permite reutilizar a mesma consulta em scripts shell.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from typing import Any


LOGICAL_DEVICE = 0


def _nvidia_driver_version(identifier: str | None) -> str | None:
    if not identifier:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={identifier}",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip().splitlines()
    return value[0].strip() if completed.returncode == 0 and value else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def _uuid(value: Any) -> str | None:
    """Normaliza o UUID binario devolvido por cudaGetDeviceProperties."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        raw = bytes(value)
    except (TypeError, ValueError):
        try:
            raw = bytes(int(item) & 0xFF for item in value)
        except (TypeError, ValueError):
            return None
    if not raw or not any(raw):
        return None
    if len(raw) == 16:
        return f"GPU-{uuid.UUID(bytes=raw)}"
    return raw.hex()


def query_device() -> dict[str, Any]:
    try:
        import cupy as cp
    except ImportError as exc:
        raise SystemExit(
            "erro: CuPy ausente; ative o venv RAPIDS antes de consultar a GPU"
        ) from exc

    try:
        visible_count = int(cp.cuda.runtime.getDeviceCount())
        if visible_count < 1:
            raise RuntimeError("nenhum dispositivo CUDA visivel")

        device = cp.cuda.Device(LOGICAL_DEVICE)
        with device:
            properties = cp.cuda.runtime.getDeviceProperties(LOGICAL_DEVICE)
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            try:
                pci_bus_id = cp.cuda.runtime.deviceGetPCIBusId(LOGICAL_DEVICE)
            except Exception:
                pci_bus_id = getattr(device, "pci_bus_id", None)
                if callable(pci_bus_id):
                    pci_bus_id = pci_bus_id()
    except Exception as exc:
        raise SystemExit(
            f"erro: nao foi possivel consultar o dispositivo CUDA logico 0: {exc}"
        ) from exc

    major = int(properties["major"])
    minor = int(properties["minor"])
    free_bytes = int(free_bytes)
    total_bytes = int(total_bytes)
    device_uuid = _uuid(properties.get("uuid"))
    pci_bus_id = _text(pci_bus_id)
    return {
        "logical_device": LOGICAL_DEVICE,
        "visible_device_count": visible_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": _text(properties.get("name")),
        "uuid": device_uuid,
        "pci_bus_id": pci_bus_id,
        "compute_capability": f"{major}.{minor}",
        "compute_capability_digits": f"{major}{minor}",
        "cuda_arch": f"sm_{major}{minor}",
        "free_memory_bytes": free_bytes,
        "total_memory_bytes": total_bytes,
        # MiB preserva a unidade historicamente retornada por nvidia-smi, embora os
        # scripts antigos a chamassem apenas de MB.
        "free_memory_mib": free_bytes // (1024 * 1024),
        "total_memory_mib": total_bytes // (1024 * 1024),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "nvidia_driver_version": _nvidia_driver_version(device_uuid or pci_bus_id),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cupy_version": cp.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field",
        help="emite somente este campo em vez do objeto JSON completo",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indenta o JSON (ignorado quando --field e usado)",
    )
    args = parser.parse_args()

    result = query_device()
    if args.field:
        if args.field not in result:
            parser.error(
                f"campo desconhecido {args.field!r}; disponiveis: "
                + ", ".join(sorted(result))
            )
        value = result[args.field]
        if isinstance(value, (dict, list, bool)) or value is None:
            print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        else:
            print(value)
    else:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                indent=2 if args.pretty else None,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
