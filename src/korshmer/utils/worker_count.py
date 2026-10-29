from __future__ import annotations

import os

KIND_IO = "io"
KIND_COMPUTE = "compute"
KIND_SYNTH = "synth"

MAX_SYNTH_CORES = 4


def cpu_cores() -> int:
    return max(1, os.cpu_count() or 1)


def compute_worker_count(kind: str, cpu_count: int | None = None) -> int:
    cores = cpu_cores() if cpu_count is None else max(1, int(cpu_count))
    if kind == KIND_IO:
        return max(2, cores)
    if kind == KIND_COMPUTE:
        return max(1, cores - 1)
    if kind == KIND_SYNTH:
        return min(cores, MAX_SYNTH_CORES)
    raise ValueError(f"Unknown worker kind: {kind!r}")
