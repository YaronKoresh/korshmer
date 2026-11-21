import os
import threading
from types import SimpleNamespace

import psutil
import pytest

from korshmer.workers import generic

GIB = 1024**3
PARAMS = dict.fromkeys(
    "octave root scale grid quantize swing vel_mode vel_val "
    "stems export gm_drums cc_gen wav32".split()
)


def peak_conversions_at_once(monkeypatch, cores, memory, songs):
    pytest.importorskip("PySide6")
    monkeypatch.setattr(generic, "_heavy_executor", None, raising=False)
    monkeypatch.setattr(generic, "_executor", None)
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(total=memory)
    )
    lock = threading.Lock()
    all_started = threading.Event()
    state = {"running": 0, "peak": 0, "started": 0}

    def convert(path, *args, progress_callback=None, check_stop_callback=None):
        with lock:
            state["running"] += 1
            state["started"] += 1
            state["peak"] = max(state["peak"], state["running"])
            if state["started"] == songs:
                all_started.set()
        all_started.wait(0.3)
        with lock:
            state["running"] -= 1
        return [path]

    worker = generic.WorkerThread(
        [__file__] * songs, PARAMS, process_track_func=convert
    )
    assert len(worker.execute()) == songs
    return state["peak"]


def test_little_memory_means_one_conversion_at_a_time(monkeypatch):
    assert peak_conversions_at_once(monkeypatch, 8, 8 * GIB, 4) == 1


def test_plenty_of_memory_is_still_limited_by_the_processor(monkeypatch):
    assert peak_conversions_at_once(monkeypatch, 3, 512 * GIB, 5) == 2


def test_many_cores_are_limited_by_memory(monkeypatch):
    assert peak_conversions_at_once(monkeypatch, 16, 13 * GIB, 6) == 2


def test_a_single_core_machine_still_converts_as_a_safeguard(monkeypatch):
    assert peak_conversions_at_once(monkeypatch, 1, 1 * GIB, 2) >= 1
