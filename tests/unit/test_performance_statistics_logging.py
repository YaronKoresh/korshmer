import logging
import sys
import types

import numpy as np

from korshmer.constants import PERFORMANCE_CPU_SPIKE_PERCENT
from korshmer.core.audio_engine import AudioEngine
from korshmer.ui.managers.performance import PerformanceMonitor


def _fake_psutil(monkeypatch, values):
    it = iter(values)

    class Proc:
        def cpu_percent(self):
            return next(it)

        def memory_info(self):
            return types.SimpleNamespace(rss=100 * 1024 * 1024)

        def num_threads(self):
            return 1

    monkeypatch.setitem(
        sys.modules, "psutil", types.SimpleNamespace(Process=Proc)
    )


def test_cpu_spike_logged_once_per_spike(monkeypatch, caplog):
    high = PERFORMANCE_CPU_SPIKE_PERCENT + 5
    _fake_psutil(monkeypatch, [10, high, high, 10, high])
    mon = PerformanceMonitor()
    with caplog.at_level(logging.WARNING, logger="korshmer.performance"):
        for _ in range(5):
            mon.last_update = 0
            assert mon.update() is not None
    spikes = [r for r in caplog.records if "CPU spike" in r.message]
    assert len(spikes) == 2


class _Status:
    output_underflow = True

    def __bool__(self):
        return True


def test_total_underruns_survive_buffer_tracking_reset(caplog):
    eng = AudioEngine()
    eng.data = np.zeros((100, 1), dtype="float32")
    eng.fs = 1000
    out = np.zeros((10, 1), dtype="float32")
    eng._cb_main(out, 10, None, _Status())
    eng._cb_main(out, 10, None, _Status())
    assert eng._total_underruns == 2
    eng._reset_buffer_tracking(0.0)
    assert eng._underrun_count == 0
    assert eng._total_underruns == 2
    with caplog.at_level(logging.WARNING, logger="korshmer.performance"):
        eng._log_audio_dropouts()
        eng._log_audio_dropouts()
    msgs = [r.message for r in caplog.records if "dropouts" in r.message]
    assert len(msgs) == 1 and "total 2" in msgs[0]


def test_output_latency_logged(monkeypatch, caplog):
    monkeypatch.setattr(
        "korshmer.core.audio_engine.active_output_latency", lambda: 0.05
    )
    eng = AudioEngine()
    eng.fs = 44100
    with caplog.at_level(logging.INFO, logger="korshmer.performance"):
        eng._log_output_latency()
    assert any("Output latency: 50.0 ms" in r.message for r in caplog.records)
