import sys
import types

from korshmer.ui.managers.performance import PerformanceMonitor


def test_the_second_reading_reports_cpu_measured_since_the_first(monkeypatch):
    class Process:
        def __init__(self):
            self.readings = 0

        def cpu_percent(self):
            self.readings += 1
            return 0.0 if self.readings == 1 else 37.5

        def memory_info(self):
            return types.SimpleNamespace(rss=64 * 1024 * 1024)

        def num_threads(self):
            return 3

    monkeypatch.setitem(
        sys.modules, "psutil", types.SimpleNamespace(Process=Process)
    )
    monitor = PerformanceMonitor()
    monitor.last_update = 0
    monitor.update()
    monitor.last_update = 0
    assert monitor.update()["cpu_current"] == 37.5
