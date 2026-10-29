import threading

import pytest

from korshmer.network.health_server import HealthServer
from korshmer.utils.perf_metrics import PerfMetrics, metrics


class TestPerfMetrics:
    def test_timed_records_one_sample(self):
        m = PerfMetrics()
        with m.timed("op"):
            pass
        snap = m.snapshot()
        assert snap["op"]["count"] == 1
        assert snap["op"]["errors"] == 0
        assert snap["op"]["max_ms"] >= snap["op"]["min_ms"] >= 0

    def test_exception_is_counted_and_reraised(self):
        m = PerfMetrics()
        with pytest.raises(ValueError):
            with m.timed("op"):
                raise ValueError("boom")
        snap = m.snapshot()
        assert snap["op"]["count"] == 1
        assert snap["op"]["errors"] == 1

    def test_measure_wraps_function_and_keeps_result(self):
        m = PerfMetrics()

        def add(a, b):
            return a + b

        add.__doc__ = "Adds."
        measured = m.measure("add")(add)

        assert measured(1, 2) == 3
        assert measured(3, 4) == 7
        assert measured.__name__ == "add"
        assert measured.__doc__ == "Adds."
        assert m.snapshot()["add"]["count"] == 2

    def test_aggregates_and_p95(self):
        m = PerfMetrics()
        for ms in range(1, 101):
            m.record("op", ms / 1000)
        s = m.snapshot()["op"]
        assert s["count"] == 100
        assert s["min_ms"] == pytest.approx(1.0)
        assert s["max_ms"] == pytest.approx(100.0)
        assert s["mean_ms"] == pytest.approx(50.5)
        assert s["p95_ms"] == pytest.approx(95.0)
        assert s["last_ms"] == pytest.approx(100.0)

    def test_sample_window_is_bounded_but_totals_are_not(self):
        m = PerfMetrics(window=10)
        for _i in range(100):
            m.record("op", 0.001)
        m.record("op", 1.0)
        s = m.snapshot()["op"]
        assert s["count"] == 101
        assert s["total_ms"] == pytest.approx(1100.0)
        assert s["p95_ms"] == pytest.approx(1000.0)

    def test_negative_duration_is_clamped(self):
        m = PerfMetrics()
        m.record("op", -5.0)
        assert m.snapshot()["op"]["min_ms"] == 0

    def test_invalid_window_rejected(self):
        with pytest.raises(ValueError):
            PerfMetrics(window=0)

    def test_reset_clears_everything(self):
        m = PerfMetrics()
        m.record("op", 0.1)
        m.reset()
        assert m.snapshot() == {}
        assert m.report() == "No performance metrics recorded."

    def test_report_lists_names_sorted(self):
        m = PerfMetrics()
        m.record("b", 0.002)
        m.record("a", 0.001, failed=True)
        lines = m.report().splitlines()
        assert lines[0].startswith("a: 1 calls, 1 errors")
        assert lines[1].startswith("b: 1 calls, 0 errors")

    def test_thread_safe_counts(self):
        m = PerfMetrics()

        def work():
            for _ in range(500):
                m.record("op", 0.0001)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert m.snapshot()["op"]["count"] == 4000


class TestIntegration:
    def test_file_ops_are_recorded(self, tmp_path):
        from korshmer.project_management.file_io_log import logged_file_op

        metrics.reset()
        target = tmp_path / "a.txt"
        with logged_file_op("Save thing", target, report_size=False):
            target.write_text("x")
        with pytest.raises(OSError):
            with logged_file_op("Save thing", target, report_size=False):
                raise OSError("disk")
        s = metrics.snapshot()["file_io.Save thing"]
        assert s["count"] == 2
        assert s["errors"] == 1

    def test_health_server_serves_metrics(self):
        metrics.reset()
        metrics.record("demo", 0.01)
        code, body = HealthServer().handle("/health/metrics")
        assert code == 200
        assert body["status"] == "ok"
        assert body["metrics"]["demo"]["count"] == 1
