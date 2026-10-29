import threading
import time
from types import SimpleNamespace

import pytest

from korshmer.network.health_server import HealthServer
from korshmer.processing.queue import ProcessingQueue
from korshmer.utils import executor as core_executor, tracing as tracing_module
from korshmer.utils.perf_metrics import PerfMetrics
from korshmer.utils.tracing import (
    Tracer,
    current_ids,
    propagate,
    tracer,
)


@pytest.fixture(autouse=True)
def clean_tracer():
    tracer.reset()
    yield
    tracer.reset()


class TestSpans:
    def test_nested_spans_share_trace_and_link_parents(self):
        t = Tracer()
        with t.span("outer") as outer:
            with t.span("inner") as inner:
                assert inner.trace_id == outer.trace_id
                assert inner.parent_id == outer.span_id
        assert outer.parent_id is None
        names = [s.name for s in t.spans()]
        assert names == ["inner", "outer"]

    def test_separate_roots_get_separate_traces(self):
        t = Tracer()
        with t.span("a") as a:
            pass
        with t.span("b") as b:
            pass
        assert a.trace_id != b.trace_id

    def test_error_is_recorded_and_reraised(self):
        t = Tracer()
        with pytest.raises(ValueError):
            with t.span("boom"):
                raise ValueError("bad value")
        (span,) = t.spans()
        assert span.status == "error"
        assert span.error == "ValueError: bad value"
        assert span.duration_ms is not None

    def test_context_cleared_after_exit(self):
        t = Tracer()
        with t.span("x"):
            assert current_ids() is not None
        assert current_ids() is None

    def test_decorator_keeps_name_and_result(self):
        t = Tracer()

        @t.trace("work")
        def work(a, b=1):
            return a + b

        assert work(1, b=2) == 3
        assert work.__name__ == "work"
        assert [s.name for s in t.spans()] == ["work"]

    def test_buffer_drops_oldest(self):
        t = Tracer(max_spans=3)
        for i in range(5):
            with t.span(f"s{i}"):
                pass
        assert [s.name for s in t.spans()] == ["s2", "s3", "s4"]

    def test_invalid_size_rejected(self):
        with pytest.raises(ValueError):
            Tracer(max_spans=0)

    def test_attributes_are_stored(self):
        t = Tracer()
        with t.span("load", path="a.wav"):
            pass
        assert t.spans()[0].attributes == {"path": "a.wav"}


class TestSnapshot:
    def test_groups_spans_by_trace(self):
        t = Tracer()
        with t.span("root"):
            with t.span("child"):
                pass
        with t.span("other"):
            pass
        snap = t.snapshot()
        assert len(snap) == 2
        by_name = {tr["name"]: tr for tr in snap}
        assert by_name["root"]["span_count"] == 2
        assert by_name["root"]["status"] == "ok"
        assert [s["name"] for s in by_name["root"]["spans"]] == [
            "root",
            "child",
        ]

    def test_trace_with_failed_span_is_error(self):
        t = Tracer()
        with pytest.raises(RuntimeError):
            with t.span("root"):
                with t.span("child"):
                    raise RuntimeError("x")
        assert t.snapshot()[0]["status"] == "error"

    def test_limit_and_newest_first(self):
        t = Tracer()
        for i in range(4):
            with t.span(f"t{i}"):
                pass
        snap = t.snapshot(limit=2)
        assert [tr["name"] for tr in snap] == ["t3", "t2"]
        assert t.snapshot(limit=0) == []

    def test_order_survives_a_coarse_wall_clock(self, monkeypatch):
        coarse = SimpleNamespace(
            time=lambda: 1000.0, perf_counter=time.perf_counter
        )
        monkeypatch.setattr(tracing_module, "time", coarse)
        t = Tracer()
        with t.span("root"):
            with t.span("child"):
                pass
        for i in range(3):
            with t.span(f"t{i}"):
                pass
        snap = t.snapshot()
        assert [tr["name"] for tr in snap] == ["t2", "t1", "t0", "root"]
        assert [s["name"] for s in snap[-1]["spans"]] == ["root", "child"]


class TestThreads:
    def test_plain_thread_does_not_inherit_trace(self):
        t = Tracer()
        seen = []
        with t.span("root"):
            th = threading.Thread(target=lambda: seen.append(current_ids()))
            th.start()
            th.join()
        assert seen == [None]

    def test_propagate_carries_trace_into_thread(self):
        t = Tracer()
        seen = {}

        def work():
            with t.span("child") as child:
                seen["child"] = child

        with t.span("root") as root:
            th = threading.Thread(target=propagate(work))
            th.start()
            th.join()
        assert seen["child"].trace_id == root.trace_id
        assert seen["child"].parent_id == root.span_id
        assert seen["child"].thread != root.thread

    def test_run_async_continues_the_trace(self, monkeypatch):
        monkeypatch.setattr(core_executor, "_executor_shutdown", False)
        seen = {}

        def work():
            seen["ids"] = current_ids()

        with tracer.span("root") as root:
            core_executor.run_async(work).result(timeout=5)
        assert seen["ids"][0] == root.trace_id

    def test_processing_queue_continues_the_trace(self):
        queue = ProcessingQueue(max_workers=1)
        seen = {}

        def work():
            seen["ids"] = current_ids()
            return 1

        try:
            with tracer.span("root") as root:
                task = queue.submit(work)
            queue.drain(timeout=5)
            for _ in range(200):
                if task.future is not None:
                    break
                threading.Event().wait(0.01)
            assert task.future.result(timeout=5) == 1
        finally:
            queue.shutdown()
        assert seen["ids"][0] == root.trace_id


class TestIntegration:
    def test_perf_timed_opens_a_span(self):
        m = PerfMetrics()
        with tracer.span("root") as root:
            with m.timed("audio.step"):
                pass
        spans = {s.name: s for s in tracer.spans()}
        assert spans["audio.step"].parent_id == root.span_id
        assert m.snapshot()["audio.step"]["count"] == 1

    def test_perf_timed_failure_marks_span(self):
        m = PerfMetrics()
        with pytest.raises(KeyError):
            with m.timed("audio.fail"):
                raise KeyError("k")
        assert tracer.spans()[0].status == "error"
        assert m.snapshot()["audio.fail"]["errors"] == 1

    def test_health_endpoint_returns_traces(self):
        with tracer.span("op"):
            pass
        health = HealthServer()
        code, body = health.handle("/health/traces")
        assert code == 200
        assert body["status"] == "ok"
        assert body["traces"][0]["name"] == "op"
