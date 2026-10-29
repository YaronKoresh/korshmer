import time

from korshmer.workers.base import CallableWorker
from korshmer.workers.stem_separation import StemSeparationThread


def _make_loop(worker_holder):
    def run():
        for _ in range(200):
            worker_holder[0].check_stopped()
            time.sleep(0.01)
        return "ok"

    return run


def test_timeout_emits_error():
    holder = [None]
    w = CallableWorker(_make_loop(holder), timeout_seconds=0.1)
    holder[0] = w
    errors, done = [], []
    w.error_signal.connect(errors.append)
    w.finished_signal.connect(done.append)
    w.run()
    assert errors and "Timed out" in errors[0]
    assert not done


def test_no_timeout_finishes():
    holder = [None]
    w = CallableWorker(_make_loop(holder), timeout_seconds=None)
    holder[0] = w
    errors, done = [], []
    w.error_signal.connect(errors.append)
    w.finished_signal.connect(done.append)
    w.run()
    assert done == ["ok"] and not errors


def test_cancel_message_differs_from_timeout():
    holder = [None]
    w = CallableWorker(_make_loop(holder), timeout_seconds=5)
    holder[0] = w
    errors = []
    w.error_signal.connect(errors.append)
    w.stop()
    w.run()
    assert errors == ["Cancelled"]


def test_stem_separation_timeout(monkeypatch):
    def fake_separate(*args, check_stop_callback=None, **kwargs):
        for _ in range(200):
            if check_stop_callback():
                raise InterruptedError("Stopped by user")
            time.sleep(0.01)

    monkeypatch.setattr(
        "korshmer.audio_processing.stem_session.separate_stems", fake_separate
    )
    t = StemSeparationThread("x.wav", timeout_seconds=0.1)
    failed, errors = [], []
    t.separation_failed.connect(failed.append)
    t.error_signal.connect(errors.append)
    t.run()
    assert failed and "Timed out" in failed[0]
    assert errors and "Timed out" in errors[0]
