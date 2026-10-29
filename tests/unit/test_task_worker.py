import pytest

from korshmer.workers.base import TaskWorker


class SuccessWorker(TaskWorker):
    def execute(self):
        return "done"


class ErrorWorker(TaskWorker):
    def execute(self):
        raise ValueError("test error")


class ProgressWorker(TaskWorker):
    def __init__(self):
        super().__init__()
        self.reported = []

    def execute(self):
        self.report_progress(50, "halfway")
        return "complete"


class StoppableWorker(TaskWorker):
    def execute(self):
        self.check_stopped()
        return "should not reach"


class TestTaskWorkerInit:
    def test_defaults(self):
        w = SuccessWorker()
        assert w.is_running is True

    def test_stop(self):
        w = SuccessWorker()
        w.stop()
        assert w.is_running is False

    def test_is_running_setter(self):
        w = SuccessWorker()
        w.is_running = False
        assert w.is_running is False
        w.is_running = True
        assert w.is_running is True


class TestCheckStopped:
    def test_running_no_exception(self):
        w = StoppableWorker()
        w.check_stopped()

    def test_stopped_raises(self):
        w = StoppableWorker()
        w.stop()
        with pytest.raises(InterruptedError):
            w.check_stopped()


class TestExecute:
    def test_success_worker(self):
        w = SuccessWorker()
        assert w.execute() == "done"

    def test_error_worker(self):
        w = ErrorWorker()
        with pytest.raises(ValueError, match="test error"):
            w.execute()


class TestReportProgress:
    def test_report_when_running(self, qtbot):
        w = ProgressWorker()
        signals = []
        w.progress_signal.connect(lambda v, m: signals.append((v, m)))
        w.report_progress(75, "three quarters")
        assert len(signals) == 1
        assert signals[0] == (75, "three quarters")

    def test_report_when_stopped(self, qtbot):
        w = ProgressWorker()
        signals = []
        w.progress_signal.connect(lambda v, m: signals.append((v, m)))
        w.stop()
        w.report_progress(50, "should not emit")
        assert len(signals) == 0


class TestRunMethod:
    def test_run_emits_finished(self, qtbot):
        w = SuccessWorker()
        results = []
        w.finished_signal.connect(lambda r: results.append(r))
        w.run()
        assert results == ["done"]

    def test_run_emits_error_on_exception(self, qtbot):
        w = ErrorWorker()
        errors = []
        w.error_signal.connect(lambda e: errors.append(e))
        w.run()
        assert len(errors) == 1
        assert "test error" in errors[0]

    def test_run_emits_cancelled_on_interrupt(self, qtbot):
        w = StoppableWorker()
        w.stop()
        errors = []
        w.error_signal.connect(lambda e: errors.append(e))
        w.run()
        assert errors == ["Cancelled"]

    def test_run_no_finished_when_stopped(self, qtbot):
        w = StoppableWorker()
        w.stop()
        results = []
        w.finished_signal.connect(lambda r: results.append(r))
        w.run()
        assert results == []


class TestChordAnalysisThread:
    def test_inherits_task_worker(self):
        from korshmer.workers.chord_analysis import ChordAnalysisThread

        assert issubclass(ChordAnalysisThread, TaskWorker)

    def test_has_execute(self):
        from korshmer.workers.chord_analysis import ChordAnalysisThread

        t = ChordAnalysisThread("/fake/path", "/fake/output")
        assert hasattr(t, "execute")
        assert hasattr(t, "finished_signal")
        assert hasattr(t, "error_signal")


class TestMidiPreviewThread:
    def test_inherits_task_worker(self):
        from korshmer.workers.midi_preview import MidiPreviewThread

        assert issubclass(MidiPreviewThread, TaskWorker)

    def test_has_stop(self):
        from korshmer.workers.midi_preview import MidiPreviewThread

        t = MidiPreviewThread("/fake.mid", "/fluid", "/sf3", "/out.wav")
        t.stop()
        assert t.is_running is False

    def test_has_preview_signals(self):
        from korshmer.workers.midi_preview import MidiPreviewThread

        t = MidiPreviewThread("/fake.mid", "/fluid", "/sf3", "/out.wav")
        assert hasattr(t, "preview_ready")
        assert hasattr(t, "preview_failed")


class TestMidiPreviewDefaultRenderer:
    def test_uses_midi_renderer_when_none_given(self, monkeypatch):
        from korshmer import audio_processors
        from korshmer.workers.midi_preview import MidiPreviewThread

        calls = []

        def fake_render(midi, out, fluid, sf3, bpm=120.0, stop_callback=None):
            calls.append((midi, out, fluid, sf3, bpm))
            return out

        monkeypatch.setattr(audio_processors, "render_midi_to_wav", fake_render)
        t = MidiPreviewThread("/a.mid", "/f", "/s.sf3", "/o.wav", bpm=100.0)
        ready, failed = [], []
        t.preview_ready.connect(ready.append)
        t.preview_failed.connect(failed.append)
        t.execute()
        assert calls == [("/a.mid", "/o.wav", "/f", "/s.sf3", 100.0)]
        assert ready == ["/o.wav"]
        assert failed == []
