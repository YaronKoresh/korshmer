import os
import threading

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from korshmer.ui.widgets.arranger import timeline as timeline_module
from korshmer.ui.widgets.arranger.timeline import (
    ArrangerTimeline,
    AudioRecorderThread,
)

PROBLEM = "the microphone was unplugged"


class InputDevices:
    def query_devices(self, kind=None):
        return {"name": "Microphone", "kind": kind}


def broken_stream(*_args, **_kwargs):
    raise RuntimeError(PROBLEM)


def settle():
    for _ in range(5):
        QApplication.processEvents()


def write_tone(path, seconds=0.5):
    t = np.linspace(0, seconds, int(44100 * seconds), endpoint=False)
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * 220 * t)), 44100)
    return str(path)


@pytest.fixture
def warnings_shown(monkeypatch):
    shown = []
    monkeypatch.setattr(
        timeline_module.QMessageBox,
        "warning",
        lambda _parent, _title, text: shown.append(text),
    )
    return shown


@pytest.fixture
def failing_recorder(monkeypatch, tmp_path, warnings_shown):
    monkeypatch.setattr(timeline_module, "sd", InputDevices())
    monkeypatch.setattr(timeline_module, "open_input_stream", broken_stream)
    arranger = ArrangerTimeline()
    arranger.vm.recording_filename = lambda: str(tmp_path / "take.wav")
    said = []
    arranger.clip_edit_status.connect(said.append)
    arranger.show()
    settle()
    QTest.mouseClick(arranger.btn_rec, Qt.LeftButton)
    recorder = arranger.recorder
    if recorder is not None:
        recorder.wait(5000)
    settle()
    yield arranger, said
    arranger.close()


@pytest.fixture
def wav_path(tmp_path):
    return write_tone(tmp_path / "clip.wav")


@pytest.fixture
def analysed(monkeypatch):
    calls = []
    real_compute = timeline_module.compute_clip_waveform

    def spy(path, points=512):
        calls.append(path)
        return real_compute(path, points)

    monkeypatch.setattr(timeline_module, "compute_clip_waveform", spy)
    return calls


@pytest.fixture
def held_analysis(monkeypatch):
    gate = threading.Event()
    started = []
    real_compute = timeline_module.compute_clip_waveform

    def slow(path, points=512):
        started.append(path)
        gate.wait(30)
        return real_compute(path, points)

    monkeypatch.setattr(timeline_module, "compute_clip_waveform", slow)
    gate.started = started
    yield gate
    gate.set()


@pytest.fixture
def waits(monkeypatch):
    seen = []

    class WatchedWorker(timeline_module.ClipWaveformWorker):
        def wait(self, *args):
            seen.append(args)
            return super().wait(*args)

    monkeypatch.setattr(timeline_module, "ClipWaveformWorker", WatchedWorker)
    return seen


def test_a_recorder_that_cannot_open_the_input_reports_the_reason(
    qapp, monkeypatch, tmp_path
):
    monkeypatch.setattr(timeline_module, "open_input_stream", broken_stream)
    recorder = AudioRecorderThread(str(tmp_path / "take.wav"))
    reasons = []
    connect = getattr(recorder, "recording_failed", recorder.finished).connect
    connect(lambda *reason: reasons.append(reason))

    recorder.run()

    assert reasons == [(PROBLEM,)]


def test_a_failed_recording_puts_the_record_button_back(failing_recorder):
    arranger, _said = failing_recorder

    assert not arranger.btn_rec.isChecked()
    assert arranger.btn_rec.text() == "● REC"
    assert arranger.recorder is None
    assert arranger.live_waveform.isHidden()


def test_a_failed_recording_tells_the_user_why_in_the_status_line(
    failing_recorder,
):
    _arranger, said = failing_recorder

    assert said == [f"Recording stopped: {PROBLEM}"]


def test_a_failed_recording_shows_a_warning_with_the_reason(
    failing_recorder, warnings_shown
):
    assert warnings_shown == [f"Recording stopped: {PROBLEM}"]


def test_recording_can_be_tried_again_after_a_failure(failing_recorder):
    arranger, said = failing_recorder

    QTest.mouseClick(arranger.btn_rec, Qt.LeftButton)
    assert arranger.btn_rec.isChecked()
    arranger.recorder.wait(5000)
    settle()

    assert len(said) == 2
    assert not arranger.btn_rec.isChecked()


def test_clearing_the_editor_does_not_wait_for_waveform_analysis(
    qapp, wav_path, waits
):
    arranger = ArrangerTimeline()
    arranger.add_audio_clip(wav_path, name="Take")

    arranger.clear_editor()

    assert waits == []
    arranger.wait_for_waveforms()


def test_opening_a_project_does_not_wait_for_waveform_analysis(
    qapp, wav_path, waits
):
    arranger = ArrangerTimeline()
    arranger.add_audio_clip(wav_path, name="Take")
    saved = arranger.to_project_dict()

    arranger.load_project_dict(saved)

    assert waits == []
    arranger.wait_for_waveforms()


def test_a_waveform_asked_for_before_reopening_the_project_is_thrown_away(
    qapp, wav_path, held_analysis, waits
):
    arranger = ArrangerTimeline()
    arranger.add_audio_clip(wav_path, name="Take")
    saved = arranger.to_project_dict()
    applied = []
    apply = arranger.model.set_clip_waveform

    def counting(clip_id, waveform, duration=None):
        applied.append(clip_id)
        return apply(clip_id, waveform, duration)

    arranger.model.set_clip_waveform = counting

    arranger.load_project_dict(saved)
    assert waits == []
    held_analysis.set()
    arranger.wait_for_waveforms()

    assert len(applied) == 1
    assert len(arranger.model.clips[0].waveform) == 512


def test_opening_projects_in_a_row_does_not_pile_up_waveform_analysis(
    qapp, wav_path, held_analysis, waits
):
    arranger = ArrangerTimeline()
    arranger.add_audio_clip(wav_path, name="Take")
    saved = arranger.to_project_dict()

    for _ in range(6):
        arranger.load_project_dict(saved)
    assert waits == []
    held_analysis.set()
    arranger.wait_for_waveforms()

    assert len(held_analysis.started) <= 3
    assert len(arranger.model.clips[0].waveform) == 512


def test_undoing_a_clear_still_draws_the_waveform_of_the_clip(
    qapp, wav_path, held_analysis
):
    arranger = ArrangerTimeline()
    arranger.add_audio_clip(wav_path, name="Take")
    arranger.clear_editor()

    assert arranger.undo()
    held_analysis.set()
    arranger.wait_for_waveforms()

    assert len(arranger.model.clips[0].waveform) == 512


def test_a_clip_that_cannot_be_read_is_still_not_analysed_again_and_again(
    qapp, wav_path, monkeypatch
):
    asked = []

    def unreadable(path, points=512):
        asked.append(path)
        return [], 0.0

    monkeypatch.setattr(timeline_module, "compute_clip_waveform", unreadable)
    arranger = ArrangerTimeline()
    clip = arranger.add_audio_clip(wav_path, name="Take")
    arranger.wait_for_waveforms()

    for end in (0.4, 0.3, 0.2):
        assert arranger.piano_roll.trim_clip(clip.id, 0.0, end)
        arranger.wait_for_waveforms()

    assert asked == [wav_path]


def test_a_file_that_was_still_being_written_is_analysed_once_it_is_complete(
    qapp, wav_path, monkeypatch
):
    asked = []
    real_compute = timeline_module.compute_clip_waveform

    def incomplete_at_first(path, points=512):
        asked.append(path)
        if len(asked) == 1:
            return [], 0.0
        return real_compute(path, points)

    monkeypatch.setattr(
        timeline_module, "compute_clip_waveform", incomplete_at_first
    )
    arranger = ArrangerTimeline()
    clip = arranger.add_audio_clip(wav_path, name="Bounce")
    arranger.wait_for_waveforms()
    assert clip.waveform == []

    write_tone(wav_path, seconds=0.75)
    os.utime(wav_path, (1_900_000_000, 1_900_000_000))
    assert arranger.piano_roll.trim_clip(clip.id, 0.0, 0.4)
    arranger.wait_for_waveforms()

    assert asked == [wav_path, wav_path]
    assert len(clip.waveform) == 512
