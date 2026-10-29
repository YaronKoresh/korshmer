import sys

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from korshmer.ui.viewmodels.editor_vm import MAX_CLIP_GAIN
from korshmer.ui.widgets.arranger import timeline as timeline_module
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def waveform_calls(monkeypatch):
    calls = []
    real_compute = timeline_module.compute_clip_waveform

    def spy(path, points=512):
        calls.append(path)
        return real_compute(path, points)

    monkeypatch.setattr(timeline_module, "compute_clip_waveform", spy)
    return calls


@pytest.fixture
def wav_path(tmp_path):
    path = tmp_path / "clip.wav"
    t = np.linspace(0, 0.5, int(44100 * 0.5), endpoint=False)
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * 220 * t)), 44100)
    return str(path)


def test_clip_outside_viewport_is_not_analysed_until_scrolled(
    qapp, wav_path, waveform_calls
):
    arranger = ArrangerTimeline()
    arranger.resize(1200, 500)
    arranger.show()
    qapp.processEvents()

    clip = arranger.add_audio_clip(wav_path, start_sec=600.0, name="Far")
    arranger.wait_for_waveforms()
    assert waveform_calls == []
    assert clip.waveform == []

    bar = arranger.piano_roll.horizontalScrollBar()
    bar.setValue(bar.maximum())
    qapp.processEvents()
    arranger.wait_for_waveforms()

    assert len(waveform_calls) == 1
    assert len(clip.waveform) == 512


def test_clip_on_screen_is_analysed_when_added(qapp, wav_path, waveform_calls):
    arranger = ArrangerTimeline()
    arranger.resize(1200, 500)
    arranger.show()
    qapp.processEvents()

    clip = arranger.add_audio_clip(wav_path, start_sec=0.0, name="Near")
    arranger.wait_for_waveforms()

    assert len(waveform_calls) == 1
    assert len(clip.waveform) == 512


def test_hidden_editor_keeps_analysing_clips_immediately(
    wav_path, waveform_calls
):
    arranger = ArrangerTimeline()
    clip = arranger.add_audio_clip(wav_path, start_sec=600.0, name="Hidden")
    arranger.wait_for_waveforms()

    assert len(waveform_calls) == 1
    assert len(clip.waveform) == 512


def test_clip_gain_is_applied_and_limited(qapp, wav_path, waveform_calls):
    arranger = ArrangerTimeline()
    boosted = arranger.add_audio_clip(wav_path, name="Boost", gain=2.5)
    plain = arranger.add_audio_clip(wav_path, name="Plain")
    loud = arranger.add_audio_clip(wav_path, name="Loud", gain=99.0)
    arranger.wait_for_waveforms()
    assert boosted.gain == 2.5
    assert plain.gain == 1.0
    assert loud.gain == MAX_CLIP_GAIN
