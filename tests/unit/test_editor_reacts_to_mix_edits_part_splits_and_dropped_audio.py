import inspect

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl, Signal
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication

from korshmer.audio_processing import clip_polish
from korshmer.ui.viewmodels.editor_vm import ClipModel
from korshmer.ui.widgets.arranger import timeline as timeline_module
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.ui.widgets.arranger.track_panel import TrackPanel
from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

RATE = 44100


class MixingTrackPanel(TrackPanel):
    mix_changed = Signal()


class PannedClip(ClipModel):
    pan = 0.0


def settle():
    for _ in range(5):
        QApplication.processEvents()


def write_tone(path, seconds=1.0, **options):
    t = np.linspace(0, seconds, int(RATE * seconds), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(path), tone, RATE, **options)
    return str(path)


def drop(roll, path):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(
        QPointF(40, 40), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier
    )
    roll.dropEvent(event)
    return event


def panned(timeline, clip, pan):
    if hasattr(clip, "pan"):
        clip.pan = pan
        return clip
    stand_in = PannedClip(
        clip.id, clip.path, clip.name, clip.start, clip.duration
    )
    stand_in.pan = pan
    timeline.model.clips[timeline.model.clips.index(clip)] = stand_in
    return stand_in


def test_a_level_or_pan_edit_enables_undo_without_rebuilding_the_notes(
    qapp, monkeypatch
):
    monkeypatch.setattr(timeline_module, "TrackPanel", MixingTrackPanel)
    timeline = ArrangerTimeline()
    timeline.load_project_dict(timeline.to_project_dict())
    track = timeline.model.ensure_default_track()
    timeline.model.add_note(track.id, 60, 0.0, 0.5, 90)
    timeline.piano_roll.load_model()
    drawn = list(timeline.piano_roll.scene.items())
    assert not timeline.btn_undo.isEnabled()

    timeline.model.snapshot()
    timeline.track_panel.mix_changed.emit()

    assert timeline.btn_undo.isEnabled()
    assert list(timeline.piano_roll.scene.items()) == drawn


def test_the_parts_of_a_split_clip_keep_its_place_between_the_speakers(
    qapp, tmp_path, monkeypatch
):
    recording = write_tone(tmp_path / "take.wav", seconds=2.0)
    halves = [
        write_tone(tmp_path / "first.wav"),
        write_tone(tmp_path / "second.wav"),
    ]

    def split(_audio, _steps, *_rest, **_more):
        return {
            "parts": [
                {"path": halves[0], "start": 0.0},
                {"path": halves[1], "start": 1.0},
            ],
            "made": [],
        }

    monkeypatch.setattr(clip_polish, "fix_clip_parts", split)
    timeline = ArrangerTimeline()
    timeline.notes_audio = lambda _song, _tempo, _rate: None
    clip = panned(
        timeline, timeline.add_audio_clip(recording, name="Take"), -0.5
    )
    timeline.wait_for_waveforms()
    add_clip = timeline.model.add_clip
    accepts_pan = "pan" in inspect.signature(add_clip).parameters
    asked = []

    def recording_add(*args, **options):
        asked.append(options.get("pan"))
        if not accepts_pan:
            options.pop("pan", None)
        return add_clip(*args, **options)

    timeline.model.add_clip = recording_add

    timeline.fix_clip_automatically(clip.id)
    timeline.wait_for_clip_edits()
    settle()

    assert len(timeline.model.clips) == 2
    assert asked == [-0.5, -0.5]


@pytest.mark.parametrize("suffix", [".aac", ".au", ".AAC"])
def test_dropping_a_compressed_or_sun_audio_file_is_taken_as_audio(
    qapp, tmp_path, suffix
):
    path = tmp_path / f"take{suffix}"
    path.write_bytes(b"audio")
    roll = InteractivePianoRoll()
    dropped = []
    roll.audio_dropped.connect(dropped.append)

    event = drop(roll, path)

    assert [d.replace("\\", "/") for d in dropped] == [str(path).replace("\\", "/")]
    assert event.isAccepted()


def test_a_dropped_text_file_is_still_not_taken_as_audio(qapp, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"words")
    roll = InteractivePianoRoll()
    dropped = []
    roll.audio_dropped.connect(dropped.append)

    drop(roll, path)

    assert dropped == []


def test_a_sun_audio_file_dropped_on_the_editor_becomes_a_clip(qapp, tmp_path):
    path = write_tone(tmp_path / "take.au", format="AU", subtype="PCM_16")
    timeline = ArrangerTimeline()

    drop(timeline.piano_roll, path)
    timeline.wait_for_waveforms()

    assert [clip.name for clip in timeline.model.clips] == ["take"]
    assert len(timeline.model.clips[0].waveform) == 512
