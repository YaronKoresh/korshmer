import os
import sys

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication, QMessageBox

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer import midi_algorithms, midi_processors
from korshmer.audio_processing.tempo_align import (
    align_clip_to_tempo,
    stretch_to_tempo,
)
from korshmer.midi.tempo_map import TempoMap
from korshmer.ui.viewmodels.editor_vm import (
    EditorModel,
    TempoNormalizeError,
)
from korshmer.ui.widgets.arranger import timeline
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

SR = 22050
SONG = TempoMap((0.0, 12.0), (100.0, 110.0))
WILD = TempoMap((0.0, 12.0), (100.0, 150.0))
END_BEAT = 60


def _song_midi(tempo_map=SONG, beats=range(0, END_BEAT)):
    pm = tempo_map.new_pretty_midi()
    inst = pretty_midi.Instrument(0)
    for beat in beats:
        start = float(tempo_map.seconds_at(beat))
        inst.notes.append(pretty_midi.Note(90, 60, start, start + 0.1))
    pm.instruments.append(inst)
    return pm


def _clicks(tempo_map=SONG, beats=range(1, END_BEAT - 1, 2)):
    seconds = float(tempo_map.seconds_at(END_BEAT))
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    burst = np.hanning(256) * np.sin(2 * np.pi * 2000 * np.arange(256) / SR)
    for beat in beats:
        first = int(tempo_map.seconds_at(beat) * SR)
        audio[first : first + 256] += burst.astype(np.float32)
    return audio


def _click_times(audio):
    level = np.abs(audio)
    frame = 256
    count = len(level) // frame
    energy = level[: count * frame].reshape(count, frame).max(1)
    peaks = np.flatnonzero(
        (energy > 0.3 * energy.max())
        & (energy >= np.roll(energy, 1))
        & (energy > np.roll(energy, -1))
    )
    return peaks * frame / SR


@pytest.fixture
def clip_path(tmp_path):
    path = tmp_path / "vocals.wav"
    sf.write(path, _clicks(), SR, subtype="PCM_16")
    return str(path)


def test_stretched_audio_has_its_beats_at_an_even_tempo():
    bpm = SONG.average_bpm(float(SONG.seconds_at(END_BEAT)))
    even = stretch_to_tempo(_clicks(), SR, 0.0, SONG, bpm)
    assert even.shape[0] == 1
    assert even.shape[1] / SR == pytest.approx(END_BEAT * 60 / bpm, abs=0.05)
    found = _click_times(even[0])
    expected = np.arange(1, END_BEAT - 1, 2) * 60 / bpm
    assert len(found) == len(expected)
    assert found == pytest.approx(expected, abs=0.03)


def test_aligned_clips_are_written_once_and_reused(clip_path, tmp_path):
    out_dir = str(tmp_path / "aligned")
    path, duration = align_clip_to_tempo(clip_path, 0.0, SONG, 105.0, out_dir)
    assert os.path.dirname(path) == out_dir
    assert duration == pytest.approx(END_BEAT * 60 / 105.0, abs=0.05)
    written = os.stat(path).st_mtime_ns
    again, _ = align_clip_to_tempo(clip_path, 0.0, SONG, 105.0, out_dir)
    assert again == path
    assert os.stat(path).st_mtime_ns == written
    other, _ = align_clip_to_tempo(clip_path, 1.0, SONG, 105.0, out_dir)
    assert other != path


def _fake_align(calls):
    def align(clip, tempo_map, bpm):
        calls.append((clip.path, bpm))
        length = tempo_map.warp(clip.end, bpm) - tempo_map.warp(clip.start, bpm)
        return clip.path + ".even.wav", length

    return align


def test_normalizing_evens_out_notes_and_clips_and_can_be_undone():
    model = EditorModel(tempo=110.0)
    model.load_midi(_song_midi())
    model.grid_offset = float(SONG.seconds_at(1))
    clip = model.add_clip("vocals.wav", start=3.0, name="Vocals", duration=20.0)
    before = [(n.start, n.duration) for n in model.notes]
    calls = []

    assert model.normalize_tempo(_fake_align(calls))
    bpm = model.tempo
    assert model.tempo_map is None
    assert 100.0 < bpm < 110.0
    starts = np.array(sorted(n.start for n in model.notes))
    assert np.diff(starts) == pytest.approx(60.0 / bpm, abs=1e-6)
    assert model.grid_offset == pytest.approx(60.0 / bpm)
    assert calls == [("vocals.wav", bpm)]
    assert clip.path == "vocals.wav.even.wav"
    assert clip.start == pytest.approx(SONG.warp(3.0, bpm))
    assert clip.source == ("vocals.wav", 3.0, 20.0)
    assert model.to_editor_time(12.0) == pytest.approx(SONG.warp(12.0, bpm))
    assert not model.normalize_tempo(_fake_align(calls))

    kept = EditorModel().load_state(model.to_dict())
    assert kept.normalized["bpm"] == bpm
    assert kept.clips[0].source == ("vocals.wav", 3.0, 20.0)

    for restored in (model, kept):
        assert restored.restore_tempo()
        assert restored.tempo == 110.0
        assert restored.tempo_map.bpms == pytest.approx(SONG.bpms, rel=1e-4)
        times = np.array([(n.start, n.duration) for n in restored.notes])
        assert times == pytest.approx(np.array(before), abs=1e-6)
        only = restored.clips[0]
        assert only.path == "vocals.wav"
        assert (only.start, only.duration) == pytest.approx((3.0, 20.0))
        assert only.source is None
        assert restored.to_editor_time(12.0) == 12.0
        assert not restored.restore_tempo()


def test_a_clip_that_needs_too_much_stretching_stops_the_normalizing():
    model = EditorModel(tempo=150.0)
    model.load_midi(_song_midi(WILD))
    model.add_clip("far.wav", start=0.0, name="Far off", duration=30.0)
    before = [n.start for n in model.notes]
    with pytest.raises(TempoNormalizeError) as caught:
        model.normalize_tempo(_fake_align([]))
    text = str(caught.value)
    assert "'Far off'" in text
    assert "Turn off Even BPM" in text
    assert "remove that audio clip" in text
    assert caught.value.clip_names == ["Far off"]
    assert model.tempo_map is not None
    assert model.normalized is None
    assert [n.start for n in model.notes] == before
    assert model.clips[0].path == "far.wav"


def test_note_tools_move_by_beats_of_the_local_tempo():
    model = EditorModel(tempo=110.0)
    model.load_midi(_song_midi(beats=(18, 19)))
    first, second = model.notes
    model.shift_notes_by_beats([first.id, second.id], 2.5)
    assert SONG.beats_at(first.start) == pytest.approx(20.5)
    assert SONG.beats_at(second.start) == pytest.approx(21.5)
    assert model.beats_length(11.7, 1.0) == pytest.approx(
        SONG.seconds_at(SONG.beats_at(11.7) + 1) - 11.7
    )
    copies = model.duplicate_notes([first.id, second.id], 60.0 / 110.0)
    assert SONG.beats_at(np.array([c.start for c in copies])) == pytest.approx(
        [22.5, 23.5]
    )
    plain = EditorModel(tempo=120.0)
    note = plain.add_note(plain.add_track("Piano").id, 60, 1.0, 0.5, 90)
    plain.shift_notes_by_beats([note.id], 1.0)
    assert note.start == pytest.approx(1.5)
    assert plain.beats_length(0.0, 2.0) == pytest.approx(1.0)


def test_exports_from_the_editor_keep_the_tempo_changes():
    model = EditorModel(tempo=110.0)
    assert model.load_midi(_song_midi(), replace=False)
    assert model.tempo_map is not None
    exported = model.to_pretty_midi(tempo=110.0)
    assert len(exported.get_tempo_changes()[1]) == 2
    forced = model.to_pretty_midi(tempo=95.0)
    assert list(forced.get_tempo_changes()[1]) == pytest.approx([95.0])


def test_midi_file_tools_keep_the_tempo_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(midi_algorithms, "DATA_DIR", str(tmp_path))
    source = tmp_path / "song.mid"
    _song_midi().write(str(source))
    original = pretty_midi.PrettyMIDI(str(source))
    starts = np.array([n.start for n in original.instruments[0].notes])

    even = midi_processors.load_pretty_midi(str(source))
    assert len(even.get_tempo_changes()[1]) == 1
    spacing = np.diff([n.start for n in even.instruments[0].notes])
    assert spacing == pytest.approx(spacing[0], abs=0.002)

    moved = pretty_midi.PrettyMIDI(
        midi_processors.process_transpose(str(source), 2)
    )
    assert TempoMap.from_pretty_midi(moved).bpms == pytest.approx(
        SONG.bpms, rel=1e-3
    )
    notes = moved.instruments[0].notes
    assert [n.pitch for n in notes] == [62] * len(starts)
    assert np.array([n.start for n in notes]) == pytest.approx(
        starts, abs=0.004
    )

    slow = pretty_midi.PrettyMIDI(
        midi_processors.process_speed(str(source), 2.0)
    )
    assert TempoMap.from_pretty_midi(slow).bpms == pytest.approx(
        (50.0, 55.0), rel=1e-3
    )
    assert np.array(
        [n.start for n in slow.instruments[0].notes]
    ) == pytest.approx(2 * starts, abs=0.008)

    plain = tmp_path / "plain.mid"
    one = pretty_midi.PrettyMIDI(initial_tempo=97.0)
    one.instruments.append(pretty_midi.Instrument(0))
    one.instruments[0].notes.append(pretty_midi.Note(90, 60, 1.0, 2.0))
    one.write(str(plain))
    kept = pretty_midi.PrettyMIDI(
        midi_processors.process_transpose(str(plain), 1)
    )
    assert list(kept.get_tempo_changes()[1]) == pytest.approx([97.0], abs=0.01)
    assert kept.instruments[0].notes[0].start == pytest.approx(1.0, abs=0.002)


@pytest.fixture
def arranger(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(timeline, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, text: shown.append((title, text)),
    )
    widget = ArrangerTimeline()
    widget.shown = shown
    return widget


def test_even_bpm_is_on_by_default_and_evens_out_a_loaded_song(
    arranger, clip_path
):
    assert arranger.btn_even_bpm.isChecked()
    assert "tempo" in arranger.btn_even_bpm.toolTip()
    arranger.add_audio_clip(clip_path, 0.0, name="Vocals", kind="vocals")
    arranger.load_midi_object(_song_midi())
    model = arranger.model
    assert model.normalized is not None
    assert model.tempo_map is None
    assert not arranger.piano_roll.vm.follows_tempo_map
    assert arranger.piano_roll.tempo == model.tempo
    clip = model.clips[0]
    assert os.path.dirname(clip.path).endswith(timeline.ALIGNED_CLIPS_DIR)
    assert _click_times(sf.read(clip.path)[0]) == pytest.approx(
        np.arange(1, END_BEAT - 1, 2) * 60 / model.tempo, abs=0.03
    )
    assert arranger.to_editor_time(12.0) == pytest.approx(
        SONG.warp(12.0, model.tempo)
    )
    assert arranger.shown == []

    arranger.btn_even_bpm.setChecked(False)
    assert model.normalized is None
    assert arranger.piano_roll.vm.follows_tempo_map
    assert model.clips[0].path == clip_path
    assert arranger.to_editor_time(12.0) == 12.0

    arranger.btn_even_bpm.setChecked(True)
    assert model.normalized is not None


def test_a_wild_clip_reports_what_to_do_and_removing_it_retries(
    arranger, clip_path
):
    arranger.add_audio_clip(clip_path, 0.0, name="Far off")
    arranger.load_midi_object(_song_midi(WILD))
    model = arranger.model
    assert model.normalized is None
    assert arranger.piano_roll.vm.follows_tempo_map
    assert len(arranger.shown) == 1
    title, text = arranger.shown[0]
    assert title == "Even BPM"
    assert "'Far off'" in text and "Turn off Even BPM" in text

    arranger.apply_even_bpm()
    assert len(arranger.shown) == 1

    model.remove_clip(model.clips[0].id)
    arranger.piano_roll.clips_changed.emit()
    assert model.clips == []
    assert model.normalized is not None
    assert len(arranger.shown) == 1
