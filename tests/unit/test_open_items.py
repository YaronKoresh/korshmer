import logging
import sys
from unittest import mock

import numpy as np
import pretty_midi
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

if not QApplication.instance():
    app = QApplication(sys.argv)

import korshmer.__main__ as app_main
from korshmer.audio_processing import vocal_melody
from korshmer.midi import transcription_cleanup as cleanup
from korshmer.midi_generation import composer
from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
)
from korshmer.mixins.files import FilesMixin
from korshmer.project_management.project_manager import ProjectManager
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline


class _Studio(ProjectManagementActionsMixin):
    _recovery_song_path = app_main.MainWindow._recovery_song_path
    _save_recovery_state = app_main.MainWindow._save_recovery_state
    _check_auto_recovery = app_main.MainWindow._check_auto_recovery

    def __init__(self, base):
        self.project_manager = ProjectManager(base)
        self.arranger = ArrangerTimeline()
        self.current_session = None


def _two_notes(studio):
    track = studio.arranger.model.add_track("Lead")
    studio.arranger.model.add_note(track.id, 60, 0.0, 0.5, 90)
    studio.arranger.model.add_note(track.id, 64, 0.5, 0.5, 70)
    studio.arranger.piano_roll.load_model()


def test_an_unsaved_song_is_recovered_with_its_notes(tmp_path):
    studio = _Studio(tmp_path)
    _two_notes(studio)
    studio._save_recovery_state()
    later = _Studio(tmp_path)
    with (
        mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Yes
        ),
        mock.patch.object(QMessageBox, "information"),
    ):
        later._check_auto_recovery()
    pitches = sorted(n.pitch for n in later.arranger.model.notes)
    assert pitches == [60, 64]
    assert later.project_manager.enable_auto_recovery() is None


def test_an_empty_unsaved_editor_leaves_no_recovery_file(tmp_path):
    studio = _Studio(tmp_path)
    studio._save_recovery_state()
    assert studio.project_manager.enable_auto_recovery() is None


def test_a_failed_file_release_is_logged(caplog):
    class _Broken(FilesMixin):
        @property
        def control_bar(self):
            raise RuntimeError("player is gone")

    with caplog.at_level(logging.WARNING):
        _Broken().release_file_lock()
    assert "player is gone" in caplog.text


def _midi(notes):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(80, p, a, b) for p, a, b in notes]
    midi.instruments.append(inst)
    return midi


def _level(sounding, seconds=4.0, fps=50.0):
    level = np.full((cleanup.PITCH_FLUX_BINS, int(seconds * fps)), -70.0)
    for pitch, start, end in sounding:
        row = pitch - cleanup.PITCH_FLUX_LOWEST
        level[row, int(start * fps) : int(end * fps)] = -6.0
    return level, fps


def test_a_second_pass_note_is_adopted_only_when_the_audio_confirms_it():
    first = _midi([(60, 0.0, 1.0)])
    second = _midi([(60, 0.05, 1.0), (64, 1.5, 2.0), (67, 2.5, 3.0)])
    level, fps = _level([(60, 0.0, 1.0), (64, 1.5, 2.0), (72, 2.5, 3.0)])
    added = cleanup.adopt_heard_notes(first, [second], level, fps)
    assert added == 1
    assert [n.pitch for n in first.instruments[0].notes] == [60, 64]


def test_adopting_notes_never_touches_drums_or_empty_songs():
    drums = pretty_midi.PrettyMIDI()
    drums.instruments.append(pretty_midi.Instrument(0, is_drum=True))
    level, fps = _level([(64, 1.5, 2.0)])
    assert (
        cleanup.adopt_heard_notes(drums, [_midi([(64, 1.5, 2.0)])], level, fps)
        == 0
    )
    assert drums.instruments[0].notes == []


def test_a_sung_note_rings_until_the_next_when_the_gap_is_tiny():
    note = vocal_melody.VocalNote
    sung = [
        note(0.0, 0.2, 60, 80),
        note(0.25, 0.5, 62, 80),
        note(1.0, 1.2, 64, 80),
    ]
    joined = vocal_melody._sing_through(sung, 0.12)
    assert [(n.start, n.end) for n in joined] == [
        (0.0, 0.25),
        (0.25, 0.5),
        (1.0, 1.2),
    ]
    assert vocal_melody._sing_through(sung, 0.0)[0].end == 0.2
    with pytest.raises(ValueError, match="Legato gap"):
        vocal_melody.VocalMelodyOptions(legato_gap_s=-1)


def test_a_short_scoop_or_fall_joins_its_longer_neighbour():
    hop = 256 / 22050
    spans = [(0, 6, 59), (6, 60, 60), (60, 100, 64), (100, 104, 65)]
    assert vocal_melody._absorb_scoops(spans, hop) == [
        (0, 60, 60),
        (60, 104, 64),
    ]
    step = [(0, 40, 59), (40, 80, 60)]
    assert vocal_melody._absorb_scoops(step, hop) == step


BPM = 120.0
BAR = 240.0 / BPM


def _quiet_middle_song():
    song = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    keys = pretty_midi.Instrument(0)
    for index in range(48):
        busy = not 16 <= index < 32
        for step in range(8 if busy else 1):
            start = index * BAR + step * BAR / 8
            for pitch in (57, 60, 64):
                keys.notes.append(
                    pretty_midi.Note(
                        90 if busy else 60,
                        pitch,
                        start,
                        start + BAR / 8 * (1 if busy else 8),
                    )
                )
    song.instruments.append(keys)
    return song


def _layer(seed=3):
    options = composer.ComposeOptions(
        "A",
        "Minor",
        bpm=BPM,
        style="Pop",
        seed=seed,
        mode="Layer over editor",
        roles=("drums", "bass", "chords", "pad"),
    )
    return composer.compose(options, existing=_quiet_middle_song())


def test_bass_and_drums_keep_playing_through_a_quiet_middle_section():
    made = _layer()
    breaks = [s for s in made.sections if s.name == "Break"]
    assert breaks
    for section in breaks:
        assert {"drums", "bass"} <= set(section.roles)
        lo = section.start_bar * BAR + made.grid_offset
        hi = lo + section.bars * BAR
        for inst in made.instruments:
            if inst.is_drum or "Bass" in (inst.name or ""):
                assert any(lo <= n.start < hi for n in inst.notes)
    quiet = [s for s in made.sections if s.name in ("Intro", "Outro")]
    assert all("drums" not in s.roles for s in quiet)


def test_layers_draw_from_wider_rhythm_pools():
    pool = composer._widened(("808", "driving"), composer.LAYER_BASS_RHYTHMS)
    assert pool[:2] == ("808", "driving") and len(set(pool)) == len(pool) == 5
    hits = composer._widened(([0.0], [0.0, 0.5]), composer.LAYER_CHORD_HITS)
    assert len(hits) == 5
    rhythms = set()
    made = _layer()
    for inst in made.instruments:
        if "Bass" not in (inst.name or ""):
            continue
        for index in range(made.total_bars):
            lo = index * BAR + made.grid_offset
            steps = tuple(
                sorted(
                    {
                        round((n.start - lo) / BAR * 16)
                        for n in inst.notes
                        if lo <= n.start < lo + BAR
                    }
                )
            )
            if steps:
                rhythms.add(steps)
    assert len(rhythms) >= 5


def test_a_pad_is_struck_again_in_the_last_bar_of_a_varied_phrase():
    pad = pretty_midi.Instrument(89)
    pad.notes = [pretty_midi.Note(70, p, 6.0, 8.0) for p in (57, 60, 64)]
    composer.vary_phrase(pad, "pad", 2, 8.0, 2.0)
    starts = sorted({n.start for n in pad.notes})
    assert starts == [6.0, 7.0]
    assert len(pad.notes) == 6
