import sys

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication, QComboBox

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer import midi_algorithms
from korshmer.analysis import (
    detect_key_and_mode,
    detect_midi_key,
    export_scale_stats,
    rank_scales,
)
from korshmer.constants import NOTE_NAMES, SCALES
from korshmer.midi.transcription_cleanup import (
    CleanupOptions,
    clean_transcription,
    estimate_key,
)
from korshmer.midi.variation import _scale_pitch_classes
from korshmer.midi_processors import process_scale_detection
from korshmer.mixins.actions.editor import EditorActionsMixin

BAR = 2.0
HOME_CHORD_LAST = (5, 2, 6, 0, 5, 2, 6, 0)
HOME_CHORD_LAST_TUNE = ((5, 4, 2, 0), (2, 4, 6, 4), (6, 4, 1, 2), (2, 1, 0, 0))
AWAY_THEN_HOME = (2, 6, 0, 0, 2, 6, 0, 0)
AWAY_THEN_HOME_TUNE = (
    (2, 4, 6, 4),
    (6, 4, 1, 2),
    (0, 2, 4, 2),
    (4, 2, 0, 0),
)
MAJOR_LOOP = (0, 4, 5, 3, 0, 4, 3, 0)
MAJOR_LOOP_TUNE = (
    (0, 2, 4, 2),
    (4, 6, 1, 6),
    (5, 0, 2, 0),
    (3, 5, 0, 5),
    (0, 2, 4, 2),
    (4, 6, 1, 6),
    (3, 5, 0, 5),
    (2, 1, 0, 0),
)
RAISED_SEVENTH = (0, 3, 4, 0, 0, 5, 4, 0)
RAISED_SEVENTH_TUNE = (
    (0, 2, 4, 2),
    (3, 5, 0, 5),
    (4, 6, 1, 6),
    (0, 2, 4, 2),
    (0, 2, 4, 2),
    (5, 0, 2, 0),
    (4, 6, 1, 6),
    (2, 1, 0, 0),
)
DORIAN_VAMP = (3, 0, 3, 0, 3, 0, 3, 0)
DORIAN_VAMP_TUNE = ((3, 5, 0, 5), (0, 2, 4, 5), (3, 5, 0, 5), (4, 2, 1, 0))


@pytest.fixture(autouse=True)
def processed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(midi_algorithms, "DATA_DIR", str(tmp_path / "data"))


def _song(tonic, scale, bars, tune, band=True):
    steps = SCALES[scale]

    def pitch(degree, base):
        octave, step = divmod(degree, len(steps))
        return base + tonic + steps[step] + 12 * octave

    lead = pretty_midi.Instrument(0)
    bass = pretty_midi.Instrument(33)
    pad = pretty_midi.Instrument(48)
    for index, chord in enumerate(bars):
        start = index * BAR
        if band:
            bass.notes.append(
                pretty_midi.Note(95, pitch(chord, 36), start, start + BAR)
            )
            for part in (0, 2, 4):
                pad.notes.append(
                    pretty_midi.Note(
                        70, pitch(chord + part, 48), start, start + BAR
                    )
                )
        line = tune[index % len(tune)]
        length = BAR / len(line)
        for beat, degree in enumerate(line):
            lead.notes.append(
                pretty_midi.Note(
                    100,
                    pitch(degree, 60),
                    start + beat * length,
                    start + (beat + 1) * length,
                )
            )
    song = pretty_midi.PrettyMIDI(initial_tempo=120)
    song.instruments += [i for i in (lead, bass, pad) if i.notes]
    return song


def _minor_loop(tonic=9, band=True):
    return _song(tonic, "Minor", HOME_CHORD_LAST, HOME_CHORD_LAST_TUNE, band)


def _saved(song, tmp_path, name="song.mid"):
    path = tmp_path / name
    song.write(str(path))
    return str(path)


def _all_notes(song):
    notes = [n for inst in song.instruments for n in inst.notes]
    return sorted(notes, key=lambda n: (n.start, n.pitch))


@pytest.mark.parametrize("tonic", range(12))
def test_a_minor_loop_that_comes_home_to_its_minor_chord_is_named_minor(
    tonic, tmp_path
):
    path = _saved(_minor_loop(tonic), tmp_path)
    assert detect_midi_key(path) == (NOTE_NAMES[tonic], "Minor")


@pytest.mark.parametrize("tonic", range(12))
def test_the_same_notes_centred_on_the_major_chord_are_still_named_major(
    tonic, tmp_path
):
    song = _song(tonic, "Major", MAJOR_LOOP, MAJOR_LOOP_TUNE)
    assert detect_midi_key(_saved(song, tmp_path)) == (
        NOTE_NAMES[tonic],
        "Major",
    )


def test_minor_and_its_relative_major_use_the_same_notes_but_differ_in_home(
    tmp_path,
):
    minor = _minor_loop(9)
    major = _song(0, "Major", MAJOR_LOOP, MAJOR_LOOP_TUNE)
    assert {n.pitch % 12 for n in _all_notes(minor)} == {
        n.pitch % 12 for n in _all_notes(major)
    }
    assert detect_midi_key(_saved(minor, tmp_path, "minor.mid")) == (
        "A",
        "Minor",
    )
    assert detect_midi_key(_saved(major, tmp_path, "major.mid")) == (
        "C",
        "Major",
    )


@pytest.mark.parametrize("tonic", [9, 2, 6])
def test_note_cleanup_reports_the_minor_key_of_a_minor_song(tonic):
    song = _minor_loop(tonic)
    assert estimate_key(_all_notes(song)) == (tonic, "Minor")
    _, report = clean_transcription(
        song, options=CleanupOptions(snap_to_key=True)
    )
    assert report["key"] == (tonic, "Minor")


class _TheoryTab(EditorActionsMixin):
    def __init__(self):
        self.combo_root = QComboBox()
        self.combo_root.addItems(["Auto", *NOTE_NAMES])
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Auto (Detect)", *SCALES])


def test_generator_on_auto_key_takes_the_minor_home_of_the_song_in_the_editor():
    assert _TheoryTab()._resolve_key(_minor_loop(9)) == ("A", "Minor")


def test_generator_on_auto_root_only_takes_the_minor_home_note():
    tab = _TheoryTab()
    tab.combo_scale.setCurrentText("Minor")
    assert tab._resolve_key(_minor_loop(4)) == ("E", "Minor")


def test_scale_statistics_report_names_the_minor_key(tmp_path):
    out = export_scale_stats(_minor_loop(9), str(tmp_path / "stats.txt"))
    text = open(out, encoding="utf-8").read()
    assert "Top Estimation: A Minor" in text


def test_scale_detection_tool_lists_the_minor_home_first(tmp_path):
    song = _song(9, "Minor", AWAY_THEN_HOME, AWAY_THEN_HOME_TUNE)
    report = open(process_scale_detection(_saved(song, tmp_path))).read()
    first = report.split("\n\n", 1)[1].splitlines()[0]
    assert first.startswith("A ")
    assert "Minor" in first


def test_minor_song_with_a_raised_seventh_keeps_its_minor_home(tmp_path):
    song = _song(9, "Harmonic Minor", RAISED_SEVENTH, RAISED_SEVENTH_TUNE)
    assert detect_midi_key(_saved(song, tmp_path)) == ("A", "Harmonic Minor")
    assert estimate_key(_all_notes(song)) == (9, "Minor")


def test_dorian_song_that_opens_away_from_home_is_centred_on_its_own_home(
    tmp_path,
):
    song = _song(2, "Dorian", DORIAN_VAMP, DORIAN_VAMP_TUNE)
    assert detect_midi_key(_saved(song, tmp_path)) == ("D", "Dorian")


def test_variation_tool_keeps_to_the_notes_of_the_minor_key():
    assert _scale_pitch_classes(_minor_loop(9)) == [0, 2, 4, 5, 7, 9, 11]


def test_notes_with_no_centre_at_all_are_named_minor_rather_than_major():
    plain = np.zeros(12)
    plain[SCALES["Major"]] = 1.0
    assert rank_scales(plain, 1)[0][:2] == (9, "Minor")


def test_a_clear_major_profile_is_not_pulled_to_its_relative_minor():
    clear = np.zeros(12)
    clear[SCALES["Major"]] = 2.5
    clear[[0, 4, 7]] = (6.0, 3.2, 4.0)
    assert rank_scales(clear, 1)[0][:2] == (0, "Major")


def _render(song, rate=22050):
    audio = np.zeros(int((song.get_end_time() + 0.5) * rate))
    for note in _all_notes(song):
        tone = 440.0 * 2 ** ((note.pitch - 69) / 12)
        clock = np.arange(int((note.end - note.start) * rate)) / rate
        wave = sum(
            np.sin(2 * np.pi * tone * part * clock) / part
            for part in (1, 2, 3, 4)
        )
        wave = wave * np.exp(-1.5 * clock) * note.velocity / 127
        wave[-200:] *= np.linspace(1.0, 0.0, 200)
        begin = int(note.start * rate)
        audio[begin : begin + wave.size] += wave
    return (audio / np.abs(audio).max() * 0.5).astype(np.float32), rate


@pytest.mark.parametrize("tonic", [9, 4])
def test_recording_of_a_minor_loop_is_detected_as_minor(tonic, tmp_path):
    audio, rate = _render(_minor_loop(tonic))
    path = tmp_path / f"minor_{tonic}.wav"
    sf.write(str(path), audio, rate)
    assert detect_key_and_mode(str(path)) == (tonic, "Minor")
