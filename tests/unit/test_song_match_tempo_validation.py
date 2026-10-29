import numpy as np
import pretty_midi
import pytest

from korshmer.constants import SCALES
from korshmer.midi_generation import song_match
from korshmer.utils.tempo import MAX_BPM, MIN_BPM

NAN = float("nan")
BAD_TEMPOS = [
    0,
    -120,
    NAN,
    float("inf"),
    "120",
    None,
    True,
    1e9,
    MIN_BPM - 1,
    MAX_BPM + 1,
]


def one_note_song():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(90, 60, 0.0, 3.0))
    midi.instruments.append(inst)
    return midi


@pytest.mark.parametrize("bpm", BAD_TEMPOS)
def test_bar_count_rejects_a_tempo_outside_the_supported_range(bpm):
    with pytest.raises(ValueError, match="bpm"):
        song_match.bar_count(one_note_song(), bpm)


@pytest.mark.parametrize("bpm", BAD_TEMPOS)
def test_estimate_offset_rejects_a_tempo_outside_the_supported_range(bpm):
    with pytest.raises(ValueError, match="bpm"):
        song_match.estimate_offset(one_note_song(), bpm)


@pytest.mark.parametrize("bpm", BAD_TEMPOS)
def test_analysis_and_triads_reject_a_tempo_outside_the_supported_range(bpm):
    song = one_note_song()
    with pytest.raises(ValueError, match="bpm"):
        song_match.analyze_song(song, bpm, 0, SCALES["Major"])
    with pytest.raises(ValueError, match="bpm"):
        song_match.bar_triads(song, bpm)


@pytest.mark.parametrize("bpm", [MIN_BPM, 96.5, 120, MAX_BPM, np.float64(90)])
def test_tempos_in_range_keep_working(bpm):
    song = one_note_song()
    assert song_match.bar_count(song, bpm) % song_match.BLOCK_BARS == 0
    assert song_match.estimate_offset(song, bpm) >= 0.0
    assert len(song_match.bar_triads(song, bpm)) > 0


def test_an_empty_song_needs_no_tempo_check_to_count_zero_bars():
    assert song_match.bar_count(None, 120) == 0
