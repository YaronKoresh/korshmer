from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from korshmer.analysis import analyze_chords_full, create_chord_sheet
from tests.fixtures.bar_songs import SAMPLE_RATE, make_chord_song


@pytest.fixture(scope="module")
def song_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("chord_song") / "song.wav"
    sf.write(path, make_chord_song(), SAMPLE_RATE)
    return str(path)


def _sheet(song_path, tmp_path, mode="1 Bar"):
    with (
        patch("korshmer.analysis.analyse_tempo", return_value=120.0),
        patch("korshmer.analysis.estimate_meter", return_value=4),
    ):
        out = create_chord_sheet(song_path, str(tmp_path), mode)
    return open(out, encoding="utf-8").read().splitlines()


def test_sheet_puts_the_pickup_chord_before_bar_one(song_path, tmp_path):
    lines = _sheet(song_path, tmp_path)

    assert "Pickup: 1 beat before the first bar line" in lines
    assert lines.index("Pickup") < lines.index("Bar 1")
    pickup_at = lines.index("Pickup")
    assert lines[pickup_at + 2].endswith("GMaj")


def test_sheet_bars_hold_the_chord_played_in_that_bar(song_path, tmp_path):
    lines = _sheet(song_path, tmp_path)

    for bar, chord in enumerate(("CMaj", "FMaj", "GMaj", "CMaj"), start=1):
        at = lines.index(f"Bar {bar}")
        assert lines[at + 2].endswith(chord), (bar, lines[at + 2])


def test_sheet_marks_the_partial_last_bar(song_path, tmp_path):
    lines = _sheet(song_path, tmp_path)

    assert "Bar 5 (2 of 4 beats)" in lines
    at = lines.index("Bar 5 (2 of 4 beats)")
    assert lines[at + 2].endswith("FMaj")


def test_sheet_without_a_pickup_has_no_pickup_section(tmp_path):
    from tests.fixtures.bar_songs import make_bar_song

    audio, _ = make_bar_song(full_bars=4)
    path = tmp_path / "plain.wav"
    sf.write(path, audio, SAMPLE_RATE)

    lines = _sheet(str(path), tmp_path)

    assert "Pickup" not in lines
    assert not any(line.startswith("Pickup:") for line in lines)
    assert "Bar 1" in lines


def test_windows_start_on_the_origin_without_drifting(song_path):
    chords = analyze_chords_full(song_path, interval=2.0, origin=0.5)

    times = [t for t, _ in chords]
    assert times[0] == 0.0
    assert times[1:] == [0.5 + 2.0 * k for k in range(len(times) - 1)]


def test_long_interval_does_not_drift_from_the_requested_time(song_path):
    chords = analyze_chords_full(song_path, interval=0.9)

    for k, (t, _) in enumerate(chords):
        assert t == pytest.approx(k * 0.9)


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, None])
def test_origin_and_start_are_validated(song_path, bad):
    with pytest.raises(ValueError):
        analyze_chords_full(song_path, 1.0, origin=bad)
    with pytest.raises(ValueError):
        analyze_chords_full(song_path, 1.0, start=bad)


def test_pickup_window_holds_only_the_pickup(song_path):
    chords = analyze_chords_full(song_path, interval=2.0, origin=0.5)

    assert chords[0][1] == "GMaj"
    assert np.isclose(chords[1][0], 0.5)
