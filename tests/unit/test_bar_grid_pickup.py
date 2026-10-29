import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing.bar_grid import BarGrid, find_bar_grid
from korshmer.audio_processing.beat_detection import BeatDetector
from tests.fixtures.bar_songs import SAMPLE_RATE, make_bar_song

TOLERANCE = 0.03


def _assert_bars(grid, truth):
    expected = truth["bar_lines"]
    assert len(grid.downbeat_times) == len(expected)
    assert np.allclose(grid.downbeat_times, expected, atol=TOLERANCE)


@pytest.mark.parametrize(
    ("pickup", "tail"), [(0, 0), (1, 2), (3, 1), (2, 3), (1, 0), (0, 2)]
)
def test_bar_lines_follow_the_real_downbeat(pickup, tail):
    audio, truth = make_bar_song(pickup_beats=pickup, tail_beats=tail)

    grid = find_bar_grid(audio, SAMPLE_RATE, 120.0)

    _assert_bars(grid, truth)
    assert grid.pickup_beats == pickup
    assert grid.last_bar_beats == truth["last_bar_beats"]
    assert grid.ends_mid_bar is (tail != 0)


def test_leading_silence_is_not_a_pickup():
    audio, truth = make_bar_song(pickup_beats=1, tail_beats=2, lead_seconds=0.8)

    grid = find_bar_grid(audio, SAMPLE_RATE, 120.0)

    _assert_bars(grid, truth)
    assert grid.pickup_beats == 1
    assert grid.content_start == pytest.approx(0.8, abs=0.01)
    assert grid.pickup_seconds == pytest.approx(0.5, abs=TOLERANCE)


def test_three_four_song_with_a_pickup():
    audio, truth = make_bar_song(
        bpm=90.0, beats_per_bar=3, pickup_beats=1, tail_beats=1
    )

    grid = find_bar_grid(audio, SAMPLE_RATE, 90.0, beats_per_bar=3)

    _assert_bars(grid, truth)
    assert grid.pickup_beats == 1
    assert grid.last_bar_beats == 1


def test_detect_downbeats_on_a_file_with_a_pickup(tmp_path):
    audio, truth = make_bar_song(pickup_beats=1, tail_beats=2)
    path = tmp_path / "pickup.wav"
    sf.write(path, audio, SAMPLE_RATE)

    downbeats = BeatDetector().detect_downbeats(str(path), 120.0)

    assert np.allclose(downbeats, truth["bar_lines"], atol=TOLERANCE)
    assert downbeats[0] == pytest.approx(0.5, abs=TOLERANCE)


def test_detect_bar_grid_reports_pickup_and_partial_last_bar(tmp_path):
    audio, _ = make_bar_song(pickup_beats=3, tail_beats=2)
    path = tmp_path / "song.wav"
    sf.write(path, audio, SAMPLE_RATE)

    grid = BeatDetector().detect_bar_grid(str(path), 120.0)

    assert isinstance(grid, BarGrid)
    assert grid.pickup_beats == 3
    assert grid.pickup_seconds == pytest.approx(1.5, abs=TOLERANCE)
    assert grid.last_bar_beats == 2


def test_silent_audio_has_no_bar_lines():
    grid = find_bar_grid(
        np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, 120
    )

    assert len(grid.beat_times) == 0
    assert grid.pickup_beats == 0


@pytest.mark.parametrize("bad", [1, 13, 0, -4, 2.5, True])
def test_beats_per_bar_must_be_a_usable_integer(bad):
    audio, _ = make_bar_song()
    with pytest.raises(ValueError):
        find_bar_grid(audio, SAMPLE_RATE, 120.0, beats_per_bar=bad)


def test_chunked_envelope_matches_the_whole_song_envelope(monkeypatch):
    import librosa

    from korshmer.audio_processing import bar_grid

    audio, _ = make_bar_song(pickup_beats=1, full_bars=10, tail_beats=2)
    whole = librosa.onset.onset_strength(
        y=audio, sr=SAMPLE_RATE, hop_length=bar_grid.HOP_LENGTH
    )
    monkeypatch.setattr(bar_grid, "ENVELOPE_CHUNK_SECONDS", 4.0)
    monkeypatch.setattr(bar_grid, "ENVELOPE_PAD_SECONDS", 0.5)

    chunked = bar_grid._onset_envelope(audio, SAMPLE_RATE)

    assert len(chunked) == len(whole)
    assert np.allclose(chunked, whole, atol=0.05 * whole.max())


def test_long_song_is_found_with_small_chunks(monkeypatch):
    from korshmer.audio_processing import bar_grid

    monkeypatch.setattr(bar_grid, "ENVELOPE_CHUNK_SECONDS", 5.0)
    audio, truth = make_bar_song(pickup_beats=1, full_bars=12, tail_beats=2)

    grid = find_bar_grid(audio, SAMPLE_RATE, 120.0)

    assert np.allclose(grid.downbeat_times, truth["bar_lines"], atol=TOLERANCE)
