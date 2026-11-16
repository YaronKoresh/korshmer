import numpy as np
import pytest
import soundfile as sf

from korshmer.audio import analyse_tempo as audio_analyse_tempo
from korshmer.audio_processing.core_analysis import analyse_tempo
from korshmer.audio_processing.stem_session import normalize_detected_tempo
from korshmer.audio_processing.tempo_estimation import estimate_tempo
from tests.fixtures.drum_loops import SAMPLE_RATE, make_drum_loop

ODD_TEMPOS = (123.4, 97.6, 142.3, 87.35, 100.0, 128.0)


@pytest.mark.parametrize("bpm", ODD_TEMPOS)
def test_odd_tempos_are_found_within_a_fraction_of_a_bpm(bpm):
    audio, _ = make_drum_loop(bpm, bars=16)
    assert estimate_tempo(audio, SAMPLE_RATE) == pytest.approx(bpm, abs=0.3)


def test_result_is_a_float_with_at_most_two_decimals():
    audio, _ = make_drum_loop(123.4, bars=12)
    tempo = estimate_tempo(audio, SAMPLE_RATE)
    assert isinstance(tempo, float)
    assert tempo == round(tempo, 2)
    assert tempo != round(tempo)


@pytest.mark.parametrize("bpm", (123.4, 142.3))
def test_analyse_tempo_keeps_the_fraction(tmp_path, bpm):
    audio, _ = make_drum_loop(bpm, bars=16)
    path = tmp_path / "loop.wav"
    sf.write(path, audio, SAMPLE_RATE)
    for function in (analyse_tempo, audio_analyse_tempo):
        assert function(str(path)) == pytest.approx(bpm, abs=0.3)
    assert normalize_detected_tempo(analyse_tempo(str(path))) == pytest.approx(
        bpm, abs=0.3
    )


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(0, dtype=np.float32),
        np.full(2048, np.nan, dtype=np.float32),
        np.zeros(SAMPLE_RATE * 3, dtype=np.float32),
    ],
)
def test_audio_without_a_beat_gives_none_or_a_valid_tempo(audio):
    tempo = estimate_tempo(audio, SAMPLE_RATE)
    assert tempo is None or 20.0 <= tempo <= 300.0


def test_missing_file_still_gives_none():
    assert analyse_tempo("nonexistent_file.wav") is None


@pytest.mark.parametrize("sr", [0, -1, float("nan"), float("inf"), "x", None])
def test_estimate_tempo_rejects_invalid_sample_rate(sr):
    audio, _ = make_drum_loop(120.0, bars=4)
    with pytest.raises(ValueError):
        estimate_tempo(audio, sr)


def test_a_slower_intro_does_not_blur_the_tempo_of_the_song():
    intro, _ = make_drum_loop(109.1, bars=20)
    body, _ = make_drum_loop(118.0, bars=40)
    audio = np.concatenate([intro, body])
    assert estimate_tempo(audio, SAMPLE_RATE) == pytest.approx(118.0, abs=0.3)


def test_local_tempos_follow_a_tempo_change():
    import librosa

    from korshmer.audio_processing.tempo_estimation import (
        HOP_LENGTH,
        local_tempos,
    )

    intro, _ = make_drum_loop(109.1, bars=12)
    body, _ = make_drum_loop(118.0, bars=24)
    envelope = librosa.onset.onset_strength(
        y=np.concatenate([intro, body]).astype(np.float32),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    found = local_tempos(envelope, SAMPLE_RATE, 118.0)
    assert found[0][1] == pytest.approx(109.1, abs=0.5)
    assert found[-1][1] == pytest.approx(118.0, abs=0.5)
