import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing import clip_level, clip_polish
from korshmer.audio_processing.clip_stems import MUSIC, VOCALS
from korshmer.audio_processing.music_cleanup import clean_music

RATE = 44100


def _tones(seconds, freqs, level=0.1):
    time = np.arange(int(seconds * RATE)) / RATE
    return level * sum(np.sin(2 * np.pi * f * time) for f in freqs)


def _hiss(samples, level_db, low_hz, high_hz, seed=2):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.normal(0, 1, samples))
    freqs = np.fft.rfftfreq(samples, 1 / RATE)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    noise = np.fft.irfft(spectrum, samples)
    return noise * 10 ** (level_db / 20.0) / np.sqrt(np.mean(noise**2))


def _band_db(audio, low_hz, high_hz):
    power = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / RATE)
    chosen = (freqs >= low_hz) & (freqs < high_hz)
    return 10 * np.log10(np.mean(power[chosen]) + 1e-30)


def _rms_db(audio):
    return 10 * np.log10(np.mean(audio**2) + 1e-30)


def _middle(audio):
    return audio[RATE // 2 : -RATE // 2]


def test_the_bass_of_a_music_clip_is_kept():
    music = _tones(4.0, (55.0, 220.0, 660.0))

    cleaned = clean_music(music, RATE)

    assert _band_db(_middle(cleaned), 50, 60) == pytest.approx(
        _band_db(_middle(music), 50, 60), abs=1.0
    )


def test_a_music_clip_keeps_its_own_tone():
    dark = _tones(4.0, (110.0, 220.0, 440.0, 880.0))
    bright = dark + _tones(4.0, (6000.0, 9000.0), level=0.05)

    for music, bands in (
        (dark, ((100, 1000),)),
        (bright, ((100, 1000), (5000, 10000))),
    ):
        cleaned = clean_music(music, RATE)
        for low, high in bands:
            assert _band_db(_middle(cleaned), low, high) == pytest.approx(
                _band_db(_middle(music), low, high), abs=1.0
            )


def test_steady_hiss_under_playing_music_is_turned_down():
    music = _tones(4.0, (110.0, 330.0, 990.0, 2970.0))
    hiss = _hiss(music.shape[0], -50.0, 8000.0, 16000.0)

    cleaned = clean_music(music + hiss, RATE)

    removed = _band_db(_middle(music + hiss), 8000, 16000) - _band_db(
        _middle(cleaned), 8000, 16000
    )
    assert removed >= 6.0


def test_a_stretch_of_only_noise_between_music_is_lowered():
    music = _tones(2.0, (220.0, 440.0))
    gap = np.zeros(RATE * 2)
    clip = np.concatenate([music, gap, music])
    clip = clip + _hiss(clip.shape[0], -55.0, 50.0, 16000.0)

    cleaned = clean_music(clip, RATE)

    inside = slice(int(2.5 * RATE), int(3.5 * RATE))
    assert _rms_db(clip[inside]) - _rms_db(cleaned[inside]) >= 6.0


def test_a_noisy_sound_that_is_part_of_the_music_is_kept():
    music = _tones(4.0, (110.0, 220.0))
    shaker = _hiss(int(0.3 * RATE), -30.0, 400.0, 3500.0, seed=7)
    start = 2 * RATE
    music[start : start + shaker.shape[0]] += shaker

    cleaned = clean_music(music, RATE)

    span = slice(start + 2048, start + shaker.shape[0] - 2048)
    assert _band_db(cleaned[span], 500, 3000) == pytest.approx(
        _band_db(music[span], 500, 3000), abs=1.0
    )


def test_the_level_of_a_music_clip_is_kept():
    music = _tones(4.0, (110.0, 220.0, 440.0))

    cleaned = clean_music(music, RATE)

    assert _rms_db(_middle(cleaned)) == pytest.approx(
        _rms_db(_middle(music)), abs=1.0
    )


def test_removing_noise_from_a_music_clip_keeps_its_bass(tmp_path):
    path = tmp_path / "bass.wav"
    sf.write(path, _tones(3.0, (45.0, 330.0)).astype(np.float32), RATE)

    as_music = clip_polish.remove_noise_file(str(path), tmp_path, kind=MUSIC)
    as_vocals = clip_polish.remove_noise_file(str(path), tmp_path, kind=VOCALS)

    kept = _band_db(_middle(sf.read(as_music)[0]), 40, 50)
    cut = _band_db(_middle(sf.read(as_vocals)[0]), 40, 50)
    assert kept - cut >= 10.0


def test_a_music_clip_is_set_to_the_level_of_the_instruments_not_above_them():
    music = _tones(4.0, (220.0, 440.0), level=0.02)
    notes = _tones(4.0, (220.0, 440.0), level=0.1)

    improved, readings = clip_level.improve(music, RATE, notes, kind=MUSIC)

    names = {reading.name for reading in readings}
    assert clip_level.OVER_NOTES not in names
    assert clip_level.PHRASES not in names
    level = _rms_db(_middle(improved))
    assert level == pytest.approx(clip_level.music_level_goal_db(), abs=1.5)
