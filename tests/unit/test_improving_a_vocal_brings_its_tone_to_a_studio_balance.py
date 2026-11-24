import numpy as np
import pytest

from korshmer.audio_processing import clip_level

RATE = 44100
BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
NATURAL = {
    125: -6.0,
    250: 0.0,
    500: 1.0,
    1000: 0.0,
    2000: -5.0,
    4000: -11.0,
    8000: -17.0,
}
DARK_AND_MUDDY = {
    125: 8.0,
    250: 8.0,
    500: 8.0,
    1000: 0.0,
    2000: -12.0,
    4000: -23.0,
    8000: -30.0,
}


def _interpolated(curve, freq):
    centres = np.log2(np.array(sorted(curve)))
    values = np.array([curve[c] for c in sorted(curve)])
    return np.interp(np.log2(max(freq, 1.0)), centres, values)


def _phrase(seconds, shape, f0):
    time = np.arange(int(seconds * RATE)) / RATE
    pitch = f0 * (1.0 + 0.01 * np.sin(2 * np.pi * 5.5 * time))
    phase = 2 * np.pi * np.cumsum(pitch) / RATE
    voice = np.zeros_like(time)
    for harmonic in range(1, int(12000.0 / f0)):
        gain_db = _interpolated(shape, harmonic * f0)
        voice += 10 ** (gain_db / 20.0) * np.sin(harmonic * phase)
    fade = np.minimum(1.0, np.minimum(time, time[-1] - time) / 0.05)
    return voice * fade


def _singing(shape, level_db=-20.0):
    rng = np.random.default_rng(3)
    parts = []
    for f0 in (130.0, 147.0, 165.0, 196.0, 147.0, 130.0):
        parts.append(_phrase(1.5, shape, f0))
        parts.append(np.zeros(int(0.5 * RATE)))
    voice = np.concatenate(parts)
    sung = np.abs(voice) > 0
    voice *= 10 ** (level_db / 20.0) / np.sqrt(np.mean(voice[sung] ** 2))
    voice += rng.normal(0.0, 10 ** (-75.0 / 20.0), voice.shape[0])
    return voice.astype(np.float32)


def _balance(audio):
    power = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / RATE)

    def octave(centre):
        chosen = (freqs >= centre / np.sqrt(2)) & (freqs < centre * np.sqrt(2))
        return 10 * np.log10(np.mean(power[chosen]) + 1e-30)

    body = octave(1000)
    return {centre: octave(centre) - body for centre in BANDS}


def test_a_dark_muddy_vocal_is_brought_to_a_studio_tone_balance():
    recording = _singing(DARK_AND_MUDDY)

    improved, readings = clip_level.improve(recording, RATE)

    after = _balance(improved)
    for centre in (250, 500, 2000, 4000, 8000):
        assert after[centre] == pytest.approx(NATURAL[centre], abs=3.5)
    assert "tone" in clip_level.summary(readings)
    assert "not reached" not in clip_level.summary(readings)
