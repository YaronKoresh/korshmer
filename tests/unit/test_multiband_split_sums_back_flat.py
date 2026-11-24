import numpy as np
import pytest

from korshmer.audio_processing.dynamics import MultiBandCompressor

RATE = 44100
TONES_NEAR_THE_CROSSOVERS = [
    (2, 40),
    (2, 100),
    (2, 160),
    (2, 300),
    (3, 100),
    (3, 300),
    (3, 500),
    (3, 1000),
    (4, 100),
    (4, 500),
    (4, 2000),
    (4, 3500),
    (5, 100),
    (5, 500),
    (5, 2000),
    (5, 8000),
    (5, 12000),
]


def _level_change_db(processed, original):
    settled = slice(RATE // 2, None)
    after = np.sqrt(np.mean(processed[..., settled] ** 2))
    before = np.sqrt(np.mean(original[..., settled] ** 2))
    return 20 * np.log10(after / before)


def _tone(freq):
    t = np.arange(RATE) / RATE
    return 0.5 * np.sin(2 * np.pi * freq * t)


@pytest.mark.parametrize(("bands", "freq"), TONES_NEAR_THE_CROSSOVERS)
def test_with_no_compression_every_tone_comes_back_at_its_own_level(
    bands, freq
):
    tone = _tone(freq)

    out = MultiBandCompressor(RATE, n_bands=bands).process(tone, threshold_db=0)

    assert abs(_level_change_db(out, tone)) < 0.1


def test_with_no_compression_a_stereo_mix_keeps_its_spectrum():
    rng = np.random.default_rng(5)
    mix = rng.uniform(-0.5, 0.5, (2, RATE * 2))

    out = MultiBandCompressor(RATE, n_bands=5).process(mix, threshold_db=0)

    before = np.abs(np.fft.rfft(mix[0])) ** 2
    after = np.abs(np.fft.rfft(out[0])) ** 2
    bins = np.arange(len(before)) * RATE / mix.shape[1]
    changes = []
    for low, high in [(60, 140), (350, 700), (1500, 2600), (6000, 10000)]:
        band = (bins >= low) & (bins < high)
        changes.append(
            10 * np.log10(np.sum(after[band]) / np.sum(before[band]))
        )
    assert out.shape == mix.shape
    assert np.max(np.abs(changes)) < 0.1
