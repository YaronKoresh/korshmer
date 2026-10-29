import numpy as np
import pytest
import soundfile as sf

from korshmer.analysis import analyze_chords_full

SR = 22050


@pytest.fixture(scope="module")
def tone_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("chords") / "tone.wav"
    t = np.arange(SR * 4) / SR
    sf.write(path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)
    return str(path)


def test_normal_interval_returns_chords(tone_path):
    chords = analyze_chords_full(tone_path, 1.0)

    assert len(chords) == 5
    assert chords[0][0] == 0.0


def test_interval_below_one_frame_uses_one_frame(tone_path):
    chords = analyze_chords_full(tone_path, 0.001)

    assert len(chords) > 100


@pytest.mark.parametrize(
    "interval", [0, -1, float("nan"), float("inf"), True, None, "1"]
)
def test_invalid_interval_is_rejected(tone_path, interval):
    with pytest.raises(ValueError):
        analyze_chords_full(tone_path, interval)
