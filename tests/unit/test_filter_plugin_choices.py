import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing.basic_effects import apply_filter
from korshmer.plugins.builtin import FilterPlugin

SR = 22050


def _write_tone(path):
    t = np.arange(SR) / SR
    sf.write(str(path), np.sin(2 * np.pi * 440 * t).astype(np.float32), SR)


def _choices():
    return next(
        p.choices for p in FilterPlugin.params() if p.name == "filter_type"
    )


def test_filter_plugin_offers_only_types_the_filter_accepts():
    for choice in _choices():
        apply_filter(np.zeros(2048, dtype=np.float32), SR, choice, 1000)


def test_filter_plugin_runs_every_offered_type(tmp_path):
    source = tmp_path / "in.wav"
    _write_tone(source)

    for index, choice in enumerate(_choices()):
        target = tmp_path / f"out{index}.wav"
        result = FilterPlugin().process(
            str(source), str(target), filter_type=choice, cutoff=1000
        )
        assert result == str(target)
        assert target.exists()


def test_filter_plugin_does_not_offer_the_unsupported_band_type():
    assert "band" not in _choices()
    with pytest.raises(ValueError, match="Unknown filter_type"):
        apply_filter(np.zeros(2048, dtype=np.float32), SR, "band", 1000)
