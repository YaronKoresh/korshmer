import numpy as np
import pytest

from korshmer.audio_processing.distortion_noise import NoiseReduction
from korshmer.audio_processing.validation import (
    validate_spectral_gate_threshold,
)


class TestValidateSpectralGateThreshold:
    @pytest.mark.parametrize("db", [-60, -120, 0, -120.0, 0.0, -0.5])
    def test_in_range(self, db):
        assert validate_spectral_gate_threshold(db) == db

    @pytest.mark.parametrize("db", [1, -130, 0.1, -120.5])
    def test_out_of_range(self, db):
        with pytest.raises(ValueError):
            validate_spectral_gate_threshold(db)

    @pytest.mark.parametrize(
        "db", [True, False, None, float("nan"), float("inf"), "0"]
    )
    def test_invalid_types(self, db):
        with pytest.raises(ValueError):
            validate_spectral_gate_threshold(db)


class TestNoiseReductionGate:
    def _signal(self, n=4096):
        rng = np.random.default_rng(3)
        return (rng.standard_normal(n) * 0.1).astype(np.float64)

    def test_no_gate_no_regression(self):
        nr = NoiseReduction()
        audio = self._signal()

        out = nr.process(audio)
        assert np.array_equal(out, audio)

    def _energy(self, x):
        return float(np.sum(x.astype(np.float64) ** 2))

    def test_higher_threshold_removes_more_energy(self):
        nr = NoiseReduction()
        audio = self._signal()
        e_none = self._energy(nr.process(audio))
        e_low = self._energy(nr.process(audio, spectral_gate_db=-120))
        e_high = self._energy(nr.process(audio, spectral_gate_db=0))

        assert e_high < e_low <= e_none

    def test_gate_minus_120_barely_changes(self):
        nr = NoiseReduction()
        audio = self._signal()
        out = nr.process(audio, spectral_gate_db=-120)
        assert np.max(np.abs(out)) > 0.0

    def test_gate_out_of_range_raises(self):
        nr = NoiseReduction()
        with pytest.raises(ValueError):
            nr.process(self._signal(), spectral_gate_db=5)
