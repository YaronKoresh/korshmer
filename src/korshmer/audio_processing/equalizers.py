import numpy as np
from scipy import signal

from .validation import (
    MAX_EQ_BANDS,
    passthrough_empty_audio,
    validate_effect_integer,
    validate_effect_number,
    validate_eq_band_count,
    validate_q_factor,
    validate_sample_rate,
)


class ParametricEQ:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)
        self.bands = []

    def add_band(
        self, freq, gain_db, q_factor=1.0, filter_type="peaking"
    ) -> None:
        validate_effect_number(freq, "Frequency", 10.0, self.sr / 2)
        validate_effect_number(gain_db, "Gain", -48.0, 48.0)
        validate_q_factor(q_factor)
        if filter_type not in ("peaking", "lowpass", "highpass"):
            raise ValueError(
                f"Unknown filter_type: {filter_type!r}; use 'peaking', "
                f"'lowpass' or 'highpass'"
            )
        if len(self.bands) >= MAX_EQ_BANDS:
            raise ValueError(
                f"EQ band count would exceed maximum of {MAX_EQ_BANDS} "
                f"(currently {len(self.bands)})"
            )
        self.bands.append(
            {
                "freq": freq,
                "gain_db": gain_db,
                "q": q_factor,
                "type": filter_type,
            }
        )

    def clear_bands(self) -> None:
        self.bands = []

    def _design_peaking(self, freq, gain_db, q):
        w0 = 2 * np.pi * freq / self.sr
        A = 10.0 ** (gain_db / 40.0)
        alpha = np.sin(w0) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
        return sos

    def _design_lowpass(self, freq, q):
        w0 = 2 * np.pi * freq / self.sr
        alpha = np.sin(w0) / (2 * q)

        b0 = (1 - np.cos(w0)) / 2
        b1 = 1 - np.cos(w0)
        b2 = (1 - np.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha

        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
        return sos

    def _design_highpass(self, freq, q):
        w0 = 2 * np.pi * freq / self.sr
        alpha = np.sin(w0) / (2 * q)

        b0 = (1 + np.cos(w0)) / 2
        b1 = -(1 + np.cos(w0))
        b2 = (1 + np.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha

        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
        return sos

    @passthrough_empty_audio
    def process(self, audio_data):
        if len(self.bands) == 0:
            return audio_data

        sos_list = []
        for band in self.bands:
            if band["type"] == "peaking":
                sos = self._design_peaking(
                    band["freq"], band["gain_db"], band["q"]
                )
            elif band["type"] == "lowpass":
                sos = self._design_lowpass(band["freq"], band["q"])
            elif band["type"] == "highpass":
                sos = self._design_highpass(band["freq"], band["q"])
            else:
                continue
            sos_list.append(sos)

        if not sos_list:
            return audio_data

        full_sos = np.concatenate(sos_list, axis=0)

        if audio_data.ndim > 1:
            channels = []
            for ch in range(audio_data.shape[0]):
                channels.append(signal.sosfilt(full_sos, audio_data[ch]))
            return np.array(channels)
        else:
            return signal.sosfilt(full_sos, audio_data)


class GraphicEQ:
    def __init__(self, sample_rate=44100, bands=10):
        self.sr = validate_sample_rate(sample_rate)
        self.n_bands = validate_eq_band_count(bands)
        self.gains = np.zeros(self.n_bands)
        self.frequencies = self._generate_frequencies()

    def _generate_frequencies(self):
        return np.logspace(np.log10(20), np.log10(20000), self.n_bands)

    def set_gain(self, band_index, gain_db):
        validate_effect_integer(band_index, "Band index", -(10**9), 10**9)
        validate_effect_number(gain_db, "Gain", -48.0, 48.0)
        if 0 <= band_index < self.n_bands:
            self.gains[band_index] = gain_db

    @passthrough_empty_audio
    def process(self, audio_data):
        peq = ParametricEQ(self.sr)
        for i, freq in enumerate(self.frequencies):
            if abs(self.gains[i]) > 0.01 and freq <= self.sr / 2:
                peq.add_band(freq, self.gains[i], q_factor=1.4)
        return peq.process(audio_data)
