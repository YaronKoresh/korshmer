import numpy as np
from scipy import signal

from .validation import (
    passthrough_empty_audio,
    validate_effect_number,
    validate_sample_rate,
)


class RangeEQ:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sr = validate_sample_rate(sample_rate)

    def _get_shaped_weight(self, pos: float, curve_val: float) -> float:
        if curve_val == 0:
            return pos
        elif curve_val > 0:
            return pos ** (1.0 + (curve_val / 1.5))
        else:
            exponent = 1.0 / (1.0 + (abs(curve_val) / 1.5))
            return pos**exponent

    def _design_peaking_sos(
        self, freq: float, gain_db: float, bw_oct: float
    ) -> np.ndarray:
        w0 = 2 * np.pi * freq / self.sr

        A = 10.0 ** (gain_db / 40.0)
        alpha = np.sin(w0) * np.sinh(np.log(2) / 2 * bw_oct * w0 / np.sin(w0))

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        return np.array([b0, b1, b2, a0, a1, a2]) / a0

    @passthrough_empty_audio
    def process(
        self,
        audio_data: np.ndarray,
        freq_start: float = 200.0,
        freq_end: float = 4000.0,
        total_gain_db: float = 0.0,
        peak_pos_pct: float = 50,
        curve_left: float = 0.0,
        curve_right: float = 0.0,
    ) -> np.ndarray:
        validate_effect_number(freq_start, "Start frequency", 1.0, 100000.0)
        validate_effect_number(freq_end, "End frequency", 1.0, 100000.0)
        validate_effect_number(total_gain_db, "Total gain", -48.0, 48.0)
        validate_effect_number(peak_pos_pct, "Peak position", 0.0, 100.0)
        validate_effect_number(curve_left, "Left curve", -10.0, 10.0)
        validate_effect_number(curve_right, "Right curve", -10.0, 10.0)

        if len(audio_data) == 0:
            return audio_data

        nyquist_hz = self.sr / 2.0

        real_start = min(freq_start, freq_end)
        real_end = min(nyquist_hz - 100.0, max(freq_start, freq_end))

        if (real_end - real_start) < 3.0:
            real_end = real_start + 3.0

        log_start = np.log(real_start)
        log_end = np.log(real_end)

        total_octaves = (log_end - log_start) / np.log(2.0)
        safe_octaves = max(abs(total_octaves), 0.1)

        bands_per_octave = 4.0
        calc_bands = int(0.5 + (safe_octaves * bands_per_octave))
        bands = max(3, calc_bands)

        slice_width = safe_octaves / bands
        final_width = slice_width * 1.5

        bias_factor = peak_pos_pct / 100.0
        log_center = log_start + (bias_factor * (log_end - log_start))
        f_center = np.exp(log_center)

        sos_chain = []

        for i in range(bands):
            fraction = i / (bands - 1)
            log_curr = log_start + fraction * (log_end - log_start)
            current_f = np.exp(log_curr)

            if current_f < f_center:
                range_len = log_center - log_start
                curr_pos = log_curr - log_start
                rel_pos = (curr_pos / range_len) if range_len > 0 else 0
                weight = self._get_shaped_weight(rel_pos, curve_left)
            else:
                range_len = log_end - log_center
                curr_pos = log_end - log_curr
                rel_pos = (curr_pos / range_len) if range_len > 0 else 0
                weight = self._get_shaped_weight(rel_pos, curve_right)

            gain_at_slice = total_gain_db * weight

            if abs(gain_at_slice) > 0.01 and current_f < nyquist_hz:
                sos_band = self._design_peaking_sos(
                    current_f, gain_at_slice, final_width
                )
                sos_chain.append(sos_band)

        if not sos_chain:
            return audio_data

        full_sos = np.vstack(sos_chain)

        if audio_data.ndim > 1:
            channels = []
            for ch in range(audio_data.shape[0]):
                channels.append(signal.sosfilt(full_sos, audio_data[ch]))
            return np.array(channels)
        else:
            return signal.sosfilt(full_sos, audio_data)
