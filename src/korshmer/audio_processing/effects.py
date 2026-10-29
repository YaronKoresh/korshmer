import numpy as np
from scipy.signal import lfilter

from .validation import (
    passthrough_empty_audio,
    validate_delay_time,
    validate_effect_integer,
    validate_effect_number,
    validate_reverb_decay,
    validate_reverb_decay_time,
    validate_sample_rate,
)


class Reverb:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)

    @passthrough_empty_audio
    def process(
        self,
        audio_data,
        room_size=1.0,
        decay=0.9,
        wet=0.3,
        pre_delay_ms=0,
        decay_time_s=None,
    ):
        validate_reverb_decay(decay)
        validate_effect_number(room_size, "Room size", 0.05, 10.0)
        validate_effect_number(wet, "Wet level", 0.0, 1.0)
        validate_effect_number(pre_delay_ms, "Pre-delay", 0.0, 5000.0)
        if decay_time_s is not None:
            validate_reverb_decay_time(decay_time_s)

        pre_delay_samples = int(self.sr * pre_delay_ms / 1000.0)

        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        if decay_time_s is None:
            tail_length = int(self.sr * 2.0)
        else:
            tail_length = int(self.sr * decay_time_s)
        padded_audio = np.pad(
            audio_data, ((0, 0), (0, tail_length)), mode="constant"
        )
        output = np.zeros_like(padded_audio)

        delay_times = [37, 41, 43, 47, 53, 59, 61, 67]
        delays = [
            max(1, int(self.sr * dt / 1000.0 * room_size)) for dt in delay_times
        ]

        if decay_time_s is None:
            feedbacks = [decay * 0.95] * len(delays)
        else:
            feedbacks = [
                10.0 ** (-3.0 * (d / self.sr) / decay_time_s) for d in delays
            ]

        for ch in range(padded_audio.shape[0]):
            input_signal = padded_audio[ch]
            wet_signal = np.zeros_like(input_signal)

            for d, feedback in zip(delays, feedbacks, strict=False):
                comb_wet = np.zeros_like(input_signal)

                if d < len(input_signal):
                    comb_wet[d:] = input_signal[:-d]

                    for i in range(2 * d, len(comb_wet), d):
                        end = min(i + d, len(comb_wet))
                        length = end - i

                        comb_wet[i:end] += (
                            feedback * comb_wet[i - d : i - d + length]
                        )

                wet_signal += comb_wet

            wet_signal /= len(delays)

            if pre_delay_samples > 0:
                wet_signal = np.pad(
                    wet_signal, (pre_delay_samples, 0), mode="constant"
                )[: len(wet_signal)]

            output[ch] = input_signal * (1 - wet) + wet_signal * wet

        if output.shape[0] == 1:
            return output[0]
        return output


class Delay:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)

    @staticmethod
    def _feedback_delay(signal, delay_samples, feedback):
        n = signal.shape[0]
        wet = np.zeros(n, dtype=np.result_type(signal.dtype, np.float64))
        for start in range(delay_samples, n, delay_samples):
            stop = min(start + delay_samples, n)
            length = stop - start
            prev = start - delay_samples
            wet[start:stop] = (
                signal[prev : prev + length]
                + feedback * wet[prev : prev + length]
            )
        return wet

    @passthrough_empty_audio
    def process(self, audio_data, delay_ms=500, feedback=0.3, mix=0.5):
        validate_delay_time(delay_ms)
        validate_effect_number(feedback, "Feedback", 0.0, 0.99)
        validate_effect_number(mix, "Mix", 0.0, 1.0)
        delay_samples = max(1, int(self.sr * delay_ms / 1000.0))

        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        output = np.zeros_like(audio_data)

        for ch in range(audio_data.shape[0]):
            wet_signal = self._feedback_delay(
                audio_data[ch], delay_samples, feedback
            )
            output[ch] = audio_data[ch] * (1 - mix) + wet_signal * mix

        if output.shape[0] == 1:
            return output[0]
        return output


class Chorus:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)

    @passthrough_empty_audio
    def process(self, audio_data, rate_hz=0.3, depth=2.0, mix=0.6):
        validate_effect_number(rate_hz, "Rate", 0.0, 20.0)
        validate_effect_number(depth, "Depth", 0.0, 10.0)
        validate_effect_number(mix, "Mix", 0.0, 1.0)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        output = np.zeros_like(audio_data)
        n_samples = audio_data.shape[1]
        t = np.arange(n_samples)

        base_delays_ms = [12.0, 17.0, 23.0, 29.0]
        rate_multipliers = [1.0, 0.83, 1.13, 0.91]
        phases = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

        swing_samples = int(self.sr * 2.0 / 1000.0 * depth)

        for ch in range(audio_data.shape[0]):
            wet_signal = np.zeros(n_samples)

            stereo_offset = ch * (np.pi / 4)

            for i in range(4):
                base_delay_samples = int(self.sr * base_delays_ms[i] / 1000.0)
                current_rate = rate_hz * rate_multipliers[i]
                current_phase = phases[i] + stereo_offset

                lfo = np.sin(
                    2 * np.pi * current_rate * t / self.sr + current_phase
                )
                delay_in_samples = base_delay_samples + lfo * swing_samples

                read_indices = t - delay_in_samples

                voice = np.interp(
                    read_indices, t, audio_data[ch], left=0, right=0
                )
                wet_signal += voice

            wet_signal /= 4.0
            output[ch] = audio_data[ch] * (1 - mix) + wet_signal * mix

        if output.shape[0] == 1:
            return output[0]
        return output


class Flanger:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)

    @passthrough_empty_audio
    def process(self, audio_data, rate_hz=0.5, depth=0.7, mix=0.5):
        validate_effect_number(rate_hz, "Rate", 0.0, 20.0)
        validate_effect_number(depth, "Depth", 0.0, 5.0)
        validate_effect_number(mix, "Mix", 0.0, 1.0)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        output = np.zeros_like(audio_data)
        n_samples = audio_data.shape[1]
        t = np.arange(n_samples)

        base_delay_samples = int(self.sr * 3 / 1000.0)
        swing_samples = int(self.sr * 2 / 1000.0 * depth)

        for ch in range(audio_data.shape[0]):
            lfo = np.sin(2 * np.pi * rate_hz * t / self.sr)
            delay_in_samples = base_delay_samples + lfo * swing_samples

            read_indices = t - delay_in_samples

            wet_signal = np.interp(
                read_indices, t, audio_data[ch], left=0, right=0
            )

            output[ch] = audio_data[ch] * (1 - mix) + wet_signal * mix

        if output.shape[0] == 1:
            return output[0]
        return output


class Phaser:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)

    @passthrough_empty_audio
    def process(
        self,
        audio_data,
        rate_hz=0.5,
        depth=0.7,
        stages=4,
        mix=0.5,
        block_size=64,
    ):
        validate_effect_number(rate_hz, "Rate", 0.0, 20.0)
        validate_effect_number(depth, "Depth", 0.0, 1.0)
        stages = validate_effect_integer(stages, "Stages", 1, 12)
        validate_effect_number(mix, "Mix", 0.0, 1.0)
        block_size = validate_effect_integer(block_size, "Block size", 1, 4096)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        output = np.zeros_like(audio_data)
        n_samples = audio_data.shape[1]
        t = np.arange(n_samples) / self.sr

        min_freq = 200.0
        max_freq = 4000.0

        for ch in range(audio_data.shape[0]):
            wet = np.copy(audio_data[ch])

            for stage in range(stages):
                phase_offset = stage * np.pi / stages
                mod = np.sin(2 * np.pi * rate_hz * t + phase_offset) * depth
                center_freq = min_freq + (max_freq - min_freq) * (
                    mod * 0.5 + 0.5
                )
                omega = 2 * np.pi * center_freq / self.sr
                allpass_coeff = (np.tan(omega / 2) - 1) / (
                    np.tan(omega / 2) + 1
                )

                filtered = np.zeros(n_samples)
                z = np.array([0.0])

                for i in range(0, n_samples, block_size):
                    end = min(i + block_size, n_samples)
                    c = np.mean(allpass_coeff[i:end])

                    b = [c, 1.0]
                    a = [1.0, c]

                    chunk, z = lfilter(b, a, wet[i:end], zi=z)
                    filtered[i:end] = chunk

                wet = filtered

            output[ch] = audio_data[ch] * (1 - mix) + wet * mix

        if output.shape[0] == 1:
            return output[0]
        return output


class Tremolo:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)

    @passthrough_empty_audio
    def process(self, audio_data, rate_hz=5.0, depth=0.5, shape="sine"):
        validate_effect_number(rate_hz, "Rate", 0.0, 100.0)
        validate_effect_number(depth, "Depth", 0.0, 1.0)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        n_samples = audio_data.shape[1]
        t = np.arange(n_samples) / self.sr

        if shape == "square":
            lfo = np.sign(np.sin(2 * np.pi * rate_hz * t))
        elif shape == "triangle":
            lfo = (
                2 * np.abs(2 * (rate_hz * t - np.floor(rate_hz * t + 0.5))) - 1
            )
        else:
            lfo = np.sin(2 * np.pi * rate_hz * t)

        modulation = 1.0 - depth * (lfo * 0.5 + 0.5)
        output = audio_data * modulation[np.newaxis, :]

        if output.shape[0] == 1:
            return output[0]
        return output


class StereoWidener:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)

    @passthrough_empty_audio
    def process(self, audio_data, width=1.5):
        if audio_data.ndim == 1:
            return audio_data

        if audio_data.shape[0] < 2:
            return audio_data

        left = audio_data[0]
        right = audio_data[1]

        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        side_scaled = side * width

        new_left = mid + side_scaled
        new_right = mid - side_scaled

        peak = max(np.max(np.abs(new_left)), np.max(np.abs(new_right)))
        if peak > 1.0:
            new_left /= peak
            new_right /= peak

        output = np.vstack((new_left, new_right))
        return output
