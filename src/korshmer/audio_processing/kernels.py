import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda func: func


def _compress_channel(
    src,
    dst,
    threshold_linear,
    ratio,
    attack_coef,
    release_coef,
    makeup_linear,
):
    envelope = 0.0
    for i in range(src.shape[0]):
        input_level = abs(src[i])

        if input_level > envelope:
            envelope = attack_coef * envelope + (1 - attack_coef) * input_level
        else:
            envelope = (
                release_coef * envelope + (1 - release_coef) * input_level
            )

        if envelope > threshold_linear:
            gain_reduction = (
                threshold_linear + (envelope - threshold_linear) / ratio
            )
            gain = gain_reduction / (envelope + 1e-10)
        else:
            gain = 1.0

        dst[i] = src[i] * gain * makeup_linear


def _limit_channel(src, dst, threshold_linear, release_coef):
    envelope = 0.0
    for i in range(src.shape[0]):
        input_level = abs(src[i])
        envelope = max(input_level, release_coef * envelope)

        if envelope > threshold_linear:
            gain = threshold_linear / (envelope + 1e-10)
        else:
            gain = 1.0

        dst[i] = src[i] * gain


compress_channel_python = _compress_channel
limit_channel_python = _limit_channel

compress_channel = njit(cache=True, fastmath=True)(_compress_channel)
limit_channel = njit(cache=True, fastmath=True)(_limit_channel)


def compress(
    audio_data,
    output,
    threshold_linear,
    ratio,
    attack_coef,
    release_coef,
    makeup_linear,
):
    for ch in range(audio_data.shape[0]):
        compress_channel(
            np.asarray(audio_data[ch]),
            output[ch],
            float(threshold_linear),
            float(ratio),
            float(attack_coef),
            float(release_coef),
            float(makeup_linear),
        )


def limit(audio_data, output, threshold_linear, release_coef):
    for ch in range(audio_data.shape[0]):
        limit_channel(
            np.asarray(audio_data[ch]),
            output[ch],
            float(threshold_linear),
            float(release_coef),
        )
