from collections.abc import Callable
from typing import TypeVar, overload

import numpy as np

_Kernel = TypeVar("_Kernel", bound=Callable[..., object])


@overload
def _passthrough_njit(function: _Kernel, /) -> _Kernel: ...


@overload
def _passthrough_njit(**kwargs: object) -> Callable[[_Kernel], _Kernel]: ...


def _passthrough_njit(*args: object, **kwargs: object) -> object:
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda func: func


njit: Callable[..., Callable[[Callable[..., object]], Callable[..., object]]]
try:
    from numba import njit

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    njit = _passthrough_njit


def _compress_channel(
    src,
    dst,
    threshold_linear,
    ratio,
    attack_coef,
    release_coef,
    makeup_linear,
):
    slope = 1.0 / ratio - 1.0
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
            gain = (envelope / threshold_linear) ** slope
        else:
            gain = 1.0

        dst[i] = src[i] * gain * makeup_linear


def _compress_linked(
    src,
    dst,
    threshold_linear,
    ratio,
    attack_coef,
    release_coef,
    makeup_linear,
):
    slope = 1.0 / ratio - 1.0
    channels = src.shape[0]
    envelope = 0.0
    for i in range(src.shape[1]):
        input_level = 0.0
        for ch in range(channels):
            level = abs(src[ch, i])
            if level > input_level:
                input_level = level

        if input_level > envelope:
            envelope = attack_coef * envelope + (1 - attack_coef) * input_level
        else:
            envelope = (
                release_coef * envelope + (1 - release_coef) * input_level
            )

        if envelope > threshold_linear:
            gain = (envelope / threshold_linear) ** slope * makeup_linear
        else:
            gain = makeup_linear

        for ch in range(channels):
            dst[ch, i] = src[ch, i] * gain


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
compress_linked_python = _compress_linked
limit_channel_python = _limit_channel

compress_channel = njit(cache=True, fastmath=True)(_compress_channel)
compress_linked = njit(cache=True, fastmath=True)(_compress_linked)
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
    settings = (
        float(threshold_linear),
        float(ratio),
        float(attack_coef),
        float(release_coef),
        float(makeup_linear),
    )
    if audio_data.shape[0] == 1:
        compress_channel(np.asarray(audio_data[0]), output[0], *settings)
    else:
        compress_linked(np.asarray(audio_data), output, *settings)


def limit(audio_data, output, threshold_linear, release_coef):
    for ch in range(audio_data.shape[0]):
        limit_channel(
            np.asarray(audio_data[ch]),
            output[ch],
            float(threshold_linear),
            float(release_coef),
        )
