import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from korshmer.audio_processing.basic_effects import apply_normalize
from korshmer.audio_processing.dynamics import Limiter
from korshmer.audio_processing.metering import AudioMetering

PROPS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)

_values = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

mono_audio = hnp.arrays(
    np.float32, st.integers(1, 4096).map(lambda n: (n,)), elements=_values
)
stereo_audio = hnp.arrays(
    np.float32,
    st.integers(1, 4096).map(lambda n: (2, n)),
    elements=_values,
)
any_audio = st.one_of(mono_audio, stereo_audio)


@PROPS
@given(audio=any_audio)
def test_limiter_preserves_shape_and_is_finite(audio):
    out = Limiter(44100).process(audio)
    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))


@PROPS
@given(audio=any_audio)
def test_normalize_peak_at_most_one(audio):
    out = apply_normalize(audio)
    assert out.shape == audio.shape
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


@PROPS
@given(
    shape=st.one_of(
        st.integers(1, 4096).map(lambda n: (n,)),
        st.integers(1, 4096).map(lambda n: (2, n)),
    )
)
def test_normalize_keeps_silence_silent(shape):
    out = apply_normalize(np.zeros(shape, dtype=np.float32))
    assert not np.any(out)


@PROPS
@given(audio=any_audio)
def test_metering_peak_not_below_rms(audio):
    meter = AudioMetering(44100)
    assert meter.peak_level(audio) >= meter.rms_level(audio) - 1e-4
