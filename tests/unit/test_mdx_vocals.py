import hashlib
import sys
import types

import numpy as np
import pytest

from korshmer.audio_processing import mdx_vocals as mv

RATE = 44100
ERROR_LIMIT_DB = -40.0
BAND_HZ = (100.0, 16000.0)


class IdentitySession:
    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def run(self, _outputs, feed):
        self.calls += 1
        return [feed["input"].copy()]


@pytest.fixture
def identity_session(monkeypatch):
    session = IdentitySession()
    monkeypatch.setattr(mv, "_open_session", lambda _path, _providers: session)
    return session


def make_mix(seconds, seed=0):
    rng = np.random.default_rng(seed)
    count = int(seconds * RATE)
    t = np.arange(count) / RATE
    tones = sum(
        np.sin(2 * np.pi * freq * t + phase)
        for freq, phase in ((220.0, 0.0), (1000.0, 1.0), (5000.0, 2.0))
    )
    left = 0.2 * tones + 0.05 * rng.standard_normal(count)
    right = 0.15 * tones[::-1] + 0.05 * rng.standard_normal(count)
    return np.stack([left, right]).astype(np.float32)


def band_error_db(reference, result):
    spectrum = np.fft.rfftfreq(reference.shape[1], 1.0 / RATE)
    band = (spectrum >= BAND_HZ[0]) & (spectrum <= BAND_HZ[1])
    error_power = signal_power = 0.0
    for channel in range(2):
        ref = np.fft.rfft(reference[channel]) * band
        out = np.fft.rfft(result[channel]) * band
        signal_power += float(np.sum(np.abs(ref) ** 2))
        error_power += float(np.sum(np.abs(out - ref) ** 2))
    return 10.0 * np.log10(error_power / signal_power + 1e-30)


@pytest.mark.parametrize("overlap", [0.0, 0.25])
@pytest.mark.parametrize("extra_samples", [0, 1234])
def test_identity_model_reconstructs_the_mix(
    identity_session, overlap, extra_samples
):
    mix = make_mix(12.0)
    mix = np.concatenate([mix, mix[:, :extra_samples]], axis=1)
    params = mv.MdxParams(overlap=overlap)
    result = mv.separate_vocals(mix, RATE, "unused.onnx", params)
    assert result.shape == mix.shape
    assert result.dtype == np.float32
    assert identity_session.calls >= 3
    assert band_error_db(mix, result) < ERROR_LIMIT_DB


def test_short_clip_and_quiet_level_survive_the_round_trip(identity_session):
    mix = make_mix(0.5) * 0.01
    result = mv.separate_vocals(mix, RATE, "unused.onnx")
    assert result.shape == mix.shape
    assert band_error_db(mix, result) < ERROR_LIMIT_DB


def test_silence_returns_silence(monkeypatch):
    def never_open(_path, _providers):
        raise AssertionError("silence must not load the model")

    monkeypatch.setattr(mv, "_open_session", never_open)
    silence = np.zeros((2, RATE), dtype=np.float32)
    result = mv.separate_vocals(silence, RATE, "unused.onnx")
    assert result.shape == silence.shape
    assert not np.any(result)


@pytest.mark.parametrize(
    "mix, rate",
    [
        (np.zeros(RATE, dtype=np.float32), RATE),
        (np.zeros((3, RATE), dtype=np.float32), RATE),
        (np.zeros((1, RATE), dtype=np.float32), RATE),
        (np.zeros((2, RATE), dtype=np.float32), 22050),
    ],
)
def test_wrong_shape_or_rate_is_rejected(mix, rate):
    with pytest.raises(ValueError):
        mv.separate_vocals(mix, rate, "unused.onnx")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"overlap": 0.95},
        {"overlap": -0.1},
        {"dim_f": 5000},
        {"dim_t": 4},
        {"sample_rate": 100},
    ],
)
def test_params_are_validated(kwargs):
    with pytest.raises(ValueError, match="got"):
        mv.MdxParams(**kwargs)


def test_stop_request_interrupts(identity_session):
    mix = make_mix(12.0)
    polls = []

    def should_stop():
        polls.append(1)
        return len(polls) > 1

    with pytest.raises(InterruptedError):
        mv.separate_vocals(mix, RATE, "unused.onnx", should_stop=should_stop)
    assert len(polls) == 2
    assert identity_session.calls == 1


def test_progress_reports_every_chunk(identity_session):
    mix = make_mix(12.0)
    seen = []
    mv.separate_vocals(
        mix,
        RATE,
        "unused.onnx",
        progress=lambda fraction, text: seen.append(fraction),
    )
    assert len(seen) == identity_session.calls
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


def test_gpu_session_failure_retries_on_the_cpu(monkeypatch, tmp_path):
    attempts = []

    class FakeOptions:
        pass

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            attempts.append(list(providers))
            if "CUDAExecutionProvider" in providers:
                raise RuntimeError("no cuda libraries")

    fake = types.SimpleNamespace(
        SessionOptions=FakeOptions,
        InferenceSession=FakeSession,
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    session = mv._open_session(tmp_path / "model.onnx", None)
    assert isinstance(session, FakeSession)
    assert attempts == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    payload = b"not a real model" * 100
    source = tmp_path / "source.onnx"
    source.write_bytes(payload)
    monkeypatch.setattr(mv, "MDX_MODEL_URL", source.as_uri())
    monkeypatch.setattr(
        mv, "MDX_MODEL_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(mv, "DOWNLOAD_RETRY_DELAY_S", 0.0)
    return types.SimpleNamespace(
        source=source, payload=payload, models_dir=tmp_path / "models"
    )


def test_ensure_model_rejects_bad_hash_and_never_raises(
    fake_model, monkeypatch
):
    monkeypatch.setattr(mv, "MDX_MODEL_SHA256", "0" * 64)
    assert mv.ensure_mdx_model(fake_model.models_dir) is None
    target = mv.mdx_model_path(fake_model.models_dir)
    assert not target.exists()
    assert not list(target.parent.glob("*.part"))

    def offline(*_args, **_kwargs):
        raise OSError("network is unreachable")

    monkeypatch.setattr(mv.urllib.request, "urlopen", offline)
    assert mv.ensure_mdx_model(fake_model.models_dir) is None
    monkeypatch.setattr(mv, "MDX_MODEL_URL", "not a url")
    assert mv.ensure_mdx_model(fake_model.models_dir) is None
    assert mv.ensure_mdx_model(fake_model.models_dir, download=False) is None


def test_ensure_model_downloads_a_good_file_once(fake_model):
    progress = []
    path = mv.ensure_mdx_model(
        fake_model.models_dir,
        progress=lambda fraction, text: progress.append(fraction),
    )
    assert path == mv.mdx_model_path(fake_model.models_dir)
    assert path.read_bytes() == fake_model.payload
    assert progress[-1] == pytest.approx(1.0)
    assert not list(path.parent.glob("*.part"))


def test_ensure_model_accepts_verified_file_without_network(
    fake_model, monkeypatch
):
    target = mv.mdx_model_path(fake_model.models_dir)
    target.parent.mkdir(parents=True)
    target.write_bytes(fake_model.payload)

    def no_network(*_args, **_kwargs):
        raise AssertionError("the network must not be used")

    monkeypatch.setattr(mv.urllib.request, "urlopen", no_network)
    assert mv.ensure_mdx_model(fake_model.models_dir, download=False) == target
    assert mv._verified_sidecar(target).is_file()

    def no_rehash(_path):
        raise AssertionError("a verified file must not be hashed again")

    with monkeypatch.context() as patch:
        patch.setattr(mv, "_sha256_of", no_rehash)
        assert mv.ensure_mdx_model(fake_model.models_dir) == target

    target.write_bytes(fake_model.payload + b"damage")
    assert mv.ensure_mdx_model(fake_model.models_dir, download=False) is None


def test_stop_request_ends_the_download_without_retrying(fake_model):
    calls = []

    def should_stop():
        calls.append(1)
        return True

    with pytest.raises(InterruptedError):
        mv.ensure_mdx_model(fake_model.models_dir, should_stop=should_stop)
    target = mv.mdx_model_path(fake_model.models_dir)
    assert len(calls) == 1
    assert not target.exists()
    assert not list(target.parent.glob("*.part"))
