import sys
import types

import numpy as np
import pytest

from korshmer.exceptions import GPUNotAvailableError
from korshmer.processing.device import DeviceManager


def _fake_torch(cuda=False, mps=False):
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    t.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return t


def test_default_order_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=True))
    assert DeviceManager().select_device() == "cuda"
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True))
    assert DeviceManager().select_device() == "mps"
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    assert DeviceManager().select_device() == "cpu"


def test_unavailable_explicit_device_falls_back_with_warning(
    monkeypatch, caplog
):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    with caplog.at_level("WARNING"):
        assert DeviceManager().select_device("cuda") == "cpu"
    assert "not available" in caplog.text


def test_require_gpu_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    with pytest.raises(GPUNotAvailableError):
        DeviceManager().select_device("cuda", require_gpu=True)


def test_stem_separator_cpu_when_cuda_missing(monkeypatch):
    from korshmer.audio_processing import stem_separator as ss

    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    model = types.SimpleNamespace(to=lambda d: None)
    monkeypatch.setattr(ss, "torch", _fake_torch())
    monkeypatch.setattr(ss, "get_model", lambda n: model)
    monkeypatch.setattr(ss, "apply_model", lambda *a, **k: None)
    assert ss.StemSeparator(device="cuda").device == "cpu"


def test_separate_retries_on_cpu_after_runtime_error(monkeypatch):
    from korshmer.audio_processing import stem_separator as ss

    class Tensor:
        def __init__(self, a):
            self.a = a

        def float(self):
            return self

        def dim(self):
            return self.a.ndim

        def unsqueeze(self, i):
            return Tensor(self.a[None])

        def to(self, d):
            return self

    class FakeTorch(types.ModuleType):
        pass

    ft = _fake_torch(cuda=True)
    ft.from_numpy = Tensor
    ft.no_grad = lambda: __import__("contextlib").nullcontext()
    monkeypatch.setitem(sys.modules, "torch", ft)
    monkeypatch.setattr(ss, "torch", ft)
    moved = []
    model = types.SimpleNamespace(to=moved.append)
    monkeypatch.setattr(ss, "get_model", lambda n: model)

    class Src:
        shape = (1, 4, 2, 3)

        def __getitem__(self, k):
            return types.SimpleNamespace(
                cpu=lambda: types.SimpleNamespace(
                    numpy=lambda: np.zeros((2, 3))
                )
            )

    def apply(m, x, device):
        if device == "cuda":
            raise RuntimeError("out of memory")
        return Src()

    monkeypatch.setattr(ss, "apply_model", apply)
    sep = ss.StemSeparator(device="cuda")
    assert sep.device == "cuda"
    out = sep.separate(np.zeros((2, 3)))
    assert sep.device == "cpu" and "cpu" in moved
    assert set(out) == {"drums", "bass", "other", "vocals"}
