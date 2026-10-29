import ctypes
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

from korshmer.audio_processing import gpu_effects
from korshmer.audio_processing.gpu_effects import (
    OPENCL_SOURCE,
    CpuBackend,
    CudaBackend,
    GpuEffects,
    OpenCLBackend,
    select_backend,
)
from korshmer.exceptions import GPUNotAvailableError


@pytest.fixture
def stereo():
    return np.random.default_rng(3).uniform(-0.9, 0.9, (2, 3000))


@pytest.fixture
def impulse():
    rng = np.random.default_rng(4)
    return rng.uniform(-1, 1, 200) * np.exp(-np.arange(200) / 40.0)


def _fake_cupy(devices=1):
    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: devices,
        getDeviceProperties=lambda index: {"name": b"Fake CUDA GPU"},
    )
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(runtime=runtime),
        asarray=np.asarray,
        asnumpy=np.asarray,
        tanh=np.tanh,
        float32=np.float32,
        fft=np.fft,
    )


class _FakeBuffer:
    def __init__(self, context, flags, size=None, hostbuf=None):
        self.data = None if hostbuf is None else np.array(hostbuf)
        self.size = size


def _fake_opencl(gpu=True):
    log = []

    def kernel_gain(src, dst, factor):
        dst.data = src.data * factor

    def kernel_soft_clip(src, dst, drive):
        dst.data = np.tanh(src.data * drive).astype(np.float32)

    def kernel_convolve(src, taps, dst, n, m):
        assert src.data.shape[0] == n
        assert taps.data.shape[0] == m
        dst.data = np.convolve(src.data, taps.data).astype(np.float32)

    kernels = {
        "gain": (kernel_gain, 2),
        "soft_clip": (kernel_soft_clip, 2),
        "convolve": (kernel_convolve, 3),
    }

    class Device:
        name = "Fake OpenCL GPU "

    class Platform:
        def get_devices(self, device_type=None):
            return [Device()] if gpu else []

    class Program:
        def __init__(self, context, source):
            assert "__kernel void convolve" in source

        def build(self):
            return self

        def __getattr__(self, name):
            function, _ = kernels[name]

            def launch(queue, global_size, local_size, *args):
                log.append((name, global_size))
                buffers = [a for a in args if isinstance(a, _FakeBuffer)]
                scalars = [a for a in args if not isinstance(a, _FakeBuffer)]
                function(*buffers, *scalars)

            return launch

    class Queue:
        def __init__(self, context):
            pass

        def finish(self):
            pass

    def enqueue_copy(queue, host, buffer):
        host[:] = buffer.data

    return types.SimpleNamespace(
        get_platforms=lambda: [Platform()],
        device_type=types.SimpleNamespace(GPU=4),
        mem_flags=types.SimpleNamespace(
            READ_ONLY=1, WRITE_ONLY=2, COPY_HOST_PTR=4
        ),
        Context=lambda devices: object(),
        CommandQueue=Queue,
        Program=Program,
        Buffer=_FakeBuffer,
        enqueue_copy=enqueue_copy,
        launches=log,
    )


@pytest.mark.parametrize(
    "backend_factory",
    [
        lambda: CudaBackend(_fake_cupy()),
        lambda: OpenCLBackend(_fake_opencl()),
    ],
    ids=["cuda", "opencl"],
)
def test_gpu_backends_match_cpu(backend_factory, stereo, impulse):
    gpu = backend_factory()
    cpu = CpuBackend()
    assert np.allclose(gpu.gain(stereo, 0.5), cpu.gain(stereo, 0.5), atol=1e-5)
    assert np.allclose(
        gpu.soft_clip(stereo, 3.0), cpu.soft_clip(stereo, 3.0), atol=1e-5
    )
    expected = cpu.convolve(stereo, impulse)
    result = gpu.convolve(stereo, impulse)
    assert result.shape == (2, 3000 + 200 - 1)
    assert np.allclose(result, expected, atol=1e-3)


def test_backend_device_names():
    assert CudaBackend(_fake_cupy()).device_name == "Fake CUDA GPU"
    assert OpenCLBackend(_fake_opencl()).device_name == "Fake OpenCL GPU"


def test_cuda_backend_without_device_raises():
    with pytest.raises(GPUNotAvailableError):
        CudaBackend(_fake_cupy(devices=0))


def test_opencl_backend_without_gpu_raises():
    with pytest.raises(GPUNotAvailableError):
        OpenCLBackend(_fake_opencl(gpu=False))


def test_opencl_launches_one_kernel_per_channel(stereo):
    cl = _fake_opencl()
    OpenCLBackend(cl).gain(stereo, 2.0)
    assert cl.launches == [("gain", (3000,)), ("gain", (3000,))]


def test_auto_prefers_cuda_then_opencl_then_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setitem(sys.modules, "pyopencl", _fake_opencl())
    assert select_backend("auto").name == "cuda"
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert select_backend("auto").name == "opencl"
    monkeypatch.setitem(sys.modules, "pyopencl", None)
    assert select_backend("auto").name == "cpu"


def test_missing_libraries_fall_back_to_cpu(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(sys.modules, "pyopencl", None)
    with caplog.at_level("WARNING"):
        assert select_backend("cuda").name == "cpu"
    assert "not available" in caplog.text
    assert select_backend("opencl").name == "cpu"
    assert select_backend("cpu").name == "cpu"


def test_require_gpu_raises_without_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(sys.modules, "pyopencl", None)
    with pytest.raises(GPUNotAvailableError):
        select_backend("auto", require_gpu=True)
    with pytest.raises(GPUNotAvailableError):
        select_backend("cpu", require_gpu=True)


def test_unknown_preference_is_rejected():
    with pytest.raises(ValueError):
        select_backend("metal")


def test_effects_on_cpu_match_numpy(stereo, impulse):
    effects = GpuEffects("cpu")
    assert not effects.is_gpu
    assert effects.backend_name == "cpu"
    assert np.allclose(effects.apply_gain(stereo, 6.0), stereo * 10 ** (6 / 20))
    assert np.allclose(
        effects.soft_clip(stereo, 12.0), np.tanh(stereo * 10 ** (12 / 20))
    )
    assert effects.convolve(stereo, impulse).shape == stereo.shape
    tail = effects.convolve(stereo, impulse, keep_tail=True)
    assert tail.shape == (2, 3000 + 199)


def test_mono_input_stays_mono(impulse):
    mono = np.linspace(-0.5, 0.5, 500)
    effects = GpuEffects("cpu")
    assert effects.apply_gain(mono, 0).shape == mono.shape
    assert effects.soft_clip(mono).shape == mono.shape
    assert effects.convolve(mono, impulse).shape == mono.shape


def test_convolution_reverb_mixes_dry_and_wet(stereo, impulse):
    effects = GpuEffects("cpu")
    dry = effects.convolution_reverb(stereo, impulse, wet=0.0)
    assert np.allclose(dry, stereo)
    wet = effects.convolution_reverb(stereo, impulse, wet=1.0)
    assert np.allclose(wet, effects.convolve(stereo, impulse))
    with pytest.raises(ValueError):
        effects.convolution_reverb(stereo, impulse, wet=1.5)


def test_empty_audio_is_returned_as_copy(impulse):
    effects = GpuEffects("cpu")
    empty = np.zeros((2, 0))
    for result in (
        effects.apply_gain(empty),
        effects.soft_clip(empty),
        effects.convolve(empty, impulse),
        effects.convolution_reverb(empty, impulse),
    ):
        assert result.shape == (2, 0)


@pytest.mark.parametrize(
    "bad",
    [np.array([0.1, np.nan]), np.zeros((2, 2, 2))],
    ids=["nan", "3d"],
)
def test_invalid_audio_is_rejected(bad):
    with pytest.raises(ValueError):
        GpuEffects("cpu").apply_gain(bad)


def test_invalid_impulse_is_rejected(stereo):
    effects = GpuEffects("cpu")
    with pytest.raises(ValueError):
        effects.convolve(stereo, np.array([]))
    with pytest.raises(ValueError):
        effects.convolve(stereo, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        effects.convolve(stereo, np.array([1.0, np.inf]))


def test_gpu_failure_switches_to_cpu(stereo, caplog):
    class Failing(CpuBackend):
        name = "cuda"

        def gain(self, audio, factor):
            raise RuntimeError("out of memory")

    effects = GpuEffects(backend=Failing())
    assert effects.is_gpu
    with caplog.at_level("WARNING"):
        result = effects.apply_gain(stereo, 0.0)
    assert np.allclose(result, stereo)
    assert effects.backend_name == "cpu"
    assert "out of memory" in caplog.text


def test_cpu_failure_is_not_swallowed(stereo):
    class Broken(CpuBackend):
        def gain(self, audio, factor):
            raise RuntimeError("broken")

    with pytest.raises(RuntimeError):
        GpuEffects(backend=Broken()).apply_gain(stereo)


def test_get_gpu_effects_is_a_singleton(monkeypatch):
    monkeypatch.setattr(gpu_effects, "_gpu_effects", None)
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(sys.modules, "pyopencl", None)
    first = gpu_effects.get_gpu_effects()
    assert first is gpu_effects.get_gpu_effects()
    assert first.backend_name == "cpu"


_C_SHIM = """
#include <math.h>
#define __kernel
#define __global
static int current_id;
static int get_global_id(int dim) { return current_id; }
static int max(int a, int b) { return a > b ? a : b; }
static int min(int a, int b) { return a < b ? a : b; }
%s
void run_gain(const float *s, float *d, int count, float f) {
    for (current_id = 0; current_id < count; current_id++) gain(s, d, f);
}
void run_soft_clip(const float *s, float *d, int count, float f) {
    for (current_id = 0; current_id < count; current_id++) soft_clip(s, d, f);
}
void run_convolve(const float *s, const float *k, float *d, int n, int m) {
    for (current_id = 0; current_id < n + m - 1; current_id++)
        convolve(s, k, d, n, m);
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
def test_opencl_kernel_source_matches_numpy(tmp_path, stereo, impulse):
    source = tmp_path / "kernels.c"
    library = tmp_path / "kernels.so"
    source.write_text(_C_SHIM % OPENCL_SOURCE)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(library), str(source), "-lm"],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    float_p = ctypes.POINTER(ctypes.c_float)

    def pointer(array):
        return array.ctypes.data_as(float_p)

    src = np.ascontiguousarray(stereo[0], dtype=np.float32)
    taps = np.ascontiguousarray(impulse, dtype=np.float32)
    n, m = src.shape[0], taps.shape[0]

    out = np.zeros(n, dtype=np.float32)
    lib.run_gain(pointer(src), pointer(out), n, ctypes.c_float(0.25))
    assert np.allclose(out, src * 0.25, atol=1e-6)

    lib.run_soft_clip(pointer(src), pointer(out), n, ctypes.c_float(4.0))
    assert np.allclose(out, np.tanh(src * 4.0), atol=1e-6)

    full = np.zeros(n + m - 1, dtype=np.float32)
    lib.run_convolve(pointer(src), pointer(taps), pointer(full), n, m)
    assert np.allclose(full, np.convolve(src, taps), atol=1e-3)
