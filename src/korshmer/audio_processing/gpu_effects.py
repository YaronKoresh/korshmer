from __future__ import annotations

import logging
import threading

import numpy as np
from scipy import signal

from korshmer.exceptions import GPUNotAvailableError

_logger = logging.getLogger(__name__)

BACKEND_PREFERENCES = ("auto", "cuda", "opencl", "cpu")

OPENCL_SOURCE = """
__kernel void gain(
    __global const float *src, __global float *dst, const float factor)
{
    int i = get_global_id(0);
    dst[i] = src[i] * factor;
}

__kernel void soft_clip(
    __global const float *src, __global float *dst, const float drive)
{
    int i = get_global_id(0);
    dst[i] = tanh(src[i] * drive);
}

__kernel void convolve(
    __global const float *src, __global const float *kernel_taps,
    __global float *dst, const int n, const int m)
{
    int i = get_global_id(0);
    int first = max(0, i - n + 1);
    int last = min(i, m - 1);
    float acc = 0.0f;
    for (int k = first; k <= last; k++) {
        acc += kernel_taps[k] * src[i - k];
    }
    dst[i] = acc;
}
"""


class GpuBackend:
    name = "base"
    device_name = ""

    def gain(self, audio: np.ndarray, factor: float) -> np.ndarray:
        raise NotImplementedError

    def soft_clip(self, audio: np.ndarray, drive: float) -> np.ndarray:
        raise NotImplementedError

    def convolve(self, audio: np.ndarray, impulse: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CpuBackend(GpuBackend):
    name = "cpu"
    device_name = "CPU"

    def gain(self, audio, factor):
        return audio * factor

    def soft_clip(self, audio, drive):
        return np.tanh(audio * drive)

    def convolve(self, audio, impulse):
        return signal.fftconvolve(audio, impulse[np.newaxis, :], axes=-1)


class CudaBackend(GpuBackend):
    name = "cuda"

    def __init__(self, cupy_module=None):
        if cupy_module is None:
            import cupy as cupy_module

        if cupy_module.cuda.runtime.getDeviceCount() < 1:
            raise GPUNotAvailableError("No CUDA device found")
        self._cp = cupy_module
        self.device_name = self._read_device_name()

    def _read_device_name(self):
        try:
            props = self._cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            return str(name)
        except Exception:
            return "CUDA"

    def _upload(self, array):
        return self._cp.asarray(array, dtype=self._cp.float32)

    def _download(self, array):
        return np.asarray(self._cp.asnumpy(array), dtype=np.float64)

    def gain(self, audio, factor):
        return self._download(self._upload(audio) * np.float32(factor))

    def soft_clip(self, audio, drive):
        cp = self._cp
        return self._download(cp.tanh(self._upload(audio) * np.float32(drive)))

    def convolve(self, audio, impulse):
        cp = self._cp
        n = audio.shape[-1]
        m = impulse.shape[-1]
        full = n + m - 1
        size = 1 << max(1, (full - 1).bit_length())
        spectrum_audio = cp.fft.rfft(self._upload(audio), n=size, axis=-1)
        spectrum_impulse = cp.fft.rfft(self._upload(impulse), n=size)
        result = cp.fft.irfft(
            spectrum_audio * spectrum_impulse[None, :], n=size, axis=-1
        )
        return self._download(result[..., :full])


class OpenCLBackend(GpuBackend):
    name = "opencl"

    def __init__(self, cl_module=None):
        if cl_module is None:
            import pyopencl as cl_module

        self._cl = cl_module
        device = self._find_gpu_device()
        self.device_name = str(getattr(device, "name", "OpenCL")).strip()
        self._context = cl_module.Context(devices=[device])
        self._queue = cl_module.CommandQueue(self._context)
        self._program = cl_module.Program(self._context, OPENCL_SOURCE).build()

    def _find_gpu_device(self):
        cl = self._cl
        for platform in cl.get_platforms():
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
            except Exception:
                continue
            if devices:
                return devices[0]
        raise GPUNotAvailableError("No OpenCL GPU device found")

    def _buffer(self, array):
        cl = self._cl
        flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
        return cl.Buffer(self._context, flags, hostbuf=array)

    def _output(self, length):
        cl = self._cl
        host = np.empty(length, dtype=np.float32)
        buffer = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, host.nbytes)
        return host, buffer

    def _run_rows(self, audio, kernel_name, head, tail, out_length):
        cl = self._cl
        kernel = getattr(self._program, kernel_name)
        rows = np.atleast_2d(np.ascontiguousarray(audio, dtype=np.float32))
        result = np.empty((rows.shape[0], out_length), dtype=np.float64)
        for index, row in enumerate(rows):
            source = self._buffer(row)
            host, destination = self._output(out_length)
            kernel(
                self._queue,
                (out_length,),
                None,
                source,
                *head,
                destination,
                *tail,
            )
            cl.enqueue_copy(self._queue, host, destination)
            self._queue.finish()
            result[index] = host
        return result.reshape(*audio.shape[:-1], out_length)

    def gain(self, audio, factor):
        return self._run_rows(
            audio,
            "gain",
            (),
            (np.float32(factor),),
            audio.shape[-1],
        )

    def soft_clip(self, audio, drive):
        return self._run_rows(
            audio,
            "soft_clip",
            (),
            (np.float32(drive),),
            audio.shape[-1],
        )

    def convolve(self, audio, impulse):
        taps = np.ascontiguousarray(impulse, dtype=np.float32)
        taps_buffer = self._buffer(taps)
        n = audio.shape[-1]
        m = taps.shape[0]
        return self._run_rows(
            audio,
            "convolve",
            (taps_buffer,),
            (np.int32(n), np.int32(m)),
            n + m - 1,
        )


def _try_cuda():
    try:
        return CudaBackend()
    except Exception as exc:
        _logger.debug(f"CUDA backend unavailable: {exc}")
        return None


def _try_opencl():
    try:
        return OpenCLBackend()
    except Exception as exc:
        _logger.debug(f"OpenCL backend unavailable: {exc}")
        return None


def select_backend(
    preference: str = "auto", require_gpu: bool = False
) -> GpuBackend:
    if preference not in BACKEND_PREFERENCES:
        raise ValueError(
            f"Backend must be one of {', '.join(BACKEND_PREFERENCES)}, "
            f"got {preference!r}"
        )

    if preference == "cpu":
        if require_gpu:
            raise GPUNotAvailableError("A GPU backend is required")
        return CpuBackend()

    factories = {"cuda": _try_cuda, "opencl": _try_opencl}
    order = ("cuda", "opencl") if preference == "auto" else (preference,)
    for key in order:
        backend = factories[key]()
        if backend is not None:
            return backend

    if require_gpu:
        raise GPUNotAvailableError(
            f"No GPU backend is available for '{preference}'"
        )
    if preference != "auto":
        _logger.warning(f"Backend '{preference}' is not available; using cpu")
    return CpuBackend()


class GpuEffects:
    def __init__(
        self,
        preference: str = "auto",
        require_gpu: bool = False,
        backend: GpuBackend | None = None,
    ):
        self._backend = backend or select_backend(preference, require_gpu)
        self._cpu = CpuBackend()
        self._lock = threading.Lock()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def device_name(self) -> str:
        return self._backend.device_name

    @property
    def is_gpu(self) -> bool:
        return self._backend.name != "cpu"

    def _run(self, operation, *args):
        with self._lock:
            backend = self._backend
        try:
            return getattr(backend, operation)(*args)
        except Exception as exc:
            if backend.name == "cpu":
                raise
            _logger.warning(
                f"{backend.name} backend failed in {operation} ({exc}); "
                "using cpu"
            )
            with self._lock:
                self._backend = self._cpu
            return getattr(self._cpu, operation)(*args)

    @staticmethod
    def _as_rows(audio):
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim not in (1, 2):
            raise ValueError(
                f"Audio must be 1D or 2D (channels, samples), got {audio.ndim}D"
            )
        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains NaN or infinite values")
        return audio

    def apply_gain(self, audio, gain_db: float = 0.0) -> np.ndarray:
        audio = self._as_rows(audio)
        if audio.size == 0:
            return audio.copy()
        return self._run("gain", audio, 10 ** (float(gain_db) / 20.0))

    def soft_clip(self, audio, drive_db: float = 10.0) -> np.ndarray:
        audio = self._as_rows(audio)
        if audio.size == 0:
            return audio.copy()
        return self._run("soft_clip", audio, 10 ** (float(drive_db) / 20.0))

    def convolve(
        self, audio, impulse_response, keep_tail: bool = False
    ) -> np.ndarray:
        audio = self._as_rows(audio)
        impulse = np.asarray(impulse_response, dtype=np.float64)
        if impulse.ndim != 1 or impulse.size == 0:
            raise ValueError("Impulse response must be a non-empty 1D array")
        if not np.all(np.isfinite(impulse)):
            raise ValueError("Impulse response contains NaN or infinite values")
        if audio.size == 0:
            return audio.copy()

        rows = np.atleast_2d(audio)
        result = np.asarray(self._run("convolve", rows, impulse))
        if not keep_tail:
            result = result[..., : rows.shape[-1]]
        if audio.ndim == 1:
            return result[0]
        return result

    def convolution_reverb(
        self, audio, impulse_response, wet: float = 0.3
    ) -> np.ndarray:
        if isinstance(wet, bool) or not 0.0 <= float(wet) <= 1.0:
            raise ValueError(f"Wet mix must be between 0 and 1, got {wet!r}")
        audio = self._as_rows(audio)
        if audio.size == 0:
            return audio.copy()
        reverberated = self.convolve(audio, impulse_response)
        return audio * (1.0 - wet) + reverberated * wet


_gpu_effects: GpuEffects | None = None
_gpu_effects_lock = threading.Lock()


def get_gpu_effects() -> GpuEffects:
    global _gpu_effects
    with _gpu_effects_lock:
        if _gpu_effects is None:
            _gpu_effects = GpuEffects()
        return _gpu_effects
