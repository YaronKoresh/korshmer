import logging
import threading

_logger = logging.getLogger(__name__)

_GPU_DEVICE_TYPES = ("cuda", "mps")


class DeviceManager:
    def __init__(self):
        self.device_type = "cpu"
        self.device_name = "CPU"

        try:
            import torch

            if torch.cuda.is_available():
                self.device_type = "cuda"
                self.device_name = "CUDA"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.device_type = "mps"
                self.device_name = "Metal Performance Shaders (macOS)"
        except ImportError:
            pass

    @staticmethod
    def _is_available(torch, device_type: str) -> bool:
        if device_type == "cpu":
            return True
        if device_type == "cuda":
            return bool(torch.cuda.is_available())
        if device_type == "mps":
            return bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        return False

    def select_device(
        self, requested: str | None = None, require_gpu: bool = False
    ) -> str:
        from korshmer.exceptions import GPUNotAvailableError

        try:
            import torch
        except ImportError:
            torch = None

        def available(device_type):
            if device_type == "cpu":
                return True
            return torch is not None and self._is_available(torch, device_type)

        if requested is not None:
            base = str(requested).split(":", 1)[0]
            if available(base):
                return str(requested)
            if require_gpu:
                raise GPUNotAvailableError(
                    f"Requested device '{requested}' is not available"
                )
            if base != "cpu":
                _logger.warning(
                    f"Device '{requested}' is not available; using cpu"
                )
            return "cpu"

        for device_type in (*_GPU_DEVICE_TYPES, "cpu"):
            if available(device_type):
                return device_type
        return "cpu"

    def get_torch_device(self):
        try:
            import torch

            return torch.device(self.device_type)
        except ImportError:
            raise ImportError("torch is not installed") from None

    def get_device_info(self) -> dict:
        info = {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "cuda_version": None,
            "gpu_name": None,
            "gpu_memory_gb": None,
        }

        try:
            import torch

            if self.device_type == "cuda":
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
        except Exception as e:
            _logger.debug(f"Error getting device info: {e}")

        return info

    def to_device(self, tensor_or_model):
        try:
            device = self.get_torch_device()
            if hasattr(tensor_or_model, "to"):
                return tensor_or_model.to(device)
            else:
                return tensor_or_model
        except ImportError:
            return tensor_or_model

    def is_gpu_available(self) -> bool:
        return self.device_type in ("cuda", "mps")

    def recommend_batch_size(self) -> int:
        if self.is_gpu_available():
            return 8
        else:
            return 2


class DeviceContext:
    def __init__(self, manager: DeviceManager):
        self.manager = manager

    def __enter__(self):
        return self.manager

    def __exit__(self, *args):
        pass


_device_manager: DeviceManager | None = None
_device_manager_lock = threading.Lock()


def get_device_manager() -> DeviceManager:
    global _device_manager
    with _device_manager_lock:
        if _device_manager is None:
            _device_manager = DeviceManager()
        return _device_manager
