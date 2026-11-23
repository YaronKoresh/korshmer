import threading
from typing import Any

_cache: dict[str, Any] = {}
_lock = threading.Lock()


def _lazy_import(key, importer):
    if key not in _cache:
        with _lock:
            if key not in _cache:
                _cache[key] = importer()
    return _cache[key]


def _onnx_model_path():
    import importlib.util

    if importlib.util.find_spec("onnxruntime") is None:
        return None
    try:
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path

        path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    except (ImportError, AttributeError):
        return None
    return path if path.exists() else None


def load_basic_pitch_predict():
    import functools

    from basic_pitch.inference import predict

    model = _onnx_model_path()
    if model is None:
        return predict

    @functools.wraps(predict)
    def predict_with_onnx(audio_path, *args, **kwargs):
        if not args:
            kwargs.setdefault("model_or_model_path", model)
        return predict(audio_path, *args, **kwargs)

    return predict_with_onnx


def _get_basic_pitch_predict():
    return _lazy_import("basic_pitch_predict", load_basic_pitch_predict)


def _get_demucs_modules():
    def _import():
        import demucs.apply
        import demucs.audio
        import demucs.separate

        return demucs.apply, demucs.audio, demucs.separate

    return _lazy_import("demucs_modules", _import)


def _get_torch_modules():
    def _import():
        import torch
        import torchaudio

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return torch, torchaudio

    return _lazy_import("torch_modules", _import)


def _get_sklearn_kmeans():
    def _import():
        from sklearn.cluster import KMeans

        return KMeans

    return _lazy_import("sklearn_kmeans", _import)
