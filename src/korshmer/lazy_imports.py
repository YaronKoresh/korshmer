import threading

_cache = {}
_lock = threading.Lock()


def _lazy_import(key, importer):
    if key not in _cache:
        with _lock:
            if key not in _cache:
                _cache[key] = importer()
    return _cache[key]


def _get_basic_pitch_predict():
    def _import():
        from basic_pitch.inference import predict

        return predict

    return _lazy_import("basic_pitch_predict", _import)


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
