try:
    import torch
except ImportError:
    torch = None
import logging
import os
import threading

import numpy as np
import soundfile as sf

from korshmer.audio_processing.stem_session import validate_stem_model_name
from korshmer.processing.device import get_device_manager
from korshmer.utils.perf_metrics import measure

_logger = logging.getLogger(__name__)

try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
except ImportError:
    get_model = None
    apply_model = None


DEFAULT_SAMPLE_RATE = 44100


class StemSeparator:
    def __init__(self, model_name="htdemucs", device=None, require_gpu=False):
        if torch is None or get_model is None or apply_model is None:
            raise ImportError(
                "StemSeparator requires the 'torch' and 'demucs' packages to be installed."
            )
        self._thread_state = threading.local()
        self.device = get_device_manager().select_device(
            device, require_gpu=require_gpu
        )
        validate_stem_model_name(model_name)
        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model_name = model_name
        self.last_sample_rate = DEFAULT_SAMPLE_RATE

    @property
    def last_sample_rate(self) -> int:
        return getattr(self._thread_state, "sample_rate", DEFAULT_SAMPLE_RATE)

    @last_sample_rate.setter
    def last_sample_rate(self, value: int) -> None:
        self._thread_state.sample_rate = value

    @measure("audio.stem_separation")
    def separate(
        self, audio, sample_rate=44100, stems=4, progress_callback=None
    ) -> dict[str, np.ndarray]:
        self.last_sample_rate = sample_rate

        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        audio_tensor = torch.from_numpy(audio).float()

        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        if progress_callback:
            progress_callback(10)

        try:
            sources = self._apply(audio_tensor)
        except RuntimeError as e:
            if self.device == "cpu":
                raise
            _logger.warning(
                f"Separation on '{self.device}' failed ({e}); retrying on cpu"
            )
            self.device = "cpu"
            self.model.to("cpu")
            sources = self._apply(audio_tensor)

        if progress_callback:
            progress_callback(80)

        stems_dict = {}
        stem_names = ["drums", "bass", "other", "vocals"]

        for i, name in enumerate(stem_names):
            if i < sources.shape[1]:
                stems_dict[name] = sources[0, i].cpu().numpy()

        if progress_callback:
            progress_callback(100)

        return stems_dict

    def _apply(self, audio_tensor):
        with torch.no_grad():
            return apply_model(
                self.model,
                audio_tensor.to(self.device),
                device=self.device,
            )

    def set_model(self, model_name):
        validate_stem_model_name(model_name)
        self.model_name = model_name
        self.model = get_model(model_name)
        self.model.to(self.device)

    def export_stems(self, stems_dict, output_dir, format="wav") -> None:
        os.makedirs(output_dir, exist_ok=True)

        supported_formats = {"wav", "flac", "mp3", "ogg"}
        normalized_format = format.lower()
        if normalized_format not in supported_formats:
            raise ValueError(
                f"Unsupported audio format '{format}'. "
                f"Supported formats are: {', '.join(sorted(supported_formats))}."
            )

        for stem_name, audio_data in stems_dict.items():
            output_path = os.path.join(
                output_dir, f"{stem_name}.{normalized_format}"
            )
            sf.write(output_path, audio_data.T, self.last_sample_rate)
