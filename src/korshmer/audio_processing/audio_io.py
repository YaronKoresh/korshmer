import atexit
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from korshmer.constants import DATA_DIR, TEMP_DIR


def _save_normalized_audio_internal(path, audio_data, sample_rate):
    start_time = time.time()
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        audio_data = np.asarray(audio_data, dtype=np.float32)
        if audio_data.size == 0:
            raise ValueError("audio has no samples")
        if not np.all(np.isfinite(audio_data)):
            logging.warning("Replacing non-finite audio samples with zero")
            audio_data = np.nan_to_num(
                audio_data, nan=0.0, posinf=0.0, neginf=0.0
            )
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = (audio_data / max_val) * 0.95
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf.write(str(path), audio_data, sample_rate)
        duration = time.time() - start_time
        logging.debug(f"Saved audio to {path} in {duration:.3f}s")
    except (OSError, ValueError, RuntimeError) as e:
        logging.error(f"Failed to save audio to {path}: {e}")
        raise


def save_normalized_audio(
    audio_or_path, sample_rate_or_audio, output_path_or_sr=None
):

    if isinstance(audio_or_path, (str, Path)):
        path = audio_or_path
        audio_data = sample_rate_or_audio
        sample_rate = output_path_or_sr
    else:
        audio_data = audio_or_path
        sample_rate = sample_rate_or_audio
        path = output_path_or_sr

    _save_normalized_audio_internal(path, audio_data, sample_rate)


def cleanup_temp_files():
    files_to_clean = ["temp_preview.wav", "temp_spectrum.png"]
    search_dirs = [DATA_DIR, TEMP_DIR]
    for d in search_dirs:
        for f in files_to_clean:
            try:
                p = Path(d) / f
                if p.exists():
                    p.unlink()
                    logging.debug(f"Cleaned up temp file: {p}")
            except Exception as e:
                logging.debug(f"Could not clean up {p}: {e}")


_cleanup_registered = False


def register_temp_cleanup():
    global _cleanup_registered
    if _cleanup_registered:
        return
    atexit.register(cleanup_temp_files)
    _cleanup_registered = True


register_temp_cleanup()
