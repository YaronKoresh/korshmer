import os
import shutil
from pathlib import Path

import demucs.pretrained
from basic_pitch.inference import ICASSP_2022_MODEL_PATH


def normalize_path(p: str) -> str:
    return p.rstrip("/").replace(os.path.sep, "/")


def full_path(*p) -> str:
    return normalize_path(
        str(Path(os.path.join(*[str(_p).strip() for _p in p])).resolve())
    )


def copy_transcription_model(model_path, target_dir):
    model = Path(str(model_path))
    target = Path(str(target_dir))
    target.mkdir(parents=True, exist_ok=True)
    if model.is_dir():
        shutil.copytree(model, target / model.name, dirs_exist_ok=True)
    elif model.is_file():
        shutil.copy(model, target / model.name)
    else:
        raise FileNotFoundError(f"Transcription model not found: {model}")
    return target / model.name


def download_models():
    current_dir = os.path.dirname(__file__)
    models_dir = full_path(current_dir, "..", "..", "models")

    os.makedirs(models_dir, exist_ok=True)
    print(f"Downloading models to: {models_dir}")

    print("Fetching Demucs model...")
    os.environ["TORCH_HOME"] = models_dir
    demucs.pretrained.get_model("htdemucs_6s")

    print("Fetching vocal separation model...")
    try:
        from korshmer.audio_processing.mdx_vocals import ensure_mdx_model

        fetched = ensure_mdx_model(models_dir) is not None
    except ImportError:
        fetched = False
    if not fetched:
        print("Vocal model not downloaded; it is fetched when first used.")

    print(f"Basic Pitch model path found at: {ICASSP_2022_MODEL_PATH}")
    copy_transcription_model(
        ICASSP_2022_MODEL_PATH, full_path(models_dir, "hub", "checkpoints")
    )
    print("All models prepared properly.")


if __name__ == "__main__":
    download_models()
