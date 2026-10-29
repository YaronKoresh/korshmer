import os
import shutil
from pathlib import Path

import demucs.pretrained
from basic_pitch.inference import ICASSP_2022_MODEL_PATH


def normalize_path(p) -> str:
    return p.rstrip("/").replace(os.path.sep, "/")


def full_path(*p) -> str:
    return normalize_path(
        str(Path(os.path.join(*[str(_p).strip() for _p in p])).resolve())
    )


def exist(*p) -> bool:
    return Path(os.path.join(*[str(_p).strip() for _p in p])).exists()


def is_directory(*p):
    return Path(os.path.join(*[str(_p).strip() for _p in p])).is_dir()


def copy(src, dst):
    if is_directory(full_path(src)):
        shutil.copytree(
            str(src),
            str(dst),
            symlinks=False,
            ignore_dangling_symlinks=True,
        )
    else:
        shutil.copy(str(src), str(dst))


def download_models():
    current_dir = os.path.dirname(__file__)
    models_dir = full_path(current_dir, "..", "..", "models")

    if exist(models_dir):
        return

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
    bp_outpath = full_path(models_dir, "hub", "checkpoints")
    copy(
        full_path(ICASSP_2022_MODEL_PATH, "saved_model.pb"),
        full_path(bp_outpath, "saved_model.pb"),
    )
    copy(
        full_path(ICASSP_2022_MODEL_PATH, "variables"),
        full_path(bp_outpath, "variables"),
    )
    print("All models prepared properly.")


if __name__ == "__main__":
    download_models()
