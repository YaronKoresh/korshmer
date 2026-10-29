import json
import logging
import os
from pathlib import Path

from korshmer.constants import PRESETS_FILE

DEFAULT_PRESET = {
    "octave": "Original",
    "root": "Auto",
    "scale": "Auto (Detect)",
    "grid": "1/8",
    "quantize": 100,
    "swing": 0,
    "vel_mode": "Fixed",
    "vel_val": 100,
    "stems": ["drums", "bass", "other", "guitar", "piano", "vocals"],
    "export": "Separate",
    "cc": True,
    "gm": True,
    "wav32": True,
}


RESERVED_PRESET_NAME = "Default"


def validate_preset_name(name):
    if not isinstance(name, str):
        return None
    cleaned = name.strip()
    if not cleaned or cleaned.casefold() == RESERVED_PRESET_NAME.casefold():
        return None
    return cleaned


class PresetService:
    @staticmethod
    def load_all():
        defaults = {RESERVED_PRESET_NAME: dict(DEFAULT_PRESET)}
        if os.path.exists(PRESETS_FILE):
            try:
                with open(PRESETS_FILE, encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    return defaults
                for name, data in loaded.items():
                    if not isinstance(data, dict):
                        logging.debug(f"Ignoring invalid preset data: {name!r}")
                        continue
                    if validate_preset_name(name) is None:
                        logging.debug(f"Ignoring invalid preset name: {name!r}")
                        continue
                    defaults[name] = data
            except (OSError, ValueError, AttributeError) as e:
                logging.debug(f"Could not load presets file: {e}")
        return defaults

    @staticmethod
    def save_all(presets):
        try:
            target = Path(PRESETS_FILE)
            tmp = target.parent / f".{target.name}.tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(presets, f, indent=4, ensure_ascii=False)
                os.replace(tmp, target)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        except Exception as e:
            logging.warning(f"Failed to save presets: {e}")

    @staticmethod
    def get_default():
        return dict(DEFAULT_PRESET)

    @staticmethod
    def build_preset_data(
        octave,
        root,
        scale,
        grid,
        quantize,
        swing,
        vel_mode,
        vel_val,
        export_mode,
        cc,
        gm,
        wav32,
        stems,
    ):
        return {
            "octave": octave,
            "root": root,
            "scale": scale,
            "grid": grid,
            "quantize": quantize,
            "swing": swing,
            "vel_mode": vel_mode,
            "vel_val": vel_val,
            "export": export_mode,
            "cc": cc,
            "gm": gm,
            "wav32": wav32,
            "stems": stems,
        }

    @staticmethod
    def normalize_velocity_mode(mode_text):
        mapping = {
            "Original": "Original",
            "Fixed": "Fixed",
            "Compressed": "Compressed",
        }
        return mapping.get(mode_text, "Fixed")
