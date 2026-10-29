__all__ = [
    "save_normalized_audio",
    "cleanup_temp_files",
    "trim_audio_silence",
    "analyse_tempo",
    "run_demucs",
    "pitch_shift_file",
    "estimate_meter",
    "apply_noise_gate",
    "apply_normalize",
    "apply_filter",
    "apply_reverse_audio",
    "apply_mono_mix",
    "apply_transient_shaper",
    "slice_audio_onsets",
]

_IMPORT_MAP = {
    "analyse_tempo": "core_analysis",
    "estimate_meter": "core_analysis",
    "pitch_shift_file": "core_analysis",
    "run_demucs": "core_analysis",
    "trim_audio_silence": "core_analysis",
    "apply_filter": "basic_effects",
    "apply_mono_mix": "basic_effects",
    "apply_noise_gate": "basic_effects",
    "apply_normalize": "basic_effects",
    "apply_reverse_audio": "basic_effects",
    "apply_transient_shaper": "basic_effects",
    "slice_audio_onsets": "basic_effects",
    "cleanup_temp_files": "audio_io",
    "save_normalized_audio": "audio_io",
}


def __getattr__(name):
    if name in _IMPORT_MAP:
        module_name = _IMPORT_MAP[name]
        from importlib import import_module

        module = import_module(f".{module_name}", "korshmer.audio_processing")
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
