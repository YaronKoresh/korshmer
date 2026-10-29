__all__ = [
    "RangeEQ",
    "generate_bands",
    "save_audio",
    "SmartMastering",
    "benchmark",
    "validation_report",
    "VocalNote",
    "VocalSyllableDetector",
    "detect_vocal_syllables",
    "SpectralAnalyzer",
    "ParametricEQ",
    "GraphicEQ",
    "Compressor",
    "MultiBandCompressor",
    "Limiter",
    "Reverb",
    "Delay",
    "Chorus",
    "Flanger",
    "Phaser",
    "Tremolo",
    "StereoWidener",
    "Distortion",
    "GpuEffects",
    "Saturation",
    "NoiseGate",
    "NoiseReduction",
    "AudioMetering",
    "Analyzer",
    "EnhancedAudioToMIDI",
    "EnsembleTranscription",
    "MusicGenerator",
    "StyleTransfer",
    "SmartMixingAssistant",
    "GenreMixingPresets",
    "BeatDetector",
    "MetricModulationDetector",
    "RubatoDetector",
    "StemClassifier",
    "VocalProcessingAI",
    "GenreMoodDetector",
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
    "StemSession",
    "separate_stems",
    "mix_render_with_clips",
    "find_stem_files",
    "collect_stems",
    "classify_stem_name",
    "build_session_from_processing",
    "normalize_detected_tempo",
]

_IMPORT_MAP = {
    "StemSession": "stem_session",
    "separate_stems": "stem_session",
    "mix_render_with_clips": "stem_session",
    "find_stem_files": "stem_session",
    "collect_stems": "stem_session",
    "classify_stem_name": "stem_session",
    "build_session_from_processing": "stem_session",
    "normalize_detected_tempo": "stem_session",
    "EnhancedAudioToMIDI": "ai_audio_to_midi",
    "EnsembleTranscription": "ai_audio_to_midi",
    "BeatDetector": "beat_detection",
    "MetricModulationDetector": "beat_detection",
    "RubatoDetector": "beat_detection",
    "Distortion": "distortion_noise",
    "GpuEffects": "gpu_effects",
    "get_gpu_effects": "gpu_effects",
    "NoiseGate": "distortion_noise",
    "NoiseReduction": "distortion_noise",
    "Saturation": "distortion_noise",
    "Compressor": "dynamics",
    "Limiter": "dynamics",
    "MultiBandCompressor": "dynamics",
    "Chorus": "effects",
    "Delay": "effects",
    "Flanger": "effects",
    "Phaser": "effects",
    "Tremolo": "effects",
    "StereoWidener": "effects",
    "Reverb": "effects",
    "GraphicEQ": "equalizers",
    "ParametricEQ": "equalizers",
    "generate_bands": "generate_bands",
    "GenreMoodDetector": "genre_mood_detection",
    "Analyzer": "metering",
    "AudioMetering": "metering",
    "MusicGenerator": "music_generation",
    "StyleTransfer": "music_generation",
    "RangeEQ": "range_eq",
    "save_audio": "save_audio",
    "SmartMastering": "smart_mastering",
    "benchmark": "smart_mastering",
    "validation_report": "smart_mastering",
    "GenreMixingPresets": "smart_mixing",
    "SmartMixingAssistant": "smart_mixing",
    "SpectralAnalyzer": "spectral_analysis",
    "StemClassifier": "stem_classification",
    "VocalProcessingAI": "stem_classification",
    "VocalNote": "vocal_processing",
    "VocalSyllableDetector": "vocal_processing",
    "detect_vocal_syllables": "vocal_processing",
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

        module = import_module(f".{module_name}", __name__)
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    elif name == "StemSeparator":
        try:
            from .stem_separator import StemSeparator

            globals()[name] = StemSeparator
            return StemSeparator
        except ImportError:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    from .stem_separator import StemSeparator as StemSeparator

    __all__.append("StemSeparator")
except ImportError:
    pass
