from korshmer.core.di_container import DIContainer as Container

try:
    from korshmer.audio_processing.ai_audio_to_midi import (
        EnhancedAudioToMIDI,
        EnsembleTranscription,
    )
    from korshmer.audio_processing.beat_detection import (
        BeatDetector,
        MetricModulationDetector,
        RubatoDetector,
    )
    from korshmer.audio_processing.distortion_noise import (
        Distortion,
        NoiseGate,
        NoiseReduction,
        Saturation,
    )
    from korshmer.audio_processing.dynamics import (
        Compressor,
        Limiter,
        MultiBandCompressor,
    )
    from korshmer.audio_processing.effects import Chorus, Delay, Flanger, Reverb
    from korshmer.audio_processing.equalizers import GraphicEQ, ParametricEQ
    from korshmer.audio_processing.genre_mood_detection import GenreMoodDetector
    from korshmer.audio_processing.metering import Analyzer, AudioMetering
    from korshmer.audio_processing.music_generation import MusicGenerator
    from korshmer.audio_processing.smart_mixing import (
        GenreMixingPresets,
        SmartMixingAssistant,
    )
    from korshmer.audio_processing.spectral_analysis import SpectralAnalyzer
    from korshmer.audio_processing.stem_classification import (
        StemClassifier,
        VocalProcessingAI,
    )

    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

try:
    from korshmer.project_management import (
        AnalyticsManager,
        AssetManager,
        CloudSyncManager,
        CollaborationManager,
        ExportManager,
        ImportManager,
        ProjectManager,
        VersionControlManager,
    )

    _PM_AVAILABLE = True
except ImportError:
    _PM_AVAILABLE = False

try:
    from korshmer.ui.managers import (
        AccessibilityManager,
        ContextMenuManager,
        KeyboardShortcutManager,
        PerformanceMonitor,
        ThemeManager,
        TooltipManager,
        TutorialManager,
        UndoRedoManager,
        WorkspaceManager,
    )

    _UI_AVAILABLE = True
except ImportError:
    _UI_AVAILABLE = False


class ComponentRegistry:
    def __init__(self, container: Container):
        self.container = container

    def register_audio_processors(self, window):
        if not _AUDIO_AVAILABLE:
            return
        audio_processors = [
            (SpectralAnalyzer, "spectral_analyzer"),
            (ParametricEQ, "parametric_eq"),
            (GraphicEQ, "graphic_eq"),
            (Compressor, "compressor"),
            (MultiBandCompressor, "multiband_compressor"),
            (Limiter, "limiter"),
            (Reverb, "reverb"),
            (Delay, "delay"),
            (Chorus, "chorus"),
            (Flanger, "flanger"),
            (Distortion, "distortion"),
            (Saturation, "saturation"),
            (NoiseGate, "noise_gate"),
            (NoiseReduction, "noise_reduction"),
            (AudioMetering, "audio_metering"),
            (Analyzer, "analyzer"),
            (EnhancedAudioToMIDI, "enhanced_audio_to_midi"),
            (EnsembleTranscription, "ensemble_transcription"),
            (MusicGenerator, "music_generator"),
            (SmartMixingAssistant, "smart_mixing_assistant"),
            (GenreMixingPresets, "genre_mixing_presets"),
            (BeatDetector, "beat_detector"),
            (MetricModulationDetector, "metric_modulation_detector"),
            (RubatoDetector, "rubato_detector"),
            (StemClassifier, "stem_classifier"),
            (VocalProcessingAI, "vocal_processing_ai"),
            (GenreMoodDetector, "genre_mood_detector"),
        ]
        for processor_class, name in audio_processors:
            instance = processor_class()
            setattr(window, name, instance)
            self.container.register_instance(name, instance)

    def register_midi_processors(self, window):
        try:
            from korshmer.midi_processing.chord_detection import ChordDetector
            from korshmer.midi_processing.drum_map import DrumMapEditor
            from korshmer.midi_processing.instrument_filter import (
                InstrumentFilter,
            )
            from korshmer.midi_processing.midi_file_enhancement import (
                MIDIFileEnhancer,
            )
            from korshmer.midi_processing.mpe_support import MPEProcessor
            from korshmer.midi_processing.pattern_generation import (
                StepSequencer,
            )
            from korshmer.midi_processing.quantization import AdvancedQuantizer
            from korshmer.midi_processing.scale_analysis import ScaleDetector
            from korshmer.midi_processing.score_notation import (
                ScoreRenderer,
            )
            from korshmer.midi_processing.voice_leading import (
                VoiceLeadingAnalyzer,
            )
        except ImportError:
            return
        midi_processors = [
            (ChordDetector, "chord_detector"),
            (ScaleDetector, "scale_analyzer"),
            (AdvancedQuantizer, "quantizer"),
            (VoiceLeadingAnalyzer, "voice_leader"),
            (StepSequencer, "pattern_generator"),
            (InstrumentFilter, "instrument_filter"),
            (MIDIFileEnhancer, "midi_file_enhancer"),
            (MPEProcessor, "mpe_processor"),
            (DrumMapEditor, "drum_mapper"),
            (ScoreRenderer, "score_notation_converter"),
        ]
        for processor_class, name in midi_processors:
            instance = processor_class()
            setattr(window, name, instance)
            self.container.register_instance(name, instance)

    def register_project_managers(self, window, base_dir):
        if not _PM_AVAILABLE:
            return
        managers = [
            (ProjectManager, "project_manager", {"base_dir": base_dir}),
            (AssetManager, "asset_manager", {"base_dir": base_dir}),
            (ExportManager, "export_manager", {"base_dir": base_dir}),
            (ImportManager, "import_manager", {"base_dir": base_dir}),
            (
                VersionControlManager,
                "version_control_manager",
                {"base_dir": base_dir},
            ),
            (CloudSyncManager, "cloud_sync_manager", {"base_dir": base_dir}),
            (
                CollaborationManager,
                "collaboration_manager",
                {"base_dir": base_dir},
            ),
            (AnalyticsManager, "analytics_manager", {"base_dir": base_dir}),
        ]
        for manager_class, name, kwargs in managers:
            instance = manager_class(**kwargs) if kwargs else manager_class()
            setattr(window, name, instance)
            self.container.register_instance(name, instance)

    def register_ui_managers(self, window, data_dir):
        if not _UI_AVAILABLE:
            return
        ui_managers = [
            (
                KeyboardShortcutManager,
                "keyboard_shortcut_manager",
                {"data_dir": data_dir},
            ),
            (ThemeManager, "theme_manager", {"data_dir": data_dir}),
            (UndoRedoManager, "undo_redo_manager", {}),
            (WorkspaceManager, "workspace_manager", {"data_dir": data_dir}),
            (PerformanceMonitor, "performance_monitor", {}),
            (ContextMenuManager, "context_menu_manager", {}),
            (TooltipManager, "tooltip_manager", {}),
            (TutorialManager, "tutorial_manager", {"parent_window": window}),
            (AccessibilityManager, "accessibility_manager", {}),
        ]
        for manager_class, name, kwargs in ui_managers:
            instance = manager_class(**kwargs) if kwargs else manager_class()
            setattr(window, name, instance)
            self.container.register_instance(name, instance)
