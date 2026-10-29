from unittest.mock import MagicMock


def test_register_midi_processors_registers_all_processors():
    from korshmer.app.components import ComponentRegistry
    from korshmer.core.di_container import DIContainer

    container = DIContainer()
    registry = ComponentRegistry(container)
    window = MagicMock()

    registry.register_midi_processors(window)

    from korshmer.midi_processing.chord_detection import ChordDetector
    from korshmer.midi_processing.drum_map import DrumMapEditor
    from korshmer.midi_processing.instrument_filter import InstrumentFilter
    from korshmer.midi_processing.midi_file_enhancement import MIDIFileEnhancer
    from korshmer.midi_processing.mpe_support import MPEProcessor
    from korshmer.midi_processing.pattern_generation import StepSequencer
    from korshmer.midi_processing.quantization import AdvancedQuantizer
    from korshmer.midi_processing.scale_analysis import ScaleDetector
    from korshmer.midi_processing.score_notation import ScoreRenderer
    from korshmer.midi_processing.voice_leading import VoiceLeadingAnalyzer

    assert isinstance(window.chord_detector, ChordDetector)
    assert isinstance(window.scale_analyzer, ScaleDetector)
    assert isinstance(window.quantizer, AdvancedQuantizer)
    assert isinstance(window.voice_leader, VoiceLeadingAnalyzer)
    assert isinstance(window.pattern_generator, StepSequencer)
    assert isinstance(window.instrument_filter, InstrumentFilter)
    assert isinstance(window.midi_file_enhancer, MIDIFileEnhancer)
    assert isinstance(window.mpe_processor, MPEProcessor)
    assert isinstance(window.drum_mapper, DrumMapEditor)
    assert isinstance(window.score_notation_converter, ScoreRenderer)
