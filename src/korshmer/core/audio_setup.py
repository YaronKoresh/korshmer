import logging

from korshmer.audio_processing import (
    Analyzer,
    AudioMetering,
    BeatDetector,
    Chorus,
    Compressor,
    Delay,
    Distortion,
    EnhancedAudioToMIDI,
    EnsembleTranscription,
    Flanger,
    GenreMixingPresets,
    GenreMoodDetector,
    GraphicEQ,
    Limiter,
    MetricModulationDetector,
    MultiBandCompressor,
    MusicGenerator,
    NoiseGate,
    NoiseReduction,
    ParametricEQ,
    Reverb,
    RubatoDetector,
    Saturation,
    SmartMixingAssistant,
    SpectralAnalyzer,
    StemClassifier,
    StyleTransfer,
    VocalProcessingAI,
)
from korshmer.midi_processing import (
    AdvancedQuantizer,
    Arpeggiator,
    BeamingRules,
    CanonDetector,
    ChordDetector,
    ChordProgressionAnalyzer,
    ChordSymbolGenerator,
    CounterpointChecker,
    CuePointManager,
    CustomScaleBuilder,
    DottedNoteQuantizer,
    DrumFillGenerator,
    DrumHumanizer,
    DrumMapEditor,
    DrumPatternLibrary,
    DrumRollGenerator,
    DynamicsMapper,
    EuclideanRhythmGenerator,
    FugueAnalyzer,
    GraceNoteHandler,
    GrandStaffRenderer,
    GrooveExtractor,
    HiHatVariationGenerator,
    MIDI2Support,
    MIDIBatchProcessor,
    MidiCCAutomation,
    MIDIFileEnhancer,
    MIDIFormatConverter,
    MPEChannelAllocator,
    MPEControllerMapper,
    MPEGlissandoGenerator,
    MPEProcessor,
    MPEToStandardMIDIConverter,
    MPEVisualizer,
    MusicXMLExporter,
    PatternChainer,
    PolyrhythmGenerator,
    RatchetGenerator,
    ScaleDegreeHighlighter,
    ScaleDetector,
    ScoreRenderer,
    SlurTieHandler,
    StepSequencer,
    SysExHandler,
    TripletQuantizer,
    TupletDetector,
    VoiceLeadingAnalyzer,
)


def _init_audio_processors(window):
    window.spectral_analyzer = SpectralAnalyzer()
    window.parametric_eq = ParametricEQ()
    window.graphic_eq = GraphicEQ()
    window.compressor = Compressor()
    window.multiband_compressor = MultiBandCompressor()
    window.limiter = Limiter()
    window.reverb = Reverb()
    window.delay_fx = Delay()
    window.chorus = Chorus()
    window.flanger = Flanger()
    window.distortion = Distortion()
    window.saturation = Saturation()
    window.noise_gate = NoiseGate()
    window.noise_reduction = NoiseReduction()
    window.audio_metering = AudioMetering()
    window.analyzer = Analyzer()

    from korshmer.exceptions import AudioToMidiError

    try:
        window.enhanced_audio_to_midi = EnhancedAudioToMIDI()
    except AudioToMidiError as e:
        logging.warning(f"Audio-to-MIDI disabled: {e}")
        window.enhanced_audio_to_midi = None
    try:
        window.ensemble_transcription = EnsembleTranscription()
    except AudioToMidiError as e:
        logging.warning(f"Ensemble transcription disabled: {e}")
        window.ensemble_transcription = None
    window.music_generator = MusicGenerator()
    window.style_transfer = StyleTransfer()
    window.smart_mixing_assistant = SmartMixingAssistant()
    window.genre_mixing_presets = GenreMixingPresets()
    window.beat_detector = BeatDetector()
    window.metric_modulation_detector = MetricModulationDetector()
    window.rubato_detector = RubatoDetector()
    window.stem_classifier = StemClassifier()
    window.vocal_processing_ai = VocalProcessingAI()
    window.genre_mood_detector = GenreMoodDetector()


def _init_midi_processors(window):
    window.chord_detector = ChordDetector()
    window.chord_progression_analyzer = ChordProgressionAnalyzer()
    window.scale_detector = ScaleDetector()
    window.custom_scale_builder = CustomScaleBuilder()
    window.scale_degree_highlighter = ScaleDegreeHighlighter()
    window.advanced_quantizer = AdvancedQuantizer()
    window.groove_extractor = GrooveExtractor()
    window.triplet_quantizer = TripletQuantizer()
    window.dotted_note_quantizer = DottedNoteQuantizer()
    window.arpeggiator = Arpeggiator()
    window.step_sequencer = StepSequencer()
    window.ratchet_generator = RatchetGenerator()
    window.euclidean_rhythm_generator = EuclideanRhythmGenerator()
    window.polyrhythm_generator = PolyrhythmGenerator()
    window.pattern_chainer = PatternChainer()
    window.midi_cc_automation = MidiCCAutomation()
    window.voice_leading_analyzer = VoiceLeadingAnalyzer()
    window.counterpoint_checker = CounterpointChecker()
    window.fugue_analyzer = FugueAnalyzer()
    window.canon_detector = CanonDetector()
    window.mpe_channel_allocator = MPEChannelAllocator()
    window.mpe_visualizer = MPEVisualizer()
    window.mpe_to_standard_converter = MPEToStandardMIDIConverter()
    window.mpe_controller_mapper = MPEControllerMapper()
    window.mpe_processor = MPEProcessor.from_controller_mapper(
        window.mpe_controller_mapper
    )
    window.mpe_glissando_generator = MPEGlissandoGenerator()
    window.midi_file_enhancer = MIDIFileEnhancer()
    window.midi_format_converter = MIDIFormatConverter()
    window.sysex_handler = SysExHandler()
    window.midi2_support = MIDI2Support()
    window.midi_batch_processor = MIDIBatchProcessor()
    window.cue_point_manager = CuePointManager()
    window.drum_map_editor = DrumMapEditor()
    window.drum_pattern_library = DrumPatternLibrary()
    window.drum_fill_generator = DrumFillGenerator()
    window.hihat_variation_generator = HiHatVariationGenerator()
    window.drum_humanizer = DrumHumanizer()
    window.drum_roll_generator = DrumRollGenerator()
    window.score_renderer = ScoreRenderer()
    window.beaming_rules = BeamingRules()
    window.tuplet_detector = TupletDetector()
    window.grace_note_handler = GraceNoteHandler()
    window.dynamics_mapper = DynamicsMapper()
    window.slur_tie_handler = SlurTieHandler()
    window.chord_symbol_generator = ChordSymbolGenerator()
    window.grand_staff_renderer = GrandStaffRenderer()
    window.musicxml_exporter = MusicXMLExporter()


def init_audio_and_midi_processors(window):
    _init_audio_processors(window)
    _init_midi_processors(window)

    logging.info("All audio and MIDI processing components initialized")
