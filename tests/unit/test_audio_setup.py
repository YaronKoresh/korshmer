import logging
from types import SimpleNamespace

import pytest

from korshmer.core import audio_setup
from korshmer.exceptions import AudioToMidiError

PROCESSOR_NAMES = [
    "spectral_analyzer",
    "parametric_eq",
    "graphic_eq",
    "compressor",
    "multiband_compressor",
    "limiter",
    "reverb",
    "delay_fx",
    "chorus",
    "flanger",
    "distortion",
    "saturation",
    "noise_gate",
    "noise_reduction",
    "audio_metering",
    "analyzer",
    "enhanced_audio_to_midi",
    "ensemble_transcription",
    "music_generator",
    "style_transfer",
    "smart_mixing_assistant",
    "genre_mixing_presets",
    "beat_detector",
    "metric_modulation_detector",
    "rubato_detector",
    "stem_classifier",
    "vocal_processing_ai",
    "genre_mood_detector",
    "chord_detector",
    "chord_progression_analyzer",
    "scale_detector",
    "custom_scale_builder",
    "scale_degree_highlighter",
    "advanced_quantizer",
    "groove_extractor",
    "triplet_quantizer",
    "dotted_note_quantizer",
    "arpeggiator",
    "step_sequencer",
    "ratchet_generator",
    "euclidean_rhythm_generator",
    "polyrhythm_generator",
    "pattern_chainer",
    "midi_cc_automation",
    "voice_leading_analyzer",
    "counterpoint_checker",
    "fugue_analyzer",
    "canon_detector",
    "mpe_processor",
    "mpe_channel_allocator",
    "mpe_visualizer",
    "mpe_to_standard_converter",
    "mpe_controller_mapper",
    "mpe_glissando_generator",
    "midi_file_enhancer",
    "midi_format_converter",
    "sysex_handler",
    "midi2_support",
    "midi_batch_processor",
    "cue_point_manager",
    "drum_map_editor",
    "drum_pattern_library",
    "drum_fill_generator",
    "hihat_variation_generator",
    "drum_humanizer",
    "drum_roll_generator",
    "score_renderer",
    "beaming_rules",
    "tuplet_detector",
    "grace_note_handler",
    "dynamics_mapper",
    "slur_tie_handler",
    "chord_symbol_generator",
    "grand_staff_renderer",
    "musicxml_exporter",
]


def test_processor_names_are_unique_and_complete():
    assert len(PROCESSOR_NAMES) == 75
    assert len(set(PROCESSOR_NAMES)) == 75


def test_init_sets_all_processors_on_window(caplog):
    window = SimpleNamespace()
    with caplog.at_level(logging.INFO):
        audio_setup.init_audio_and_midi_processors(window)
    for name in PROCESSOR_NAMES:
        assert hasattr(window, name), name
    assert window.delay_fx is not None
    assert "All audio and MIDI processing components initialized" in caplog.text


@pytest.mark.parametrize(
    ("class_name", "attr", "message"),
    [
        (
            "EnhancedAudioToMIDI",
            "enhanced_audio_to_midi",
            "Audio-to-MIDI disabled: ",
        ),
        (
            "EnsembleTranscription",
            "ensemble_transcription",
            "Ensemble transcription disabled: ",
        ),
    ],
)
def test_audio_to_midi_error_is_tolerated(
    monkeypatch, caplog, class_name, attr, message
):
    def _raise():
        raise AudioToMidiError("basic-pitch missing")

    monkeypatch.setattr(audio_setup, class_name, _raise)
    window = SimpleNamespace()
    with caplog.at_level(logging.WARNING):
        audio_setup.init_audio_and_midi_processors(window)
    assert getattr(window, attr) is None
    assert message + "basic-pitch missing" in caplog.text
    for name in PROCESSOR_NAMES:
        assert hasattr(window, name), name
