def test_audio_available_flag_is_true():
    from korshmer.app import components

    assert components._AUDIO_AVAILABLE is True


def test_import_paths_resolve():
    from korshmer.audio_processing.beat_detection import (
        MetricModulationDetector,
        RubatoDetector,
    )
    from korshmer.audio_processing.stem_classification import (
        VocalProcessingAI,
    )

    assert MetricModulationDetector is not None
    assert RubatoDetector is not None
    assert VocalProcessingAI is not None


def test_component_registry_audio_processors_list_references_importable_classes():
    from korshmer.app import components

    assert components._AUDIO_AVAILABLE is True

    from korshmer.audio_processing.beat_detection import (
        MetricModulationDetector,
        RubatoDetector,
    )
    from korshmer.audio_processing.stem_classification import (
        VocalProcessingAI,
    )

    assert components.MetricModulationDetector is MetricModulationDetector
    assert components.RubatoDetector is RubatoDetector
    assert components.VocalProcessingAI is VocalProcessingAI
