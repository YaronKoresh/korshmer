from korshmer.midi_processing.scale_analysis import ScaleDetector, ScaleType

A_MINOR = [57, 59, 60, 62, 64, 65, 67, 69]
C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]
A_MINOR_PENTATONIC = [57, 60, 62, 64, 67]


def _names(notes):
    return [
        s.get_name() for s in ScaleDetector().detect_scale_from_notes(notes)
    ]


def test_minor_melody_ranks_its_own_tonic_first():
    assert _names(A_MINOR)[0] == "A Natural Minor"


def test_major_melody_ranks_its_own_tonic_first():
    assert _names(C_MAJOR)[0] == "C Major"


def test_minor_pentatonic_melody_ranks_minor_pentatonic_first():
    assert _names(A_MINOR_PENTATONIC)[0] == "A Pentatonic Minor"


def test_results_have_no_duplicate_scales():
    for notes in (A_MINOR, C_MAJOR, A_MINOR_PENTATONIC):
        scales = ScaleDetector().detect_scale_from_notes(notes)
        keys = [(s.root, s.intervals) for s in scales]
        assert len(keys) == len(set(keys))
        assert all(s.scale_type != ScaleType.AEOLIAN for s in scales)


def test_suggest_scale_uses_tonic_ranking():
    scale = ScaleDetector().suggest_scale(A_MINOR)
    assert scale.root == 9
    assert scale.scale_type == ScaleType.NATURAL_MINOR


def test_too_few_notes_returns_empty():
    assert ScaleDetector().detect_scale_from_notes([60, 62]) == []
