import pytest

from korshmer.audio_processing.music_generation import MusicGenerator

NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]
PROGRESSION = [[0, 4, 7], [5, 9, 0]]


@pytest.fixture
def generator():
    return MusicGenerator()


def _calls(generator):
    return [
        lambda **kw: generator.generate_melody_from_chords(PROGRESSION, **kw),
        lambda **kw: generator.generate_bass_line(PROGRESSION, **kw),
        lambda **kw: generator.generate_drum_pattern(**kw),
    ]


@pytest.mark.parametrize("value", NOT_NUMBERS + [0, -5, 19.9, 300.1, 1e9])
def test_tempo_rejected(generator, value):
    for call in _calls(generator):
        with pytest.raises(ValueError, match="Tempo"):
            call(tempo=value)


@pytest.mark.parametrize(
    "value", NOT_NUMBERS + [0, -5, 513, 1.5, 1e9, float("nan")]
)
def test_duration_bars_rejected(generator, value):
    for call in _calls(generator):
        with pytest.raises(ValueError, match="Duration bars"):
            call(duration_bars=value)


@pytest.mark.parametrize("value", NOT_NUMBERS + [-1, 128, 60.5])
def test_key_rejected(generator, value):
    with pytest.raises(ValueError, match="Key"):
        generator.generate_melody_from_chords(PROGRESSION, key=value)
    with pytest.raises(ValueError, match="Key"):
        generator.generate_bass_line(PROGRESSION, key=value)


@pytest.mark.parametrize("value", NOT_NUMBERS + [-0.1, 1.1, 1e9])
def test_intensity_rejected(generator, value):
    with pytest.raises(ValueError, match="Intensity"):
        generator.generate_drum_pattern(intensity=value)


def test_unknown_names_rejected(generator):
    with pytest.raises(ValueError, match="Unknown scale"):
        generator.generate_melody_from_chords(PROGRESSION, scale="foo")
    with pytest.raises(ValueError, match="Unknown rhythm pattern"):
        generator.generate_bass_line(PROGRESSION, rhythm_pattern="foo")
    with pytest.raises(ValueError, match="Unknown genre"):
        generator.generate_drum_pattern(genre="foo")


def test_valid_edges_still_generate(generator):
    for tempo in (20, 300):
        midi = generator.generate_drum_pattern("jazz", tempo, 1, 1.0)
        assert midi.instruments[0].notes
    assert (
        generator.generate_bass_line(PROGRESSION, 0, 120.0, 1, "pumping")
        .instruments[0]
        .notes
    )
    assert (
        generator.generate_melody_from_chords(
            PROGRESSION, 127, "minor", 120.0, 512
        )
        .instruments[0]
        .notes
    )
