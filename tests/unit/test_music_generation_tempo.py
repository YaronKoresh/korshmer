import pretty_midi

from korshmer.audio_processing.music_generation import (
    MusicGenerator,
    StyleTransfer,
)


def _make_midi_no_tempo():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(
        pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5)
    )
    midi.instruments.append(inst)
    return midi


def test_create_variation_no_tempo():
    gen = MusicGenerator()
    midi = _make_midi_no_tempo()
    result = gen.create_variation(midi, variation_type="dynamic", amount=0.3)
    assert len(result.instruments) == 1
    assert len(result.instruments[0].notes) == 1


def test_expand_motif_no_tempo():
    gen = MusicGenerator()
    midi = _make_midi_no_tempo()
    result = gen.expand_motif(
        midi, expansion_bars=2, development_type="sequence"
    )
    assert len(result.instruments) == 1
    assert len(result.instruments[0].notes) == 2


def test_apply_style_no_tempo():
    transfer = StyleTransfer()
    midi = _make_midi_no_tempo()
    features = transfer.extract_style_features(midi)
    result = transfer.apply_style(midi, features)
    assert len(result.instruments) == 1
    assert len(result.instruments[0].notes) == 1
