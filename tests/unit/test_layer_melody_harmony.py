import pretty_midi
import pytest

from korshmer.midi_generation import ComposeOptions, compose, song_match

BPM = 100.0
BAR = 240.0 / BPM


def _options(**changes):
    values = {
        "root": "D",
        "scale": "Minor",
        "bpm": BPM,
        "minutes": 0.5,
        "style": "Pop",
        "structure": "Verse / Chorus",
        "mood": "Auto",
        "seed": 3,
        "roles": ("drums", "bass", "chords", "pad", "melody"),
        "texture_batches": 0,
        "mode": "New song",
        "match": "Strong",
        "existing_tracks": 0,
    }
    values.update(changes)
    return ComposeOptions(**values)


def _song():
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    pm.instruments = compose(_options()).instruments
    return pm


def _lead(pm):
    return next(i for i in pm.instruments if i.name == "Melody")


def _layer(pm, seed, roles=("melody",)):
    return compose(
        _options(mode="Layer over editor", seed=seed, roles=roles),
        existing=pm,
    )


def _clash_share(layer, lead):
    clash = total = 0
    for note in layer.notes:
        for other in lead.notes:
            if other.start < note.end and note.start < other.end:
                total += 1
                if abs(other.pitch - note.pitch) in (0, 1, 2, 6, 10, 11):
                    clash += 1
                break
    return clash / max(1, total), total


def test_find_lead_picks_the_single_voice_high_line():
    pm = _song()
    assert song_match.find_lead(pm) == _lead(pm).notes


def test_find_lead_is_empty_for_a_song_of_chords_only():
    pm = pretty_midi.PrettyMIDI()
    chords = pretty_midi.Instrument(0)
    for bar in range(12):
        for pitch in (48, 52, 55):
            chords.notes.append(
                pretty_midi.Note(90, pitch, bar * BAR, (bar + 1) * BAR)
            )
    pm.instruments.append(chords)
    assert song_match.find_lead(pm) == []
    assert song_match.find_lead(None) == []


@pytest.mark.parametrize("seed", [9, 10, 11])
def test_layer_melody_does_not_clash_with_the_existing_lead(seed):
    pm = _song()
    composition = _layer(pm, seed)
    layer = next(i for i in composition.instruments if "Melody" in i.name)
    share, total = _clash_share(layer, _lead(pm))
    assert total > 0
    assert share < 0.05


def test_layer_melody_follows_the_rhythm_of_the_lead_and_stays_quieter():
    pm = _song()
    lead = _lead(pm)
    composition = _layer(pm, 9)
    layer = next(i for i in composition.instruments if "Melody" in i.name)
    starts = {round(n.start, 3) for n in lead.notes}
    assert {round(n.start, 3) for n in layer.notes} <= starts
    mean = lambda notes: sum(n.velocity for n in notes) / len(notes)
    assert mean(layer.notes) < mean(lead.notes)


def test_layer_melody_is_labelled_as_a_harmony():
    composition = _layer(_song(), 9)
    names = [i.name for i in composition.instruments]
    assert names == ["Melody harmony"]


def test_layer_melody_is_a_new_tune_when_the_song_has_no_lead():
    pm = pretty_midi.PrettyMIDI()
    chords = pretty_midi.Instrument(0)
    for bar in range(12):
        for pitch in (50, 53, 57):
            chords.notes.append(
                pretty_midi.Note(90, pitch, bar * BAR, (bar + 1) * BAR)
            )
    pm.instruments.append(chords)
    composition = _layer(pm, 9)
    assert [i.name for i in composition.instruments] == ["Melody"]


def test_new_song_mode_keeps_its_own_melody():
    composition = compose(_options(roles=("melody", "bass")))
    assert "Melody" in [i.name for i in composition.instruments]
