import io

import pretty_midi

from korshmer.midi_algorithms import invert_velocity


def _midi(velocities):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i, v in enumerate(velocities):
        inst.notes.append(
            pretty_midi.Note(
                velocity=v, pitch=60 + i, start=i * 0.5, end=i * 0.5 + 0.4
            )
        )
    midi.instruments.append(inst)
    return midi


def test_inverted_velocities_stay_in_valid_range():
    midi = invert_velocity(_midi([1, 2, 64, 100, 127]))
    velocities = [n.velocity for n in midi.instruments[0].notes]
    assert velocities == [127, 126, 64, 28, 1]


def test_loudest_note_is_kept_after_write_and_reload():
    midi = invert_velocity(_midi([127, 100, 1]))
    buffer = io.BytesIO()
    midi.write(buffer)
    buffer.seek(0)
    reloaded = pretty_midi.PrettyMIDI(buffer)
    assert len(reloaded.instruments[0].notes) == 3


def test_double_inversion_restores_original():
    original = [1, 30, 64, 99, 127]
    midi = invert_velocity(invert_velocity(_midi(original)))
    assert [n.velocity for n in midi.instruments[0].notes] == original
