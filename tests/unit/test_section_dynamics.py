import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer.midi.section_dynamics import (
    apply_section_dynamics,
    apply_section_dynamics_from_file,
)

SR = 22050


def _song(levels_db, seconds=10.0):
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * 220 * t)
    return np.concatenate([tone * 10 ** (db / 20) for db in levels_db])


def _midi(seconds, sections, velocity=100, drum=False):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, is_drum=drum)
    for start in np.arange(0.5, seconds * sections, 1.0):
        inst.notes.append(pretty_midi.Note(velocity, 60, start, start + 0.4))
    pm.instruments.append(inst)
    return pm


def _section_means(pm, seconds, sections):
    notes = pm.instruments[0].notes
    return [
        np.mean(
            [
                n.velocity
                for n in notes
                if i * seconds + 3 <= n.start < (i + 1) * seconds - 3
            ]
        )
        for i in range(sections)
    ]


def test_quiet_verse_and_loud_chorus_get_different_velocities():
    audio = _song([-20, -8, -20, -8])
    pm = _midi(10.0, 4)
    assert apply_section_dynamics(pm, audio, SR) > 0
    verse, chorus, verse2, chorus2 = _section_means(pm, 10.0, 4)
    assert chorus == pytest.approx(100, abs=2)
    assert chorus2 == pytest.approx(100, abs=2)

    assert verse == pytest.approx(50, abs=4)
    assert verse2 == pytest.approx(verse, abs=2)


def test_note_to_note_detail_is_kept():
    audio = _song([-10])
    pm = _midi(10.0, 1)
    for i, n in enumerate(pm.instruments[0].notes):
        n.velocity = 60 if i % 2 else 110
    apply_section_dynamics(pm, audio, SR)
    velocities = [n.velocity for n in pm.instruments[0].notes]
    for loud, soft in zip(velocities[::2], velocities[1::2], strict=True):
        assert loud / soft == pytest.approx(110 / 60, rel=0.05)
    assert max(velocities) == 110


def test_drop_is_capped_and_never_reaches_zero():
    audio = _song([-60, 0])
    pm = _midi(10.0, 2, velocity=20)
    apply_section_dynamics(pm, audio, SR, max_drop_db=18.0)
    quiet = min(n.velocity for n in pm.instruments[0].notes)
    assert quiet == round(20 * 10 ** (-18 / 40))
    assert quiet >= 1


def test_drums_and_zero_amount_are_left_alone():
    audio = _song([-20, -8])
    drums = _midi(10.0, 2, drum=True)
    assert apply_section_dynamics(drums, audio, SR) == 0
    plain = _midi(10.0, 2)
    assert apply_section_dynamics(plain, audio, SR, amount=0) == 0
    assert {n.velocity for n in plain.instruments[0].notes} == {100}


def test_silence_and_empty_midi_do_not_fail():
    assert apply_section_dynamics(_midi(1.0, 1), np.zeros(10), SR) == 0
    assert apply_section_dynamics(pretty_midi.PrettyMIDI(), _song([0]), SR) == 0


def test_from_file_matches_in_memory(tmp_path):
    audio = _song([-20, -8]).astype(np.float32)
    path = tmp_path / "stem.wav"
    sf.write(path, np.stack([audio, audio], 1), SR)
    a, b = _midi(10.0, 2), _midi(10.0, 2)
    apply_section_dynamics(a, audio, SR)
    apply_section_dynamics_from_file(b, str(path))
    va = [n.velocity for n in a.instruments[0].notes]
    vb = [n.velocity for n in b.instruments[0].notes]
    assert np.allclose(va, vb, atol=1)
