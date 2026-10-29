import pretty_midi
import pytest

from korshmer import midi_processors


def _make_midi():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for i, vel in enumerate([30, 50, 64, 90, 120]):
        inst.notes.append(pretty_midi.Note(vel, 60 + i, i * 0.5, i * 0.5 + 0.4))
    midi.instruments.append(inst)
    return midi


@pytest.fixture
def captured(monkeypatch):
    saved = {}
    midi = _make_midi()

    monkeypatch.setattr(midi_processors, "load_pretty_midi", lambda path: midi)

    def fake_save(pm, path, suffix):
        saved["pm"] = pm
        return "out.mid"

    monkeypatch.setattr(midi_processors, "_save_modified_midi", fake_save)
    return saved


def _velocities(pm):
    return [n.velocity for n in pm.instruments[0].notes]


def test_velocity_compress_maps_into_requested_range(captured):
    midi_processors.process_velocity_compress("in.mid", 40, 110)
    velocities = _velocities(captured["pm"])
    assert min(velocities) >= 40
    assert max(velocities) <= 110


def test_random_velocity_stays_in_requested_range(captured):
    midi_processors.process_rand_vel("in.mid", 60, 110)
    velocities = _velocities(captured["pm"])
    assert all(60 <= v <= 110 for v in velocities)


def test_velocity_compress_clamps_bounds_to_midi_range(captured):
    midi_processors.process_velocity_compress("in.mid", 0, 200)
    velocities = _velocities(captured["pm"])
    assert all(1 <= v <= 127 for v in velocities)


def test_random_velocity_clamps_bounds_to_midi_range(captured):
    midi_processors.process_rand_vel("in.mid", -20, 300)
    velocities = _velocities(captured["pm"])
    assert all(1 <= v <= 127 for v in velocities)


def test_compressed_velocities_survive_a_midi_round_trip(tmp_path):
    from korshmer.midi.transforms import compress_velocity

    midi = compress_velocity(_make_midi(), min_vel=0, max_vel=200)
    path = tmp_path / "out.mid"
    midi.write(str(path))
    reloaded = pretty_midi.PrettyMIDI(str(path))
    assert len(reloaded.instruments[0].notes) == 5


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "high", None])
def test_velocity_bounds_reject_non_numbers(bad):
    from korshmer.midi.transforms import compress_velocity

    with pytest.raises(ValueError):
        compress_velocity(_make_midi(), min_vel=bad, max_vel=100)


def _snapshot(midi):
    return [
        (n.velocity, n.pitch, n.start, n.end)
        for inst in midi.instruments
        for n in inst.notes
    ]


def test_compress_velocity_rejects_lowest_above_highest_and_keeps_notes():
    from korshmer.midi.transforms import compress_velocity

    midi = _make_midi()
    before = _snapshot(midi)
    with pytest.raises(ValueError, match="above the highest velocity"):
        compress_velocity(midi, min_vel=100, max_vel=40)
    assert _snapshot(midi) == before


def test_random_velocity_rejects_lowest_above_highest_and_keeps_notes():
    from korshmer.midi.transforms import randomize_velocity

    midi = _make_midi()
    before = _snapshot(midi)
    with pytest.raises(ValueError, match="Lowest velocity 100.*highest.*40"):
        randomize_velocity(midi, min_v=100, max_v=40)
    assert _snapshot(midi) == before


def test_velocity_bounds_may_be_equal():
    from korshmer.midi.transforms import compress_velocity, randomize_velocity

    squeezed = compress_velocity(_make_midi(), min_vel=70, max_vel=70)
    assert {n.velocity for n in squeezed.instruments[0].notes} == {70}
    fixed = randomize_velocity(_make_midi(), min_v=55, max_v=55)
    assert {n.velocity for n in fixed.instruments[0].notes} == {55}
