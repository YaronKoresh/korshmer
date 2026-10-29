import sys
from types import SimpleNamespace

import pretty_midi
import pytest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
)

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer import midi_generation
from korshmer.audio_processing.stem_session import StemSession
from korshmer.midi.adaptive_snap import (
    find_beat_offset,
    shared_beat_offset,
    snap_adaptive,
)
from korshmer.midi_generation import (
    ComposeOptions,
    compose,
    composer,
    song_match,
)
from korshmer.mixins.actions import editor as editor_module
from korshmer.mixins.actions.editor import EditorActionsMixin
from korshmer.mixins.actions.stems import StemsActionsMixin
from tests.unit.test_estimate_grid import circular, groove

BPM = 118.4
BEAT = 60.0 / BPM
STEP = BEAT / 4
PHASE = 0.37


def _options(**changes):
    values = {
        "root": "C",
        "scale": "Major",
        "bpm": BPM,
        "seed": 5,
        "roles": ("bass", "chords"),
        "texture_batches": 0,
        "mode": "Layer over editor",
        "match": "Strong",
    }
    values.update(changes)
    return ComposeOptions(**values)


def _pitched_starts(composition):
    return [
        n.start
        for inst in composition.instruments
        if not inst.is_drum
        for n in inst.notes
    ]


def _on_lattice(starts, offset):
    close = [circular(s, offset, STEP) < 0.004 for s in starts]
    return sum(close) / len(close)


def test_compose_uses_an_explicit_grid_offset():
    song = groove(BPM, PHASE)
    estimated = compose(_options(), existing=song)
    explicit = compose(_options(grid_offset=0.2), existing=song)
    folded = compose(_options(grid_offset=0.2 + 4 * BEAT), existing=song)

    assert _on_lattice(_pitched_starts(estimated), PHASE) > 0.95
    assert _on_lattice(_pitched_starts(explicit), 0.2) > 0.95
    assert _pitched_starts(folded) == pytest.approx(
        _pitched_starts(explicit), abs=1e-9
    )
    assert compose(_options(), existing=song).grid_offset == pytest.approx(
        PHASE, abs=0.03
    )
    assert explicit.grid_offset == 0.2


@pytest.mark.parametrize("value", [-0.1, float("nan"), True, "0.3"])
def test_compose_rejects_a_bad_grid_offset(value):
    with pytest.raises(ValueError, match="Grid offset"):
        compose(_options(grid_offset=value), existing=groove(BPM, PHASE))


class FakeRoll:
    def __init__(self, grid_offset):
        self.grid_offset = grid_offset
        self.set_calls = []

    def set_grid_offset(self, seconds):
        self.set_calls.append(seconds)
        self.grid_offset = seconds


class FakeArranger:
    def __init__(self, song, roll):
        self.piano_roll = roll
        self.model = SimpleNamespace(tracks={})
        self.song = song
        self.loaded = []
        self.merge_flags = []

    def get_midi_from_editor(self):
        return self.song

    def load_midi_file(
        self, path, replace=True, tempo=None, merge_instruments=False
    ):
        self.loaded.append((path, replace, tempo))
        self.merge_flags.append(merge_instruments)


class FakeTabs:
    def __init__(self, arranger):
        self.current = arranger

    def currentWidget(self):
        return self.current

    def setCurrentWidget(self, widget):
        self.current = widget


def _combo(*items):
    box = QComboBox()
    box.addItems(list(items))
    return box


class GeneratorHost(EditorActionsMixin):
    def __init__(self, arranger):
        self.arranger = arranger
        self.tabs = FakeTabs(arranger)
        self.combo_root = _combo("C")
        self.combo_scale = _combo("Major")
        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setRange(20, 300)
        self.spin_bpm.setValue(BPM)
        self.spin_gen_seed = QSpinBox()
        self.spin_gen_seed.setRange(0, 99999)
        self.spin_gen_seed.setValue(7)
        self.spin_gen_minutes = QDoubleSpinBox()
        self.spin_gen_minutes.setValue(0.5)
        self.combo_gen_mode = _combo("Layer over editor")
        self.combo_gen_style = _combo("Pop")
        self.combo_gen_mood = _combo("Auto")
        self.chk_gen_replace = QCheckBox()
        bass = QCheckBox()
        bass.setChecked(True)
        self.gen_role_boxes = {"bass": bass}


@pytest.fixture
def captured(monkeypatch, tmp_path):
    options = []
    real = midi_generation.compose

    def spy(opts, existing=None):
        options.append(opts)
        return real(opts, existing=existing)

    monkeypatch.setattr(midi_generation, "compose", spy)
    monkeypatch.setattr(editor_module, "_layers_dir", lambda: str(tmp_path))
    return options


def test_editor_passes_its_grid_offset_to_compose(captured):
    roll = FakeRoll(0.2)
    host = GeneratorHost(FakeArranger(groove(BPM, PHASE), roll))
    host.run_midi_generator()

    assert captured[0].grid_offset == 0.2
    assert roll.set_calls == []
    path = host.arranger.loaded[0][0]
    starts = [
        n.start
        for inst in pretty_midi.PrettyMIDI(path).instruments
        for n in inst.notes
    ]
    assert _on_lattice(starts, 0.2) > 0.95


def test_generated_tracks_merge_into_the_editors_instruments(captured):
    host = GeneratorHost(FakeArranger(groove(BPM, PHASE), FakeRoll(0.2)))
    host.run_midi_generator()

    assert host.arranger.merge_flags == [True]
    instruments = pretty_midi.PrettyMIDI(host.arranger.loaded[0][0]).instruments
    keys = [(i.is_drum, 0 if i.is_drum else i.program) for i in instruments]
    assert len(keys) == len(set(keys))


def test_editor_adopts_the_grid_it_layered_on(captured):
    roll = FakeRoll(0.0)
    host = GeneratorHost(FakeArranger(groove(BPM, PHASE), roll))
    host.run_midi_generator()

    assert captured[0].grid_offset is None
    assert len(roll.set_calls) == 1
    assert roll.set_calls[0] == pytest.approx(PHASE, abs=0.03)


def test_editor_without_a_readable_grid_offset_still_generates(captured):
    roll = SimpleNamespace(grid_offset=lambda: 0.3)
    host = GeneratorHost(FakeArranger(groove(BPM, PHASE), roll))
    host.run_midi_generator()

    assert captured[0].grid_offset is None
    assert len(host.arranger.loaded) == 1


class SessionRoll:
    def __init__(self):
        self.grid_calls = []
        self.detect_calls = []

    def set_tempo(self, bpm):
        self.tempo = bpm

    def set_grid_offset(self, seconds):
        self.grid_calls.append(seconds)

    def detect_grid_offset(self, midi):
        self.detect_calls.append(midi)


class SessionArranger:
    def __init__(self, song):
        self.piano_roll = SessionRoll()
        self.song = song

    def get_midi_from_editor(self):
        return self.song

    def load_midi_file(self, path, replace=True, tempo=None, file_tempo=True):
        pass


class SessionTabs:
    def setCurrentIndex(self, index):
        self.index = index


class SessionHost(StemsActionsMixin):
    def __init__(self, song):
        self.arranger = SessionArranger(song)
        self.tabs = SessionTabs()


def _session(tmp_path, grid_offset):
    midi_path = tmp_path / "bass.mid"
    midi_path.write_bytes(b"")
    session = StemSession(
        source_path=str(tmp_path / "song.wav"),
        output_dir=str(tmp_path),
        tempo=BPM,
        midi_files=[str(midi_path)],
    )
    session.grid_offset = grid_offset
    return session


def test_session_grid_offset_reaches_the_piano_roll(tmp_path):
    song = groove(BPM, PHASE)
    host = SessionHost(song)
    assert host.load_session_into_editor(_session(tmp_path, PHASE)) is True
    roll = host.arranger.piano_roll
    assert roll.grid_calls == [PHASE]
    assert roll.detect_calls == []

    old = SessionHost(song)
    assert old.load_session_into_editor(_session(tmp_path, None)) is True
    roll = old.arranger.piano_roll
    assert roll.grid_calls == []
    assert roll.detect_calls == [song]


def test_session_without_a_piano_roll_still_loads(tmp_path):
    host = SessionHost(groove(BPM, PHASE))
    del host.arranger.piano_roll
    assert host.load_session_into_editor(_session(tmp_path, PHASE)) is True


def _layer_with_texture(song, monkeypatch, **changes):
    made = []
    real = composer.compose_routine_tracks

    def spy(*args, **kwargs):
        tracks = real(*args, **kwargs)
        made.extend(tracks)
        return tracks

    monkeypatch.setattr(composer, "compose_routine_tracks", spy)
    options = _options(
        texture_batches=1, skip_empty_edges=False, roles=("bass",), **changes
    )
    return compose(options, existing=song), made


def test_layer_adds_no_texture_drums_when_the_song_has_drums(monkeypatch):
    song = groove(BPM, PHASE)
    result, made = _layer_with_texture(song, monkeypatch)

    assert any(t.is_drum for t in made)
    assert not any(i.is_drum for i in result.instruments)
    texture = [i for i in result.instruments if i.name.startswith("Schema")]
    assert texture == [t for t in made if not t.is_drum]

    song.instruments = [i for i in song.instruments if not i.is_drum]
    made.clear()
    result, made = _layer_with_texture(song, monkeypatch)
    assert any(i.is_drum for i in result.instruments)


def _line(name, pitches, start=0.0, gap=0.5, stack=()):
    inst = pretty_midi.Instrument(0, name=name)
    for k, pitch in enumerate(pitches):
        at = start + k * gap
        inst.notes.append(pretty_midi.Note(90, pitch, at, at + gap * 0.9))
        if k in stack:
            inst.notes.append(
                pretty_midi.Note(90, pitch + 4, at + 0.004, at + gap * 0.9)
            )
    return inst


def test_find_lead_prefers_the_track_named_vocal():
    song = pretty_midi.PrettyMIDI()
    high = _line("Other", [84, 86] * 8)
    vocal = _line("Vocals", [60, 62, 64, 65, 67, 65, 64, 62] * 2, stack=(1, 5))
    song.instruments = [high, vocal]

    lead = song_match.find_lead(song)
    assert len(vocal.notes) == 18
    assert len(lead) == 16
    assert {n.pitch for n in lead} <= {n.pitch for n in vocal.notes}
    assert lead == sorted(lead, key=lambda n: n.start)
    assert [n.pitch for n in lead][:3] == [60, 66, 64]
    assert song_match.find_lead(song, prefer=None) == high.notes

    chordal = pretty_midi.Instrument(0, name="Vocals")
    for k in range(10):
        for pitch in (60, 64, 67):
            chordal.notes.append(
                pretty_midi.Note(90, pitch, k * 0.5, k * 0.5 + 2.0)
            )
    song.instruments = [high, chordal]
    assert song_match.find_lead(song) == high.notes


def _layer_notes(composition):
    return sorted(
        (i.name, round(n.start, 4), n.pitch)
        for i in composition.instruments
        for n in i.notes
    )


def test_layer_skips_empty_leading_and_trailing_bars():
    tempo = 120.0
    bar = 240.0 / tempo
    song = groove(tempo, 8 * bar, bars=17)
    on = compose(_options(bpm=tempo, texture_batches=1), existing=song)
    off = compose(
        _options(bpm=tempo, texture_batches=1, skip_empty_edges=False),
        existing=song,
    )

    low, high = 8 * bar, 25 * bar
    assert min(n.start for i in off.instruments for n in i.notes) < low
    assert max(n.end for i in off.instruments for n in i.notes) > high
    assert min(n.start for i in on.instruments for n in i.notes) >= low
    assert max(n.end for i in on.instruments for n in i.notes) <= high + 1e-6
    inside = [r for r in _layer_notes(off) if low <= r[1] < high]
    assert _layer_notes(on) == inside
    assert on.total_bars == off.total_bars


def test_occupied_bars_report_the_bars_the_notes_sound_in():
    song = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [
        pretty_midi.Note(90, 60, 5.0, 5.5),
        pretty_midi.Note(90, 62, 11.0, 12.0),
    ]
    song.instruments.append(inst)

    assert song_match.occupied_bars(song, 120.0) == (2, 6)
    assert song_match.occupied_bars(pretty_midi.PrettyMIDI(), 120.0) == (0, 0)


def test_layer_keeps_empty_bars_inside_the_song():
    tempo = 120.0
    bar = 240.0 / tempo
    song = groove(tempo, 0.0, bars=12)
    for inst in song.instruments:
        inst.notes = [n for n in inst.notes if not 4 * bar <= n.start < 6 * bar]
    on = compose(_options(bpm=tempo), existing=song)
    off = compose(_options(bpm=tempo, skip_empty_edges=False), existing=song)

    assert _layer_notes(on) == _layer_notes(off)


def _stems(tempo):
    beat = 60.0 / tempo
    bar = 4 * beat
    drums = pretty_midi.Instrument(0, is_drum=True, name="Drums")
    bass = pretty_midi.Instrument(33, name="Bass")
    vocal = pretty_midi.Instrument(0, name="Vocals")
    for index in range(16):
        origin = index * bar
        for k, pitch in enumerate((36, 38, 36, 38)):
            start = origin + k * beat
            drums.notes.append(
                pretty_midi.Note(100, pitch, start, start + 0.05)
            )
        for k in (0.5, 2.5):
            start = origin + k * beat
            bass.notes.append(
                pretty_midi.Note(90, 40, start, start + 0.6 * beat)
            )
        start = origin + 1.25 * beat
        vocal.notes.append(pretty_midi.Note(80, 67, start, start + 0.4))
    marker = 4 * bar + beat / 4
    for inst, pitch in ((drums, 42), (bass, 41), (vocal, 70)):
        inst.notes.append(pretty_midi.Note(80, pitch, marker, marker + 0.05))
    stems = []
    for inst in (drums, bass, vocal):
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        midi.instruments.append(inst)
        stems.append(midi)
    return stems


def _marker_starts(stems, tempo, offsets, phase_div):
    out = []
    for midi, offset, pitch in zip(stems, offsets, (42, 41, 70), strict=True):
        snap_adaptive(
            midi, tempo, offset, finest=4, swing=1.0, phase_div=phase_div
        )
        out.append(
            next(n.start for n in midi.instruments[0].notes if n.pitch == pitch)
        )
    return out


def test_shared_beat_offset_aligns_all_stems():
    tempo = 120.0
    alone = [find_beat_offset(m, tempo, 0.0, 4) for m in _stems(tempo)]
    shared = shared_beat_offset(_stems(tempo), tempo, 0.0, 4)

    assert len({round(o, 4) for o in alone}) > 1
    assert shared == pytest.approx(0.0)
    assert find_beat_offset(_stems(tempo)[0], tempo, shared, 1) == shared

    own = _marker_starts(_stems(tempo), tempo, [0.0] * 3, 4)
    together = _marker_starts(_stems(tempo), tempo, [shared] * 3, 1)
    assert len({round(t, 6) for t in own}) > 1
    assert len({round(t, 6) for t in together}) == 1
