import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer import audio_processors, constants
from korshmer.midi_generation.composer import ComposeOptions, compose
from korshmer.ui.viewmodels.editor_vm import EditorModel
from korshmer.utils.track_layout import merge_by_instrument

BPM = 140.0
BAR = 240.0 / BPM
SUB_BASS = 87
PIANO = 0
SEEDS = (1, 2, 3, 4, 5, 6)
OTHER_STYLES = ("Pop", "Rock", "EDM", "Lo-Fi", "Jazz")
FULL_BEND = 8191
WIDE_RANGE = 12


def _phonk(seed, **kwargs):
    kwargs.setdefault("minutes", 1.5)
    return compose(
        ComposeOptions(
            "A", "Minor", bpm=BPM, style="Phonk", seed=seed, **kwargs
        )
    )


def _bass(instruments):
    return next(i for i in instruments if i.program == SUB_BASS)


def _declared_range(inst):
    chosen = {101: 127, 100: 127}
    semitones = 2
    for change in sorted(inst.control_changes, key=lambda c: c.time):
        if change.number in chosen:
            chosen[change.number] = change.value
        elif change.number == 6 and chosen == {101: 0, 100: 0}:
            semitones = change.value
    return semitones


def _offset_at(inst, time):
    value = 0
    for bend in sorted(inst.pitch_bends, key=lambda b: b.time):
        if bend.time > time:
            break
        value = bend.pitch
    return value * _declared_range(inst) / 8192


def _slid_notes(inst):
    slid = []
    for note in inst.notes:
        if any(
            bend.pitch and note.start <= bend.time <= note.end
            for bend in inst.pitch_bends
        ):
            slid.append(note)
    return slid


def _declare_wide_range(inst):
    inst.control_changes += [
        pretty_midi.ControlChange(101, 0, 0.0),
        pretty_midi.ControlChange(100, 0, 0.0),
        pretty_midi.ControlChange(6, WIDE_RANGE, 0.01),
    ]


def _ramp(inst, start, end, first, last, steps=60):
    for k in range(steps + 1):
        inst.pitch_bends.append(
            pretty_midi.PitchBend(
                round(first + (last - first) * k / steps),
                start + (end - start) * k / steps,
            )
        )


def _song(*instruments):
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    midi.instruments.extend(instruments)
    return midi


def _real_render(midi, tmp_path, name="song"):
    soundfont = constants.SF3_PATH
    if not soundfont.exists():
        pytest.skip("soundfont missing")
    try:
        from korshmer.core.fluidsynth import FluidSynth

        FluidSynth(lib_dir=str(constants.FLUID_DIR)).close()
    except Exception:
        pytest.skip("FluidSynth library unavailable")
    source = tmp_path / f"{name}.mid"
    midi.write(str(source))
    target = tmp_path / f"{name}.wav"
    assert audio_processors.render_midi_to_wav(
        str(source), str(target), str(constants.FLUID_DIR), str(soundfont), BPM
    )
    audio, rate = sf.read(str(target))
    return audio.mean(axis=1), rate


def _heard_pitch(audio, rate, time, window=0.06):
    size = int(window * rate)
    start = max(0, int(time * rate) - size // 2)
    frame = audio[start : start + size]
    frame = (frame - frame.mean()) * np.hanning(len(frame))
    power = np.abs(np.fft.rfft(frame, 2 * size)) ** 2
    lags = np.fft.irfft(power)[: rate // 22]
    lags = lags / lags[0]
    first_dip = int(np.argmax(lags < 0))
    lags[:first_dip] = -1
    lag = int(np.argmax(lags >= 0.9 * lags.max()))
    while lag + 1 < len(lags) and lags[lag + 1] > lags[lag]:
        lag += 1
    before, peak, after = lags[lag - 1 : lag + 2]
    lag += 0.5 * (before - after) / (before - 2 * peak + after)
    return 69 + 12 * np.log2(rate / lag / 440.0)


class _RecordingSynth:
    def __init__(self, *args, **kwargs):
        self.position = 0
        self.events = []
        self.ranges = {}
        self.programs = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sf_load(self, path):
        return 1

    def set_reverb(self, *args):
        pass

    def set_chorus(self, *args):
        pass

    def cc(self, *args):
        pass

    def program_select(self, channel, sfid, bank, program):
        self.programs[channel] = (bank, program)

    def pitch_wheel_sens(self, channel, semitones):
        self.ranges[channel] = semitones

    def pitch_bend(self, channel, value):
        self.events.append(("bend", channel, value, self.position))

    def note_on(self, channel, pitch, velocity):
        self.events.append(("on", channel, pitch, self.position))

    def note_off(self, channel, pitch):
        self.events.append(("off", channel, pitch, self.position))

    def get_samples(self, count):
        self.position += count
        return np.zeros(count, np.float32), np.zeros(count, np.float32)


def _recorded_render(monkeypatch, tmp_path, midi):
    synths = []

    def make(*args, **kwargs):
        synths.append(_RecordingSynth())
        return synths[-1]

    monkeypatch.setattr(audio_processors, "FluidSynth", make)
    monkeypatch.setattr(
        audio_processors, "save_normalized_audio", lambda *a, **k: None
    )
    monkeypatch.setattr(
        audio_processors, "read_soundfont_kits", lambda p: set()
    )
    source = tmp_path / "recorded.mid"
    midi.write(str(source))
    audio_processors.render_midi_to_wav(
        str(source), str(tmp_path / "recorded.wav"), "x", "y", bpm=BPM
    )
    return synths


def _channel_of(synth, program, bank=0):
    return next(
        channel
        for channel, sound in synth.programs.items()
        if sound == (bank, program)
    )


def test_a_slide_on_the_sub_bass_is_heard_as_a_glide_of_one_octave(tmp_path):
    flat = pretty_midi.Instrument(SUB_BASS)
    flat.notes.append(pretty_midi.Note(104, 45, 0.0, 2.0))
    slid = pretty_midi.Instrument(SUB_BASS)
    slid.notes.append(pretty_midi.Note(104, 45, 0.0, 2.0))
    _declare_wide_range(slid)
    slid.pitch_bends.append(pretty_midi.PitchBend(-8192, 0.02))
    _ramp(slid, 0.5, 1.0, -8192, 0)

    plain, rate = _real_render(_song(flat), tmp_path, "flat")
    glide, rate = _real_render(_song(slid), tmp_path, "slid")

    assert _heard_pitch(plain, rate, 0.3) == pytest.approx(45, abs=0.2)
    assert _heard_pitch(plain, rate, 1.5) == pytest.approx(45, abs=0.2)
    assert _heard_pitch(glide, rate, 0.3) == pytest.approx(33, abs=0.3)
    assert _heard_pitch(glide, rate, 0.75) == pytest.approx(39, abs=0.5)
    assert _heard_pitch(glide, rate, 1.5) == pytest.approx(45, abs=0.2)


def test_a_bend_without_a_declared_range_moves_a_piano_note_two_semitones(
    tmp_path,
):
    piano = pretty_midi.Instrument(PIANO)
    piano.notes.append(pretty_midi.Note(100, 60, 0.0, 2.0))
    _ramp(piano, 0.5, 1.0, 0, FULL_BEND)

    audio, rate = _real_render(_song(piano), tmp_path)

    assert _heard_pitch(audio, rate, 0.3) == pytest.approx(60, abs=0.2)
    assert _heard_pitch(audio, rate, 0.75) == pytest.approx(61, abs=0.3)
    assert _heard_pitch(audio, rate, 1.5) == pytest.approx(62, abs=0.2)


def test_render_sends_bends_only_to_the_channel_of_the_track_that_has_them(
    monkeypatch, tmp_path
):
    organ = pretty_midi.Instrument(19)
    organ.notes.append(pretty_midi.Note(90, 60, 0.0, 2.0))
    strings = pretty_midi.Instrument(48)
    strings.notes.append(pretty_midi.Note(90, 64, 0.0, 2.0))
    strings.pitch_bends.append(pretty_midi.PitchBend(2000, 0.5))
    strings.pitch_bends.append(pretty_midi.PitchBend(0, 1.0))

    (synth,) = _recorded_render(monkeypatch, tmp_path, _song(organ, strings))

    bends = [event for event in synth.events if event[0] == "bend"]
    assert [(channel, value) for _, channel, value, _ in bends] == [
        (_channel_of(synth, 48), 8192 + 2000),
        (_channel_of(synth, 48), 8192),
    ]
    assert [at / 44100 for *_, at in bends] == pytest.approx(
        [0.5, 1.0], abs=0.005
    )
    assert _channel_of(synth, 19) != _channel_of(synth, 48)


def test_render_uses_the_bend_range_a_track_declares_and_two_semitones_else(
    monkeypatch, tmp_path
):
    wide = pretty_midi.Instrument(48)
    wide.notes.append(pretty_midi.Note(90, 60, 0.0, 1.0))
    _declare_wide_range(wide)
    wide.pitch_bends.append(pretty_midi.PitchBend(1000, 0.5))
    plain = pretty_midi.Instrument(19)
    plain.notes.append(pretty_midi.Note(90, 64, 0.0, 1.0))
    plain.pitch_bends.append(pretty_midi.PitchBend(1000, 0.5))

    (synth,) = _recorded_render(monkeypatch, tmp_path, _song(wide, plain))

    assert synth.ranges == {
        _channel_of(synth, 48): WIDE_RANGE,
        _channel_of(synth, 19): 2,
    }


def test_a_bend_at_the_start_of_a_note_is_applied_before_the_note_sounds(
    monkeypatch, tmp_path
):
    lead = pretty_midi.Instrument(48)
    lead.notes.append(pretty_midi.Note(90, 60, 0.0, 0.5))
    lead.notes.append(pretty_midi.Note(90, 62, 0.5, 1.0))
    lead.pitch_bends.append(pretty_midi.PitchBend(-3000, 0.5))

    (synth,) = _recorded_render(monkeypatch, tmp_path, _song(lead))

    kinds = [(kind, value) for kind, _, value, _ in synth.events]
    assert kinds == [
        ("on", 60),
        ("off", 60),
        ("bend", 8192 - 3000),
        ("on", 62),
        ("off", 62),
    ]


def test_the_driven_sub_bass_gets_its_bends_on_its_own_channel_and_no_other(
    monkeypatch, tmp_path
):
    piano = pretty_midi.Instrument(PIANO)
    piano.notes.append(pretty_midi.Note(90, 60, 0.0, 1.0))
    bass = pretty_midi.Instrument(SUB_BASS)
    bass.notes.append(pretty_midi.Note(104, 33, 0.0, 1.0))
    _declare_wide_range(bass)
    bass.pitch_bends.append(pretty_midi.PitchBend(-4096, 0.5))

    synths = _recorded_render(monkeypatch, tmp_path, _song(piano, bass))

    with_bends = [
        synth for synth in synths if any(e[0] == "bend" for e in synth.events)
    ]
    assert len(synths) == 2
    assert len(with_bends) == 1
    assert set(with_bends[0].programs.values()) == {(0, SUB_BASS)}
    assert list(with_bends[0].ranges.values()) == [WIDE_RANGE]


def test_bends_of_one_drum_track_do_not_retune_another_drum_track(
    monkeypatch, tmp_path
):
    kick = pretty_midi.Instrument(0, is_drum=True)
    kick.notes.append(pretty_midi.Note(100, 36, 0.0, 0.2))
    kick.pitch_bends.append(pretty_midi.PitchBend(4000, 0.1))
    hats = pretty_midi.Instrument(0, is_drum=True)
    hats.notes.append(pretty_midi.Note(100, 42, 0.0, 0.2))

    (synth,) = _recorded_render(monkeypatch, tmp_path, _song(kick, hats))

    assert [event for event in synth.events if event[0] == "bend"] == []


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_bass_slides_on_a_few_notes_of_a_phrase_and_never_on_all(seed):
    bass = _bass(_phonk(seed).tracks)
    slid = _slid_notes(bass)
    phrases = {}
    for note in bass.notes:
        phrases.setdefault(int(note.start / BAR + 1e-6) // 4, []).append(note)

    assert slid
    assert len(slid) <= len(bass.notes) / 3
    assert _declared_range(bass) == WIDE_RANGE
    for notes in phrases.values():
        in_phrase = [note for note in notes if note in slid]
        assert len(in_phrase) <= 3
        assert len(in_phrase) < len(notes) or len(notes) == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_drops_have_more_slides_per_bar_than_the_rest_of_the_song(seed):
    song = _phonk(seed)
    bass = _bass(song.tracks)
    drop_bars = {
        bar
        for section in song.sections
        if section.name == "Drop"
        for bar in range(section.start_bar, section.start_bar + section.bars)
    }
    played_bars = {int(note.start / BAR + 1e-6) for note in bass.notes}
    slid_bars = [int(note.start / BAR + 1e-6) for note in _slid_notes(bass)]
    in_drop = sum(1 for bar in slid_bars if bar in drop_bars)
    calm_bars = played_bars - drop_bars

    assert in_drop / len(played_bars & drop_bars) > (
        (len(slid_bars) - in_drop) / len(calm_bars)
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_every_phonk_bass_note_is_at_its_written_pitch_at_its_middle(seed):
    bass = _bass(_phonk(seed).tracks)

    for note in bass.notes:
        middle = 0.5 * (note.start + note.end)
        assert _offset_at(bass, middle) == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_a_phonk_slide_moves_at_most_an_octave_and_lasts_a_fraction_of_a_bar(
    seed,
):
    bass = _bass(_phonk(seed).tracks)
    bends = sorted(bass.pitch_bends, key=lambda b: b.time)
    stretches = []
    began = None
    for bend in bends:
        if bend.pitch and began is None:
            began = bend.time
        elif not bend.pitch and began is not None:
            stretches.append(bend.time - began)
            began = None

    assert began is None
    assert max(abs(_offset_at(bass, b.time)) for b in bends) <= WIDE_RANGE
    assert max(abs(_offset_at(bass, b.time)) for b in bends) >= 5
    assert max(stretches) <= BAR / 2
    assert min(stretches) >= BAR / 64


def test_phonk_slides_are_the_same_for_one_seed_and_differ_between_seeds():
    def bends(seed):
        return [
            (b.time, b.pitch) for b in _bass(_phonk(seed).tracks).pitch_bends
        ]

    assert bends(3) == bends(3)
    assert bends(3) != bends(4)


@pytest.mark.parametrize("style", OTHER_STYLES)
def test_styles_other_than_phonk_get_no_pitch_bends(style):
    song = compose(
        ComposeOptions("A", "Minor", bpm=120.0, bars=16, style=style, seed=5)
    )

    assert all(not inst.pitch_bends for inst in song.tracks)
    assert all(not inst.control_changes for inst in song.tracks)


def _through_a_midi_file(instruments, tmp_path, name="generated.mid"):
    midi = _song(*instruments)
    midi.write(str(tmp_path / name))
    return pretty_midi.PrettyMIDI(str(tmp_path / name))


def _bend_points(inst):
    return [(round(b.time, 3), b.pitch) for b in inst.pitch_bends]


def test_slides_of_a_generated_song_survive_the_editor_a_saved_song_and_export(
    tmp_path,
):
    generated = _through_a_midi_file(_phonk(3, bars=16).tracks, tmp_path)
    wanted = _bend_points(_bass(generated.instruments))
    model = EditorModel(tempo=BPM)
    model.load_midi(generated, replace=True, merge_instruments=True)

    saved = json.loads(json.dumps(model.to_dict()))
    exported = _through_a_midi_file(
        EditorModel.from_dict(saved).to_pretty_midi(tempo=BPM).instruments,
        tmp_path,
        "exported.mid",
    )

    bass = _bass(exported.instruments)
    assert wanted
    assert _bend_points(bass) == wanted
    assert _declared_range(bass) == WIDE_RANGE
    assert all(
        not inst.pitch_bends
        for inst in exported.instruments
        if inst.program != SUB_BASS
    )


def test_slides_follow_their_notes_when_the_editor_tempo_changes_and_on_undo():
    model = EditorModel(tempo=BPM)
    model.load_midi(_song(*_phonk(3, bars=16).tracks), replace=True)
    before = _bass(model.to_pretty_midi().instruments)
    first_slid = _slid_notes(before)[0]
    lead = next(
        b.time for b in before.pitch_bends if b.time >= first_slid.start
    )

    model.snapshot()
    model.follow_tempo(2.0)
    stretched = _bass(model.to_pretty_midi().instruments)
    model.undo()
    undone = _bass(model.to_pretty_midi().instruments)

    assert [b.time for b in stretched.pitch_bends] == pytest.approx(
        [2.0 * b.time for b in before.pitch_bends]
    )
    assert _slid_notes(stretched)[0].start == pytest.approx(
        2.0 * first_slid.start
    )
    assert [b.time for b in undone.pitch_bends] == pytest.approx(
        [2.0 * b.time for b in before.pitch_bends]
    )
    assert lead - first_slid.start < BAR


def test_a_bent_track_is_not_merged_with_a_same_sound_track_it_would_retune():
    bent = pretty_midi.Instrument(48, name="Bent")
    bent.notes.append(pretty_midi.Note(90, 60, 0.0, 1.0))
    bent.pitch_bends.append(pretty_midi.PitchBend(3000, 0.2))
    bent.pitch_bends.append(pretty_midi.PitchBend(0, 0.8))
    straight = pretty_midi.Instrument(48, name="Straight")
    straight.notes.append(pretty_midi.Note(90, 67, 0.0, 1.0))
    model = EditorModel(tempo=BPM)
    model.load_midi(_song(bent, straight), replace=True)

    exported = model.to_pretty_midi().instruments

    assert sorted(len(inst.pitch_bends) for inst in exported) == [0, 2]
    assert len(merge_by_instrument([bent, straight])) == 2


def test_a_continuation_with_slides_joins_the_editor_track_of_the_same_sound():
    first = pretty_midi.Instrument(SUB_BASS)
    first.notes.append(pretty_midi.Note(104, 33, 0.0, 1.0))
    _declare_wide_range(first)
    first.pitch_bends.append(pretty_midi.PitchBend(-4096, 0.5))
    first.pitch_bends.append(pretty_midi.PitchBend(0, 0.6))
    later = pretty_midi.Instrument(SUB_BASS)
    later.notes.append(pretty_midi.Note(104, 35, 4.0, 5.0))
    _declare_wide_range(later)
    later.pitch_bends.append(pretty_midi.PitchBend(4096, 4.0))
    later.pitch_bends.append(pretty_midi.PitchBend(0, 4.1))
    model = EditorModel(tempo=BPM)
    model.load_midi(_song(first), replace=True, merge_instruments=True)
    model.load_midi(_song(later), replace=False, merge_instruments=True)

    (bass,) = model.to_pretty_midi().instruments

    assert len(model.tracks) == 1
    assert [(b.time, b.pitch) for b in bass.pitch_bends] == [
        (0.5, -4096),
        (0.6, 0),
        (4.0, 4096),
        (4.1, 0),
    ]
    assert _declared_range(bass) == WIDE_RANGE


def test_a_generated_phonk_bass_glides_in_the_audio_after_the_editor_path(
    tmp_path,
):
    model = EditorModel(tempo=BPM)
    model.load_midi(
        _through_a_midi_file(_phonk(3, bars=16).tracks, tmp_path),
        replace=True,
        merge_instruments=True,
    )
    restored = EditorModel.from_dict(json.loads(json.dumps(model.to_dict())))
    bass = _bass(restored.to_pretty_midi(tempo=BPM).instruments)

    audio, rate = _real_render(_song(bass), tmp_path, "bass")

    slid = _slid_notes(bass)
    widest = max(
        (b for b in bass.pitch_bends),
        key=lambda b: abs(b.pitch) if b.time > 0.1 else 0,
    )
    note = next(n for n in slid if n.start <= widest.time <= n.end + 1e-6)
    straight = next(n for n in bass.notes if n not in slid)
    probes = [
        b.time
        for b in bass.pitch_bends
        if note.start + 0.035 <= b.time <= note.end - 0.035
        and abs(_offset_at(bass, b.time)) >= 3
    ]
    assert probes
    for time in probes[:: max(1, len(probes) // 5)]:
        wanted = note.pitch + np.mean(
            [_offset_at(bass, time + shift) for shift in (-0.02, 0.0, 0.02)]
        )
        assert _heard_pitch(audio, rate, time) == pytest.approx(wanted, abs=1.5)
    assert _heard_pitch(
        audio, rate, 0.5 * (straight.start + straight.end)
    ) == pytest.approx(straight.pitch, abs=0.2)


def test_editor_playback_sends_the_slides_of_the_editor_song_to_the_render(
    monkeypatch, tmp_path
):
    import types

    import korshmer.__main__ as app_main

    started = []

    class _Signal:
        def connect(self, slot):
            pass

    class _Preview:
        preview_ready = _Signal()
        preview_failed = _Signal()

        def __init__(self, midi_path, *args, **kwargs):
            self.midi_path = midi_path

        def start(self):
            started.append(pretty_midi.PrettyMIDI(self.midi_path))

    soundfont = tmp_path / "soundfont.sf3"
    soundfont.write_bytes(b"RIFF")
    monkeypatch.setattr(app_main, "MidiPreviewThread", _Preview)
    monkeypatch.setattr(app_main, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(app_main, "SF3_PATH", str(soundfont))
    monkeypatch.setattr(app_main, "publish", lambda *a, **k: None)
    window = types.SimpleNamespace(
        spin_bpm=types.SimpleNamespace(value=lambda: BPM),
        arranger=types.SimpleNamespace(
            get_audio_clips=lambda: [],
            piano_roll=types.SimpleNamespace(playback_offset=0),
        ),
    )
    model = EditorModel(tempo=BPM)
    model.load_midi(_song(*_phonk(3, bars=16).tracks), replace=True)
    song = model.to_pretty_midi(tempo=BPM)

    app_main.MainWindow.handle_editor_play_request(window, song)

    (played,) = started
    bass = _bass(played.instruments)
    wanted = _bass(song.instruments).pitch_bends
    assert wanted
    assert [b.pitch for b in bass.pitch_bends] == [b.pitch for b in wanted]
    assert [b.time for b in bass.pitch_bends] == pytest.approx(
        [b.time for b in wanted], abs=0.003
    )
    assert _declared_range(bass) == WIDE_RANGE
