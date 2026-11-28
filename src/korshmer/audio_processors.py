import contextlib
import copy
import logging
import os
import struct

import numpy as np
import pretty_midi
import soundfile as sf

from korshmer.audio_file import AudioFile
from korshmer.audio_processing import (
    Chorus,
    Compressor,
    Delay,
    Limiter,
    ParametricEQ,
    Phaser,
    RangeEQ,
    Reverb,
    SmartMastering,
    StereoWidener,
    Tremolo,
    apply_filter,
    apply_mono_mix,
    apply_noise_gate,
    apply_normalize,
    apply_reverse_audio,
    apply_transient_shaper,
    pitch_shift_file,
    save_audio,
    save_normalized_audio,
    slice_audio_onsets,
)
from korshmer.audio_processing.validation import (
    EMPTY_AUDIO_MESSAGE,
    is_empty_audio,
    validate_pitch_shift_semitones,
)
from korshmer.constants import DATA_DIR
from korshmer.core.fluidsynth import FluidSynth
from korshmer.midi import generate_piano_roll_image
from korshmer.midi.event_processor import MidiEvent, MidiEventQueue
from korshmer.utils.bend_range import bend_range_of

_PROC_DIR = os.path.join(DATA_DIR, "Processed_Audio")


def _output_path(path, suffix):
    os.makedirs(_PROC_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(_PROC_DIR, f"{base}_{suffix}.wav")


def _load_2d(path):
    af = AudioFile.load(path).ensure_2d()
    if is_empty_audio(af.y):
        raise ValueError(f"{EMPTY_AUDIO_MESSAGE}: {path}")
    return af


def process_smart_mastering_file(
    path, progress_callback=None, stop_callback=None, **master_kwargs
):
    af = _load_2d(path)
    from korshmer.audio_processing.smart_mastering import (
        MASTERING_CANCELLED_MESSAGE,
        SmartMasteringConfig,
    )
    from korshmer.exceptions import MasteringCancelledError

    cfg = SmartMasteringConfig(**master_kwargs)
    processor = SmartMastering(cfg)
    af.y, sr = processor.process(
        af.y,
        af.sr,
        progress_callback=progress_callback,
        stop_callback=stop_callback,
    )
    out_path = _output_path(path, "Mastered")
    unfinished_path = _output_path(path, "Mastered_unfinished")
    try:
        save_audio(path=unfinished_path, y=af.y, sr=sr, file_format="WAV")
        if stop_callback is not None and stop_callback():
            raise MasteringCancelledError(MASTERING_CANCELLED_MESSAGE)
        os.replace(unfinished_path, out_path)
    finally:
        if os.path.exists(unfinished_path):
            os.remove(unfinished_path)
    return out_path


def process_range_eq_file(
    path, freq_start, freq_end, gain, peak_pos, curve_l, curve_r
):
    af = _load_2d(path)
    eq = RangeEQ(sample_rate=af.sr)
    af.y = eq.process(
        af.y,
        freq_start=freq_start,
        freq_end=freq_end,
        total_gain_db=gain,
        peak_pos_pct=peak_pos,
        curve_left=curve_l,
        curve_right=curve_r,
    )
    out_path = _output_path(path, "RangeEQ")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


def process_audio_pitch(audio_path, steps):
    validate_pitch_shift_semitones(steps)
    os.makedirs(_PROC_DIR, exist_ok=True)
    sign = "p" if steps >= 0 else "m"
    out_path = _output_path(audio_path, f"Pitch_{sign}{abs(steps)}")
    pitch_shift_file(audio_path, out_path, steps)
    return out_path


def process_audio_gate(path, db):
    out_path = _output_path(path, f"Gate_{int(db)}dB")
    apply_noise_gate(path, out_path, db)
    return out_path


def process_audio_norm(path):
    out_path = _output_path(path, "Norm")
    apply_normalize(path, out_path)
    return out_path


def process_audio_filter(path, freq, ftype):
    out_path = _output_path(path, f"{ftype.capitalize()}Pass_{freq}Hz")
    apply_filter(path, out_path, filter_type=ftype, cutoff=freq)
    return out_path


def process_audio_reverse(path):
    out_path = _output_path(path, "Rev")
    apply_reverse_audio(path, out_path)
    return out_path


def process_audio_mono(path):
    out_path = _output_path(path, "Mono")
    apply_mono_mix(path, out_path)
    return out_path


def process_transient(path, amt):
    out_path = _output_path(path, "Transient")
    apply_transient_shaper(path, out_path, amt)
    return out_path


def process_slicer(path, threshold):
    folder_name = os.path.splitext(os.path.basename(path))[0] + "_Slices"
    out_dir = os.path.join(_PROC_DIR, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return slice_audio_onsets(path, out_dir, threshold)


def process_parametric_eq(path, freq, gain, q=1.0):
    af = _load_2d(path)
    eq = ParametricEQ(af.sr)
    eq.add_band(freq, gain, q)
    af.y = eq.process(af.y)
    out_path = _output_path(path, "ParamEQ")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


def process_compressor(
    path, threshold=-20, ratio=4.0, attack=0.005, release=0.1
):
    af = _load_2d(path)
    comp = Compressor(af.sr)
    af.y = comp.process(
        af.y, threshold, ratio, attack * 1000.0, release * 1000.0
    )
    out_path = _output_path(path, "Compressed")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


def process_reverb(path, room_size=0.8, damping=0.9, wet_level=0.3):
    af = _load_2d(path)
    reverb = Reverb(af.sr)
    af.y = reverb.process(af.y, room_size, damping, wet_level)
    out_path = _output_path(path, "Reverb")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


def process_delay(path, delay_time=0.5, feedback=0.4, mix=0.3):
    af = _load_2d(path)
    delay_fx = Delay(af.sr)

    af.y = delay_fx.process(af.y, delay_time * 1000.0, feedback, mix)
    out_path = _output_path(path, "Delay")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


def process_chorus(path, rate=1.2, depth=1.4, mix=0.4):
    af = _load_2d(path)
    chorus_fx = Chorus(af.sr)
    af.y = chorus_fx.process(af.y, rate, depth, mix)
    out_path = _output_path(path, "Chorus")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


def process_limiter(path, threshold=-1.0, release=0.05):
    af = _load_2d(path)
    limiter = Limiter(af.sr)
    af.y = limiter.process(af.y, threshold, release * 1000.0)
    out_path = _output_path(path, "Limited")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


def process_phaser(path, rate=0.5, depth=0.7, stages=4, mix=0.5):
    af = _load_2d(path)
    phaser = Phaser(af.sr)
    af.y = phaser.process(af.y, rate, depth, stages, mix)
    out_path = _output_path(path, "Phaser")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


def process_tremolo(path, rate=5.0, depth=0.5, shape="sine"):
    af = _load_2d(path)
    tremolo = Tremolo(af.sr)
    af.y = tremolo.process(af.y, rate, depth, shape)
    out_path = _output_path(path, "Tremolo")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


def process_stereo_widen(path, width=1.5):
    af = _load_2d(path)
    widener = StereoWidener(af.sr)
    af.y = widener.process(af.y, width)
    out_path = _output_path(path, "StereoWide")
    sf.write(out_path, af.to_write_format(), af.sr)
    return out_path


MAX_SIMULTANEOUS_NOTES = 128
REVERB_SETTINGS = (0.4, 0.5, 0.6, 0.25)
CHORUS_SETTINGS = (3, 0.3, 0.3, 8.0, 0)
RENDER_TARGET_RMS_DB = -16.0
CC_PAN = 10
CC_PAN_CENTRE = 64
CC_PAN_RIGHT = 127
DRUM_CHANNEL = 9
CC_REVERB_SEND = 91
CC_CHORUS_SEND = 93
VOICE_PROGRAMS = frozenset({52, 53, 54, 85})


def effect_sends(program, is_drum=False):
    if is_drum:
        return 25, 0
    if program in VOICE_PROGRAMS:
        return 15, 0
    if 32 <= program <= 39:
        return 8, 0
    if 24 <= program <= 31:
        return 25, 10
    return 40, 20


RENDER_LIMIT_KNEE = 0.7


KIT_STANDARD = 0
DRUM_KIT_FALLBACKS = {
    0: (),
    8: (16,),
    16: (8,),
    24: (25, 26),
    25: (24, 26),
    26: (25, 24),
    32: (40,),
    40: (32,),
}


def drum_kit_fallback_order(requested):
    order = [requested]
    for kit in DRUM_KIT_FALLBACKS.get(requested, ()):
        if kit not in order:
            order.append(kit)
    if KIT_STANDARD not in order:
        order.append(KIT_STANDARD)
    return tuple(order)


def resolve_drum_kit(requested, available):
    for kit in drum_kit_fallback_order(requested):
        if kit in available:
            return kit
    return KIT_STANDARD


def read_soundfont_presets(sf3_path):
    presets = set()
    try:
        with open(sf3_path, "rb") as fh:
            if fh.read(4) != b"RIFF":
                return presets
            fh.seek(12)
            while True:
                head = fh.read(8)
                if len(head) < 8:
                    break
                cid, size = struct.unpack("<4sI", head)
                if cid == b"LIST":
                    kind = fh.read(4)
                    end = fh.tell() + size - 4
                    if kind == b"pdta":
                        while fh.tell() < end:
                            sub, sub_size = struct.unpack("<4sI", fh.read(8))
                            data = fh.read(sub_size + (sub_size & 1))
                            if sub == b"phdr":
                                for i in range(sub_size // 38 - 1):
                                    rec = data[i * 38 : (i + 1) * 38]
                                    prog, bank = struct.unpack(
                                        "<HH", rec[20:24]
                                    )
                                    presets.add((bank, prog))
                        return presets
                    fh.seek(end + (size & 1))
                else:
                    fh.seek(size + (size & 1), 1)
    except Exception:
        return presets
    return presets


def read_soundfont_kits(sf3_path):
    return {
        prog for bank, prog in read_soundfont_presets(sf3_path) if bank == 128
    }


def level_render(audio):
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms <= 1e-9:
        return audio
    gain = 10 ** (RENDER_TARGET_RMS_DB / 20.0) / rms
    audio = audio * np.float32(gain)
    knee = RENDER_LIMIT_KNEE
    over = np.abs(audio) > knee
    if np.any(over):
        magnitude = np.abs(audio[over])
        audio[over] = np.sign(audio[over]) * (
            knee + (1.0 - knee) * np.tanh((magnitude - knee) / (1.0 - knee))
        )
    return audio


PITCH_BEND_CENTER = 8192
PITCH_BEND_EVENT = "pitchbend"
PAN_EVENT = "pan"


def _bends_own_their_channel(instrument, batch):
    if not instrument.pitch_bends:
        return False
    if not instrument.is_drum:
        return True
    return sum(1 for other in batch if other.is_drum) == 1


def _with_pitch_bends(note_events, bends):
    return _with_channel_events(note_events, bends, PITCH_BEND_EVENT)


def _with_channel_events(note_events, changes, kind):
    if not changes:
        return note_events
    last = note_events[-1][0]
    waiting = sorted((c for c in changes if c[0] <= last), key=lambda c: c[0])
    merged = []
    at = 0
    for event in note_events:
        while at < len(waiting) and (
            waiting[at][0] < event[0]
            or (waiting[at][0] == event[0] and event[1] == "noteon")
        ):
            time, channel, value = waiting[at]
            merged.append((time, kind, channel, value, 0, -1))
            at += 1
        merged.append(event)
    for time, channel, value in waiting[at:]:
        merged.append((time, kind, channel, value, 0, -1))
    return merged


def pan_moves(instrument):
    return sorted(
        (c for c in instrument.control_changes if c.number == CC_PAN),
        key=lambda change: change.time,
    )


def _resting_side(instrument):
    moves = pan_moves(instrument)
    if not moves:
        return 0.0
    offset = int(moves[0].value) - CC_PAN_CENTRE
    if offset > 0:
        return min(1.0, offset / (CC_PAN_RIGHT - CC_PAN_CENTRE))
    return max(-1.0, offset / CC_PAN_CENTRE)


def _centred(instrument):
    resting = pan_moves(instrument)[0]
    twin = copy.copy(instrument)
    twin.control_changes = [
        change for change in instrument.control_changes if change is not resting
    ]
    return twin


def _placed_part(group, side, settings, render_batch, sr):
    from korshmer.audio_processing import guitar_amp
    from korshmer.audio_processing.stem_session import panned_stereo

    if side:
        group = [_centred(inst) for inst in group]
    part = render_batch(group, level=False)
    if part is None:
        return None
    if settings is not None:
        part = guitar_amp.apply_effect(part, sr, settings)
    return panned_stereo(np.asarray(part, dtype=np.float32), side)


def _sound_name(instrument):
    return instrument.name or f"program {instrument.program}"


FLUSH_BLOCK = 0.1
FLUSH_BLOCKS = 60
FLUSH_SILENCE = 1e-6


class _SharedSynth:
    def __init__(self, fluid_dir, sample_rate, sf3_path):
        self.fluid_dir = fluid_dir
        self.sample_rate = sample_rate
        self.sf3_path = sf3_path
        self.fs = None
        self.sfid = -1

    @contextlib.contextmanager
    def pass_synth(self):
        if self.fs is not None and not hasattr(self.fs, "reset"):
            self.close()
        if self.fs is None:
            synth = FluidSynth(
                lib_dir=self.fluid_dir, samplerate=self.sample_rate
            )
            self.fs = synth.__enter__()
            self.sfid = self.fs.sf_load(self.sf3_path)
        else:
            self.fs.reset()
            self._flush()
        yield self.fs

    def _flush(self):
        block = max(1, int(self.sample_rate * FLUSH_BLOCK))
        for _ in range(FLUSH_BLOCKS):
            left, right = self.fs.get_samples(block)
            loudest = max(
                float(np.max(np.abs(left), initial=0.0)),
                float(np.max(np.abs(right), initial=0.0)),
            )
            if loudest <= FLUSH_SILENCE:
                return

    def close(self):
        if self.fs is not None:
            fs, self.fs = self.fs, None
            fs.__exit__(None, None, None)


def _sits_centre(instrument):
    return all(int(c.value) == CC_PAN_CENTRE for c in pan_moves(instrument))


def _render_with_guitar_amps(instruments, render_batch, sr, bpm=120.0):
    from korshmer.audio_processing import guitar_amp, render_polish

    own_pass = {}
    rest = {}
    drum_side = None
    for inst in instruments:
        settings = None if inst.is_drum else guitar_amp.effect_for(inst)
        side = _resting_side(inst)
        if inst.is_drum and inst.notes and drum_side is None:
            drum_side = side
        if not inst.notes:
            continue
        if inst.is_drum and side != drum_side:
            own_pass.setdefault((None, side), []).append(inst)
        elif settings is not None:
            own_pass.setdefault((settings, side), []).append(inst)
        else:
            key = (render_polish.bus_of(inst), _sits_centre(inst))
            rest.setdefault(key, []).append(inst)
    parts = [
        (bus, centred, render_batch(group, level=False))
        for (bus, centred), group in rest.items()
    ]
    for (settings, side), group in own_pass.items():
        parts.append(
            (
                render_polish.bus_of(group[0]),
                side == 0.0,
                _placed_part(group, side, settings, render_batch, sr),
            )
        )
    kicks = render_polish.kick_times(instruments)
    parts = [
        render_polish.polish_bus(bus, part, sr, bpm, kicks, centred)
        for bus, centred, part in parts
        if part is not None and len(part)
    ]
    if not parts:
        return None
    length = max(len(p) for p in parts)
    mix = np.zeros((length, 2), np.float32)
    for part in parts:
        mix[: len(part)] += part
    return level_render(render_polish.polish_mix(mix, sr))


def render_midi_to_wav(
    midi_path, save_path, fluid_dir, sf3_path, bpm=120.0, stop_callback=None
):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path, initial_tempo=bpm)

        has_notes = False
        for inst in pm.instruments:
            if len(inst.notes) > 0:
                has_notes = True
                break
        if not has_notes:
            print("Warning: MIDI file has no notes.")
            return None
    except Exception as e:
        print(f"Error reading MIDI: {e}")
        return None

    sr = 44100

    from korshmer.audio_processing import render_polish

    available_kits = read_soundfont_kits(sf3_path)
    shared = _SharedSynth(fluid_dir, sr, sf3_path)

    def render_batch(instruments_batch, level=True):
        if not instruments_batch:
            return None
        try:
            with shared.pass_synth() as fs:
                sfid = shared.sfid
                if sfid == -1:
                    print(f"Error: Could not load SoundFont: {sf3_path}")
                    return None

                fs.set_reverb(*REVERB_SETTINGS)
                fs.set_chorus(*CHORUS_SETTINGS)

                melodic_channels = [i for i in range(16) if i != DRUM_CHANNEL]
                melodic_ptr = 0
                note_id = 0
                events = MidiEventQueue()
                bends = []
                pans = []

                for inst in instruments_batch:
                    is_drum = inst.is_drum
                    if is_drum:
                        channel = DRUM_CHANNEL
                    else:
                        if melodic_ptr < len(melodic_channels):
                            channel = melodic_channels[melodic_ptr]
                            melodic_ptr += 1
                        else:
                            logging.warning(
                                "Not rendered: %s. Only %d instruments "
                                "besides the drums can sound at once.",
                                _sound_name(inst),
                                len(melodic_channels),
                            )
                            continue

                    bank = 128 if is_drum else 0
                    program = inst.program
                    if is_drum and available_kits:
                        program = resolve_drum_kit(inst.program, available_kits)
                    fs.program_select(channel, sfid, bank, program)
                    reverb, chorus = effect_sends(program, is_drum)
                    fs.cc(channel, CC_REVERB_SEND, reverb)
                    fs.cc(channel, CC_CHORUS_SEND, chorus)
                    moves = pan_moves(inst)
                    if moves:
                        fs.cc(channel, CC_PAN, int(moves[0].value))
                        pans.extend(
                            (move.time, channel, int(move.value))
                            for move in moves[1:]
                        )
                    if _bends_own_their_channel(inst, instruments_batch):
                        fs.pitch_wheel_sens(channel, bend_range_of(inst))
                        bends.extend(
                            (bend.time, channel, bend.pitch + PITCH_BEND_CENTER)
                            for bend in inst.pitch_bends
                        )
                    played = render_polish.humanized_notes(
                        inst,
                        f"{program}:{channel}:{len(inst.notes)}",
                        keep_time=bool(inst.pitch_bends),
                    )
                    for start, end, pitch, velocity in played:
                        events.push(
                            MidiEvent(
                                start,
                                "noteon",
                                channel,
                                pitch,
                                velocity,
                                note_id,
                            )
                        )
                        events.push(
                            MidiEvent(
                                end,
                                "noteoff",
                                channel,
                                pitch,
                                0,
                                note_id,
                            )
                        )
                        note_id += 1

                if not len(events):
                    return None
                ordered_events = _with_channel_events(
                    _with_pitch_bends(events.sorted_events(), bends),
                    pans,
                    PAN_EVENT,
                )

                audio_left = []
                audio_right = []
                rendered = 0
                check_interval = 200

                active_notes = 0
                dropped_ids = set()
                dropped_count = 0

                for i, (time, etype, chan, note, vel, nid) in enumerate(
                    ordered_events
                ):
                    if (
                        i % check_interval == 0
                        and stop_callback
                        and stop_callback()
                    ):
                        return None
                    samples_needed = int(time * sr) - rendered
                    if samples_needed > 0:
                        l, r = fs.get_samples(samples_needed)
                        audio_left.append(l)
                        audio_right.append(r)
                        rendered += samples_needed

                    if etype == PITCH_BEND_EVENT:
                        fs.pitch_bend(chan, note)
                    elif etype == PAN_EVENT:
                        fs.cc(chan, CC_PAN, note)
                    elif etype == "noteon":
                        if active_notes >= MAX_SIMULTANEOUS_NOTES:
                            dropped_ids.add(nid)
                            dropped_count += 1
                        else:
                            fs.note_on(chan, note, vel)
                            active_notes += 1
                    elif nid in dropped_ids:
                        dropped_ids.discard(nid)
                    else:
                        fs.note_off(chan, note)
                        active_notes -= 1

                if dropped_count:
                    print(
                        f"Warning: skipped {dropped_count} notes above the "
                        f"{MAX_SIMULTANEOUS_NOTES} simultaneous notes limit."
                    )

                tail_samples = int(1.5 * sr)
                if tail_samples > 0:
                    l, r = fs.get_samples(tail_samples)
                    audio_left.append(l)
                    audio_right.append(r)

                if not audio_left:
                    return None

                final_left = np.concatenate(audio_left)
                final_right = np.concatenate(audio_right)
                mixed = np.vstack((final_left, final_right)).T
                return level_render(mixed) if level else mixed
        except Exception as e:
            print(f"FluidSynth Render Error: {e}")
            return None

    instruments = pm.instruments
    if len(instruments) == 0:
        return None

    try:
        final_audio = _render_with_guitar_amps(
            instruments, render_batch, sr, bpm=bpm
        )
    finally:
        shared.close()

    if final_audio is not None:
        save_normalized_audio(save_path, final_audio, sr)
        return save_path
    return None


def render_editor_mix(
    midi_path,
    save_path,
    fluid_dir,
    sf3_path,
    bpm=120.0,
    stop_callback=None,
    clips=None,
):
    active_clips = [c for c in (clips or []) if not c.get("muted")]
    if not active_clips:
        return render_midi_to_wav(
            midi_path,
            save_path,
            fluid_dir,
            sf3_path,
            bpm=bpm,
            stop_callback=stop_callback,
        )

    from korshmer.audio_processing.stem_session import mix_render_with_clips

    render_path = None
    if midi_path and os.path.exists(midi_path) and os.path.exists(sf3_path):
        base, _ = os.path.splitext(save_path)
        render_path = render_midi_to_wav(
            midi_path,
            f"{base}_render.wav",
            fluid_dir,
            sf3_path,
            bpm=bpm,
            stop_callback=stop_callback,
        )
    if stop_callback and stop_callback():
        return None
    try:
        return mix_render_with_clips(render_path, active_clips, save_path)
    finally:
        if render_path and os.path.exists(render_path):
            try:
                os.remove(render_path)
            except OSError:
                pass


def render_piano_roll(midi_path):
    dir_name = os.path.dirname(midi_path)
    base_name = os.path.splitext(os.path.basename(midi_path))[0]
    out_path = os.path.join(dir_name, f"{base_name}_PianoRoll.png")
    return generate_piano_roll_image(midi_path, out_path)
