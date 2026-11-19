import hashlib
import os
import re
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

from korshmer.audio_processing import clip_level, clip_tempo
from korshmer.constants import DATA_DIR

POLISHED_FOLDER = "clip_edits"
PULSE_HOP = clip_tempo.PULSE_HOP
PULSE_STEPS_PER_BEAT = clip_tempo.PULSE_STEPS_PER_BEAT
PULSE_CERTAINTY = 2.6
COARSER_PULSE_SHARE = 0.7
FIRST_SOUND_SHARE = 0.1
SILENCE_LEVEL = 1e-4
ON_PULSE = "pulse"
ON_FIRST_SOUND = "first sound"
SILENT = "silent"
MIN_PULSE_SECONDS = 1.0
NOISE_TAG = "clean"
QUALITY_TAG = "improved"
TEMPO_TAG = "tempo"
NOISE = "noise"
QUALITY = "quality"
ALIGN = "align"
EARLIER_TAGS = re.compile(
    rf"(_({NOISE_TAG}|{QUALITY_TAG}|{TEMPO_TAG}[0-9.]*)(_[0-9a-f]{{8}})?)+$"
)
NOISE_STAGE = "Removing background noise"
QUALITY_STAGE = "Measuring and improving the level"
NOTES_STAGE = "Listening to the notes of the song"
TEMPO_STAGE = "Comparing the tempo with the song"
STRETCH_STAGE = "Matching the tempo of the song"
BEAT_STAGE = "Finding the beat"
FOLLOW_STAGE = "Stretching the clip to the new tempo"
SAME_TEMPO_SHARE = 1e-4
SLOWEST_FOLLOW = 0.5
FASTEST_FOLLOW = 2.0
PARTS_STAGE = "Looking for separate parts"
MIN_PART_GAP_S = 1.5
MIN_PART_S = 0.3
PART_EDGE_S = 0.05
PART_BELOW_LOUDEST_DB = 40.0
PART_SILENT_DB = -120.0
PART_LOUD_PERCENTILE = 98.0
PART_PAUSE_PERCENTILE = 10.0
PART_STEADY_PAUSE_DB = 8.0
PART_MIN_CONTRAST_DB = 20.0
PART_RISE_OVER_PAUSE_DB = 8.0
MIN_TRIM_S = 0.1
PART_MIN_HISS_CONTRAST_DB = 6.0
PART_HISS_FLATNESS = 0.1
PART_TONE_SHARE_OF_HISS = 0.5
SHORT_PART_S = 2.0
ON_NEIGHBOUR = "nearby part"
PART_FRAME = 1024


class ClipFixCancelled(Exception):
    pass


@dataclass(frozen=True)
class ClipAudio:
    path: str
    start: float = 0.0
    offset: float = 0.0
    duration: float | None = None
    source_path: str | None = None


def fixed_clip_path(path, tag, folder=None):
    folder = folder or os.path.join(str(DATA_DIR), POLISHED_FOLDER)
    os.makedirs(folder, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0]
    stat = os.stat(path)
    key = f"{os.path.abspath(path)}|{stat.st_mtime_ns}|{stat.st_size}|{tag}"
    mark = hashlib.sha1(key.encode()).hexdigest()[:8]
    return os.path.join(
        folder, f"{EARLIER_TAGS.sub('', name)}_{tag}_{mark}.wav"
    )


def _read(path):
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return data, int(sample_rate)


def _write(path, data, sample_rate):
    partial = f"{path}.part"
    sf.write(partial, data, sample_rate, subtype="FLOAT", format="WAV")
    os.replace(partial, path)
    return path


def _kept_part(data, sample_rate, offset, duration):
    audio = np.asarray(data, dtype=np.float32)
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    first = max(0, int(round(float(offset) * sample_rate)))
    last = mono.shape[0]
    if duration is not None:
        last = min(last, first + int(round(float(duration) * sample_rate)))
    return np.nan_to_num(mono[first:last], nan=0.0, posinf=0.0, neginf=0.0)


def _beats_of(frames, sample_rate, beats_at, start):
    seconds = float(start) + frames * PULSE_HOP / sample_rate
    return np.array([beats_at(second) for second in seconds], dtype=np.float64)


def _pulse_shift(mono, sample_rate, beats_at, start):
    if mono.shape[0] < MIN_PULSE_SECONDS * sample_rate:
        return None
    seconds, weights = clip_tempo.onsets(mono, sample_rate)
    spread = float(np.sqrt(np.sum(weights**2)))
    if seconds.shape[0] == 0 or spread <= 0.0:
        return None
    beats = np.array(
        [beats_at(float(start) + second) for second in seconds],
        dtype=np.float64,
    )
    pulses = [
        (np.sum(weights * np.exp(2j * np.pi * steps * beats)), steps)
        for steps in PULSE_STEPS_PER_BEAT
    ]
    strongest = max(abs(pulse) for pulse, _steps in pulses)
    if strongest / spread < PULSE_CERTAINTY:
        return None
    pulse, steps = next(
        found
        for found in pulses
        if abs(found[0]) >= COARSER_PULSE_SHARE * strongest
    )
    return float(-np.angle(pulse) / (2.0 * np.pi * steps))


def _first_sound_shift(mono, sample_rate, beats_at, start):
    if mono.shape[0] < PULSE_HOP:
        return None
    levels = librosa.feature.rms(
        y=mono, frame_length=2 * PULSE_HOP, hop_length=PULSE_HOP, center=False
    )[0]
    peak = float(levels.max()) if levels.size else 0.0
    if peak <= SILENCE_LEVEL:
        return None
    heard = np.flatnonzero(levels >= peak * FIRST_SOUND_SHARE)
    beat = _beats_of(heard[:1], sample_rate, beats_at, start)[0]
    return float(np.round(beat) - beat)


def clip_beat_shift(
    data, sample_rate, beats_at, start=0.0, offset=0.0, duration=None
):
    mono = _kept_part(data, sample_rate, offset, duration)
    shift = _pulse_shift(mono, sample_rate, beats_at, start)
    if shift is not None:
        return shift, ON_PULSE
    shift = _first_sound_shift(mono, sample_rate, beats_at, start)
    if shift is not None:
        return shift, ON_FIRST_SOUND
    return None, SILENT


def clip_own_tempo(data, sample_rate, song_bpm, offset=0.0, duration=None):
    mono = _kept_part(data, sample_rate, offset, duration)
    if mono.shape[0] < MIN_PULSE_SECONDS * sample_rate:
        return None
    seconds, weights = clip_tempo.onsets(mono, sample_rate)
    return clip_tempo.own_tempo(seconds, weights, float(song_bpm))


def remove_noise_file(path, folder=None):
    from korshmer.audio_processing.vocal_cleanup import clean_vocal

    data, sample_rate = _read(path)
    return _write(
        fixed_clip_path(path, NOISE_TAG, folder),
        clean_vocal(data, sample_rate),
        sample_rate,
    )


def improve_quality_file(path, folder=None, notes=None, origin=0.0):
    from korshmer.audio_processing.clip_notes import beside_clip

    data, sample_rate = _read(path)
    heard = None if notes is None else notes(sample_rate)
    music = (
        None
        if heard is None
        else beside_clip(heard, sample_rate, origin, data.shape[0])
    )
    improved, readings = clip_level.improve(data, sample_rate, music)
    if not readings or clip_level.unchanged(readings):
        return path, readings
    target = fixed_clip_path(path, QUALITY_TAG, folder)
    return _write(target, improved, sample_rate), readings


def follow_tempo_file(path, recorded_bpm, song_bpm, folder=None):
    if abs(recorded_bpm - song_bpm) / song_bpm < SAME_TEMPO_SHARE:
        info = sf.info(path)
        return path, info.frames / info.samplerate
    return match_tempo_file(
        path, recorded_bpm, song_bpm, folder, f"{TEMPO_TAG}{song_bpm:.2f}"
    )


def match_tempo_file(path, clip_bpm, song_bpm, folder=None, tag=TEMPO_TAG):
    from korshmer.audio_processing.tempo_align import stretch_to_tempo
    from korshmer.midi.tempo_map import TempoMap

    data, sample_rate = _read(path)
    matched = stretch_to_tempo(
        data.T, sample_rate, 0.0, TempoMap.uniform(clip_bpm), song_bpm
    )
    target = fixed_clip_path(path, tag, folder)
    _write(target, matched.T, sample_rate)
    return target, matched.shape[1] / sample_rate


def _stages(steps, clip, song_bpm, notes=None):
    paths = 1 + (clip.source_path is not None)
    stages = []
    if NOISE in steps:
        stages += [NOISE_STAGE] * paths
    if QUALITY in steps:
        stages += [NOTES_STAGE] * (notes is not None)
        stages += [QUALITY_STAGE] * paths
    if ALIGN in steps:
        if song_bpm is not None and clip.source_path is None:
            stages += [TEMPO_STAGE, STRETCH_STAGE]
        stages.append(BEAT_STAGE)
    return stages


def fix_clip(
    clip,
    steps,
    beats_at,
    song_bpm=None,
    report=None,
    cancelled=None,
    folder=None,
    notes=None,
):
    stages = _stages(steps, clip, song_bpm, notes)
    made = []
    done = 0

    def enter(stage):
        nonlocal done
        if cancelled is not None and cancelled():
            raise ClipFixCancelled(stage)
        if report is not None:
            report(stage, done, len(stages))
        done += 1

    def kept(path, origin):
        if path != origin:
            made.append(path)
        return path

    result = {"steps": tuple(steps), "path": clip.path}
    current = {"path": clip.path, "source": clip.source_path}
    try:
        for step in (NOISE, QUALITY):
            if step not in steps:
                continue
            if step == QUALITY and notes is not None:
                enter(NOTES_STAGE)
                notes(sf.info(current["path"]).samplerate)
            for role in ("path", "source"):
                if current[role] is None:
                    continue
                if step == NOISE:
                    enter(NOISE_STAGE)
                    fixed = remove_noise_file(current[role], folder)
                else:
                    enter(QUALITY_STAGE)
                    fixed, readings = improve_quality_file(
                        current[role],
                        folder,
                        notes if role == "path" else None,
                        clip.start - clip.offset,
                    )
                    if role == "path":
                        result["readings"] = readings
                current[role] = kept(fixed, current[role])
        if ALIGN in steps:
            _align(
                clip, current, result, beats_at, song_bpm, enter, kept, folder
            )
    except Exception:
        _remove(made)
        raise
    result["path"] = current["path"]
    if current["source"] is not None:
        result["source"] = current["source"]
    finals = {current["path"], current["source"]}
    _remove([path for path in made if path not in finals])
    result["made"] = [path for path in made if path in finals]
    return result


def _align(clip, current, result, beats_at, song_bpm, enter, kept, folder):
    offset, duration = clip.offset, clip.duration
    data, sample_rate = _read(current["path"])
    if song_bpm is not None and clip.source_path is None:
        enter(TEMPO_STAGE)
        found = clip_own_tempo(data, sample_rate, song_bpm, offset, duration)
        enter(STRETCH_STAGE)
        if found is not None:
            clip_bpm, _certainty = found
            matched, length = match_tempo_file(
                current["path"], clip_bpm, song_bpm, folder
            )
            current["path"] = kept(matched, current["path"])
            ratio = clip_bpm / float(song_bpm)
            offset *= ratio
            duration = None if duration is None else duration * ratio
            result["tempo"] = (float(clip_bpm), float(song_bpm))
            result["length"] = float(length)
            data, sample_rate = _read(matched)
    enter(BEAT_STAGE)
    result["shift"], result["anchor"] = clip_beat_shift(
        data, sample_rate, beats_at, clip.start, offset, duration
    )


def _frame_levels_db(mono):
    levels = librosa.feature.rms(
        y=mono, frame_length=PART_FRAME, hop_length=PART_FRAME, center=False
    )[0]
    return 20.0 * np.log10(np.maximum(levels, 10.0 ** (PART_SILENT_DB / 20.0)))


def _frame_flatness(mono):
    return librosa.feature.spectral_flatness(
        y=mono, n_fft=PART_FRAME, hop_length=PART_FRAME, center=False
    )[0]


def _hiss_between_tones(levels, flatness, pause, loud):
    frames = min(levels.shape[0], flatness.shape[0])
    levels, flatness = levels[:frames], flatness[:frames]
    quiet = flatness[levels <= pause]
    sounding = flatness[levels >= loud - PART_RISE_OVER_PAUSE_DB]
    if quiet.size == 0 or sounding.size == 0:
        return False
    hiss = float(np.median(quiet))
    return hiss >= PART_HISS_FLATNESS and float(np.median(sounding)) <= (
        hiss * PART_TONE_SHARE_OF_HISS
    )


def _sound_threshold_db(levels, flatness=None):
    loud = float(np.percentile(levels, PART_LOUD_PERCENTILE))
    lowest = loud - PART_BELOW_LOUDEST_DB
    pauses = np.maximum(
        levels[levels <= np.percentile(levels, PART_PAUSE_PERCENTILE)], lowest
    )
    floor = float(np.max(pauses))
    steady = floor - float(np.min(pauses)) <= PART_STEADY_PAUSE_DB
    contrast = loud - floor
    if not steady:
        return lowest
    if contrast >= PART_MIN_CONTRAST_DB:
        return floor + min(PART_RISE_OVER_PAUSE_DB, contrast / 2.0)
    if (
        flatness is not None
        and contrast >= PART_MIN_HISS_CONTRAST_DB
        and _hiss_between_tones(levels, flatness, floor, loud)
    ):
        return floor + contrast / 2.0
    return lowest


def _sound_spans(heard, frame):
    spans = []
    for index in np.flatnonzero(heard):
        begin, end = index * frame, (index + 1) * frame
        if spans and begin - spans[-1][1] < MIN_PART_GAP_S:
            spans[-1][1] = end
        else:
            spans.append([begin, end])
    return [span for span in spans if span[1] - span[0] >= MIN_PART_S]


def clip_parts(data, sample_rate, offset=0.0, duration=None):
    mono = _kept_part(data, sample_rate, offset, duration)
    if mono.shape[0] < PART_FRAME:
        return []
    levels = _frame_levels_db(mono)
    if float(levels.max()) <= 20.0 * np.log10(SILENCE_LEVEL):
        return []
    heard = levels > _sound_threshold_db(levels, _frame_flatness(mono))
    whole = mono.shape[0] / sample_rate
    return [
        (
            float(offset) + max(0.0, begin - PART_EDGE_S),
            float(offset) + min(whole, end + PART_EDGE_S),
        )
        for begin, end in _sound_spans(heard, PART_FRAME / sample_rate)
    ]


def _fix_trimmed(clip, span, length, whole, options):
    begin, end = span
    kept_end = length if clip.duration is None else clip.offset + clip.duration
    if begin - clip.offset < MIN_TRIM_S and kept_end - end < MIN_TRIM_S:
        return whole()
    trimmed = ClipAudio(
        clip.path,
        clip.start + begin - clip.offset,
        begin,
        end - begin,
        clip.source_path,
    )
    result = fix_clip(trimmed, *options)
    result["trim"] = (trimmed.start, trimmed.start + end - begin)
    return result


def _short_parts_follow_neighbours(parts):
    steady = [part for part in parts if part.get("anchor") == ON_PULSE]
    for part in parts:
        short = part["seconds"] < SHORT_PART_S
        if not steady or not short or part.get("anchor") != ON_FIRST_SOUND:
            continue
        nearest = min(
            steady, key=lambda other: abs(other["start"] - part["start"])
        )
        part["shift"] = nearest["shift"]
        part["anchor"] = ON_NEIGHBOUR


def fix_clip_parts(
    clip,
    steps,
    beats_at,
    song_bpm=None,
    report=None,
    cancelled=None,
    folder=None,
    notes=None,
):
    def whole():
        return fix_clip(
            clip, steps, beats_at, song_bpm, report, cancelled, folder, notes
        )

    if clip.source_path is not None:
        return whole()
    if report is not None:
        report(PARTS_STAGE, 0, 1)
    data, sample_rate = _read(clip.path)
    spans = clip_parts(data, sample_rate, clip.offset, clip.duration)
    if len(spans) == 1:
        return _fix_trimmed(
            clip,
            spans[0],
            data.shape[0] / sample_rate,
            whole,
            (steps, beats_at, song_bpm, report, cancelled, folder, notes),
        )
    if len(spans) < 2:
        return whole()
    made = []
    parts = []
    try:
        for number, (begin, end) in enumerate(spans, 1):
            cut = fixed_clip_path(clip.path, f"part{number}", folder)
            first, last = int(begin * sample_rate), int(end * sample_rate)
            _write(cut, data[first:last], sample_rate)
            made.append(cut)
            piece = ClipAudio(cut, clip.start + begin - clip.offset)

            def part_report(stage, done, total, number=number):
                if report is not None:
                    report(
                        f"Part {number} of {len(spans)}: {stage}",
                        (number - 1) * total + done,
                        len(spans) * total,
                    )

            fixed = fix_clip(
                piece,
                steps,
                beats_at,
                song_bpm,
                part_report,
                cancelled,
                folder,
                notes,
            )
            fixed["start"] = piece.start
            fixed["seconds"] = end - begin
            made.extend(fixed["made"])
            if fixed["path"] != cut:
                _remove([cut])
                made.remove(cut)
            parts.append(fixed)
    except Exception:
        _remove(made)
        raise
    _short_parts_follow_neighbours(parts)
    return {
        "steps": tuple(steps),
        "path": clip.path,
        "parts": parts,
        "made": made,
    }


def _remove(paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            continue
