import logging
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.special import expit

from .validation import validate_effect_integer, validate_effect_number

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH
BIN_LOW = 3
BIN_HIGH = 384
HPSS_KERNEL = 17
MAGNITUDE_FLOOR = 1e-9
TOP_DB = 80.0

WINDOW_S = 2.0
STEP_S = 0.5
PULSE_CONTEXT_S = 6.0
KEY_CONTEXT_S = 4.0
PULSE_LAG_S = (0.25, 1.5)
EDGE_SNAP_S = 2.0
RHYTHM_START = 0.55
RHYTHM_FULL = 0.70
RHYTHM_SCORE = 0.80
BASS_BINS = 16
BASS_START = 0.45
BASS_FULL = 0.60

BEAT_WINDOW_S = 14.0
BEAT_MIN_WINDOW_S = 3.0
BEAT_LAG_S = (0.25, 1.5)
BEAT_MIN_COUNT = 6
BEAT_MIN_PULSE = 0.35
BEAT_REGULARITY = 0.06
BEAT_BINS_PER_FRAME = 2
BEAT_FOCUS_S = 3.0
BEAT_REFINE_SPAN = 0.02
BEAT_REFINE_STEPS = 25
BEAT_MIN_MATCH = 0.6
BEAT_PERIOD_TOLERANCE = 0.04

FEATURES = ("lowE", "hr", "pulse", "flat", "kclar")
COEF = (-7.4074, 10.878, 6.6993, -0.8954, 5.3788)
BIAS = -10.1925
SILENCE_RANGE_DB = 45.0
SILENCE_FLOOR_DB = -80.0

CHUNK_S = 40.0
CHUNK_MARGIN_S = 1.0
LOAD_BLOCK_S = 60
LOAD_OVERLAP_S = 1
BLOCK_FRAMES = 1 << 18

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
KEY_PROFILES = np.array(
    [np.roll(p, r) for p in (MAJOR_PROFILE, MINOR_PROFILE) for r in range(12)]
)
KEY_PROFILES = KEY_PROFILES - KEY_PROFILES.mean(1, keepdims=True)
KEY_PROFILES = KEY_PROFILES / np.linalg.norm(
    KEY_PROFILES, axis=1, keepdims=True
)


@dataclass(frozen=True)
class MusicRegionOptions:
    enabled: bool = True
    min_duration_s: float = 12.0
    enter: float = 0.30
    exit: float = 0.20
    min_music_s: float = 5.0
    bridge_gap_s: float = 8.0
    start_pad_s: float = 0.75
    end_pad_s: float = 1.0
    min_music_share: float = 0.30
    align_beats: bool = True
    smooth_windows: int = 5
    fade_s: float = 0.05
    manual_start_s: float | None = None
    manual_end_s: float | None = None

    def __post_init__(self):
        validate_effect_number(self.min_duration_s, "Minimum duration", 0, 3600)
        validate_effect_number(self.enter, "Enter threshold", 0, 1)
        validate_effect_number(self.exit, "Exit threshold", 0, 1)
        if self.exit > self.enter:
            raise ValueError(
                "Exit threshold must not exceed the enter threshold, "
                f"got {self.exit!r}"
            )
        validate_effect_number(self.min_music_s, "Minimum music length", 0, 600)
        validate_effect_number(self.bridge_gap_s, "Bridge gap", 0, 600)
        validate_effect_number(self.start_pad_s, "Start padding", 0, 30)
        validate_effect_number(self.end_pad_s, "End padding", 0, 30)
        validate_effect_number(
            self.min_music_share, "Minimum music share", 0, 1
        )
        validate_effect_integer(self.smooth_windows, "Smoothing windows", 1, 51)
        validate_effect_number(self.fade_s, "Fade length", 0, 5)
        for name, value in (
            ("Manual start", self.manual_start_s),
            ("Manual end", self.manual_end_s),
        ):
            if value is not None:
                validate_effect_number(value, name, 0, 86400)
        if (
            self.manual_start_s is not None
            and self.manual_end_s is not None
            and self.manual_end_s <= self.manual_start_s
        ):
            raise ValueError(
                "Manual end must be after the manual start, "
                f"got {self.manual_end_s!r}"
            )


@dataclass(frozen=True)
class MusicRegions:
    regions: tuple[tuple[float, float], ...]
    duration: float
    applied: bool
    reason: str
    times: np.ndarray | None = field(default=None, compare=False, repr=False)
    scores: np.ndarray | None = field(default=None, compare=False, repr=False)
    timeline: tuple[tuple[float, float, float], ...] = field(
        default=(), compare=False, repr=False
    )

    def is_identity(self, tolerance: float = 0.5) -> bool:
        if len(self.regions) != 1:
            return False
        start, end = self.regions[0]
        return start <= tolerance and self.duration - end <= tolerance

    def cut_timeline(self) -> tuple[tuple[float, float, float], ...]:
        if self.timeline:
            return self.timeline
        cursor = 0.0
        plain = []
        for start, end in self.regions:
            plain.append((start, end, cursor))
            cursor += end - start
        return tuple(plain)

    def contains(self, t: float) -> bool:
        if not self.applied:
            return True
        return any(start <= t <= end for start, end in self.regions)


def _mono(audio):
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=int(np.argmin(y.shape)))
    return np.ascontiguousarray(y)


def _chunks(y):
    frames = 1 + len(y) // HOP_LENGTH
    core = max(1, int(CHUNK_S * FRAMES_PER_SECOND))
    margin = int(CHUNK_MARGIN_S * FRAMES_PER_SECOND)
    chunks = []
    for first in range(0, frames, core):
        last = min(frames, first + core)
        low = max(0, first - margin)
        high = min(frames, last + margin)
        stop = min(len(y), (high - 1) * HOP_LENGTH + N_FFT // 2 + 1)
        chunks.append((first - low, last - low, y[low * HOP_LENGTH : stop]))
    return chunks


def _magnitude(chunk):
    import librosa

    return np.abs(librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH))


def _signal_statistics(chunks):
    import librosa

    peak = 0.0
    pitches = []
    magnitudes = []
    for first, last, chunk in chunks:
        spectrum = _magnitude(chunk)[:, first:last]
        peak = max(
            peak, float((spectrum[BIN_LOW:BIN_HIGH] + MAGNITUDE_FLOOR).max())
        )
        pitch, magnitude = librosa.piptrack(S=spectrum**2, sr=SAMPLE_RATE)
        voiced = pitch > 0
        pitches.append(pitch[voiced])
        magnitudes.append(magnitude[voiced])
    pitch = np.concatenate(pitches)
    magnitude = np.concatenate(magnitudes)
    tuning = 0.0
    if len(magnitude):
        loud = magnitude >= np.median(magnitude)
        tuning = float(librosa.pitch_tuning(pitch[loud], bins_per_octave=12))
    db_floor = float(librosa.amplitude_to_db(np.float32(peak), top_db=None))
    return db_floor - TOP_DB, tuning


def _chunk_features(chunk, db_floor, tuning):
    import librosa

    spectrum = _magnitude(chunk)
    band = spectrum[BIN_LOW:BIN_HIGH] + MAGNITUDE_FLOOR
    harmonic = median_filter(band, size=(1, HPSS_KERNEL), mode="nearest")
    percussive = median_filter(band, size=(HPSS_KERNEL, 1), mode="nearest")
    share = harmonic**2 / (harmonic**2 + percussive**2)
    power = band**2
    level_db = np.maximum(librosa.amplitude_to_db(band, top_db=None), db_floor)
    return {
        "rms": np.sqrt((spectrum**2).sum(0)) / N_FFT,
        "hr": (share * power).sum(0) / power.sum(0),
        "chroma": librosa.feature.chroma_stft(
            S=spectrum**2,
            sr=SAMPLE_RATE,
            n_chroma=12,
            norm=None,
            tuning=tuning,
        ),
        "flat": np.exp(np.mean(np.log(band), 0)) / np.mean(band, 0),
        "onset": librosa.onset.onset_strength(
            S=level_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        ),
        "bass": librosa.onset.onset_strength(
            S=level_db[:BASS_BINS], sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        ),
    }


def _frame_features(y, on_chunk=None):
    chunks = _chunks(y)
    db_floor, tuning = _signal_statistics(chunks)
    pieces = []
    for index, (first, last, chunk) in enumerate(chunks):
        features = _chunk_features(chunk, db_floor, tuning)
        pieces.append({k: v[..., first:last] for k, v in features.items()})
        if on_chunk is not None:
            on_chunk((index + 1) / len(chunks))
    return {k: np.concatenate([p[k] for p in pieces], -1) for k in pieces[0]}


def _key_clarity(chroma_mean):
    centred = chroma_mean - chroma_mean.mean()
    norm = np.linalg.norm(centred)
    if norm < 1e-9:
        return 0.0
    return float((KEY_PROFILES @ (centred / norm)).max())


def _window_features(frames):
    count = len(frames["rms"])
    window = int(WINDOW_S * FRAMES_PER_SECOND)
    step = int(STEP_S * FRAMES_PER_SECOND)
    pulse_span = int(PULSE_CONTEXT_S * FRAMES_PER_SECOND)
    key_span = int(KEY_CONTEXT_S * FRAMES_PER_SECOND)
    lag_low = int(PULSE_LAG_S[0] * FRAMES_PER_SECOND)
    lag_high = int(PULSE_LAG_S[1] * FRAMES_PER_SECOND)
    rows = []
    for a in range(0, max(1, count - window + 1), step):
        b = a + window
        centre = a + window // 2
        pulse_a = max(0, centre - pulse_span // 2)
        pulse_b = min(count, pulse_a + pulse_span)
        pulse_a = max(0, pulse_b - pulse_span)
        rms = frames["rms"][a:b]
        onset = frames["onset"][pulse_a:pulse_b]
        onset = onset - onset.mean()
        auto = np.correlate(onset, onset, "full")[len(onset) - 1 :]
        auto = auto / (auto[0] + 1e-9)
        key_a = max(0, centre - key_span // 2)
        key_b = min(count, key_a + key_span)
        key_a = max(0, key_b - key_span)
        rows.append(
            (
                a / FRAMES_PER_SECOND,
                20 * np.log10(rms.mean() + 1e-9),
                float((rms < 0.5 * rms.mean()).mean()),
                float(frames["hr"][a:b].mean()),
                float(auto[lag_low:lag_high].max())
                if len(auto) > lag_high
                else 0.0,
                float(np.median(frames["flat"][a:b])),
                _key_clarity(frames["chroma"][:, key_a:key_b].mean(1)),
                _pulse_strength(frames["bass"][pulse_a:pulse_b]),
            )
        )
    columns = np.array(rows).T
    return dict(zip(("start", "db", *FEATURES, "bass"), columns, strict=False))


def _ramp(value, start, full):
    return np.clip((value - start) / (full - start), 0.0, 1.0)


def _score_curve(audio, sample_rate, on_chunk=None):
    import librosa

    validate_effect_number(sample_rate, "Sample rate", 1000, 768000)
    y = _mono(audio)
    if sample_rate != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(y) // HOP_LENGTH < int(WINDOW_S * FRAMES_PER_SECOND):
        return np.zeros(0), np.zeros(0)
    features = _window_features(_frame_features(y, on_chunk))
    logit = BIAS + sum(
        coef * features[name]
        for name, coef in zip(FEATURES, COEF, strict=False)
    )
    level = features["db"]
    quiet = (level < level.max() - SILENCE_RANGE_DB) | (
        level < SILENCE_FLOOR_DB
    )
    rhythm = np.maximum(
        _ramp(features["pulse"], RHYTHM_START, RHYTHM_FULL),
        _ramp(features["bass"], BASS_START, BASS_FULL),
    )
    scores = np.where(
        quiet, 0.0, np.maximum(expit(logit), RHYTHM_SCORE * rhythm)
    )
    return features["start"] + WINDOW_S / 2, scores


def music_score_curve(
    audio: np.ndarray,
    sample_rate: int,
    options: MusicRegionOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    options = options or MusicRegionOptions()
    if not options.enabled:
        return np.zeros(0), np.zeros(0)
    return _score_curve(audio, sample_rate)


def segment_music(
    times,
    scores,
    duration: float,
    options: MusicRegionOptions | None = None,
) -> list[tuple[float, float]]:
    options = options or MusicRegionOptions()
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(times) == 0:
        return []
    scores = median_filter(scores, size=options.smooth_windows, mode="nearest")
    spans = []
    start = None
    for t, score in zip(times, scores, strict=False):
        if start is None:
            if score >= options.enter:
                start = t
        elif score < options.exit:
            spans.append([start, t])
            start = None
    if start is not None:
        spans.append([start, np.inf])
    merged = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= options.bridge_gap_s:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    regions = []
    for begin, end in merged:
        if min(end, times[-1]) - begin < options.min_music_s:
            continue
        begin = max(0.0, begin - WINDOW_S / 2 - options.start_pad_s)
        end = min(duration, end + options.end_pad_s)
        if regions and begin <= regions[-1][1]:
            regions[-1] = (regions[-1][0], float(end))
        else:
            regions.append((float(begin), float(end)))
    if regions and regions[0][0] < EDGE_SNAP_S:
        regions[0] = (0.0, regions[0][1])
    if regions and duration - regions[-1][1] < EDGE_SNAP_S:
        regions[-1] = (regions[-1][0], float(duration))
    return regions


def _onset_envelope(audio):
    import librosa

    return librosa.onset.onset_strength(
        y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )


def _pulse_strength(envelope):
    centred = envelope - envelope.mean()
    auto = np.correlate(centred, centred, "full")[len(centred) - 1 :]
    low = int(PULSE_LAG_S[0] * FRAMES_PER_SECOND)
    high = int(PULSE_LAG_S[1] * FRAMES_PER_SECOND)
    if len(auto) <= high or auto[0] <= 0:
        return 0.0
    return float(auto[low:high].max() / auto[0])


def _beat_period(envelope):
    import librosa

    if len(envelope) < BEAT_MIN_WINDOW_S * FRAMES_PER_SECOND:
        return None
    if _pulse_strength(envelope) < BEAT_MIN_PULSE:
        return None
    _, frames = librosa.beat.beat_track(
        onset_envelope=envelope, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    beats = np.asarray(frames, dtype=float) / FRAMES_PER_SECOND
    if len(beats) < BEAT_MIN_COUNT:
        return None
    gaps = np.diff(beats)
    period = float(np.median(gaps))
    if not BEAT_LAG_S[0] <= period <= BEAT_LAG_S[1]:
        return None
    if np.median(np.abs(gaps - period)) > BEAT_REGULARITY * period:
        return None
    return period


def _folded(envelope, first, period, toward):
    bins = max(8, int(round(period * FRAMES_PER_SECOND * BEAT_BINS_PER_FRAME)))
    times = first + np.arange(len(envelope)) / FRAMES_PER_SECOND
    position = (times % period) / period * bins
    low = np.floor(position).astype(int) % bins
    fraction = position - np.floor(position)
    weights = np.maximum(envelope - envelope.mean(), 0.0)
    weights = weights * np.exp(
        -np.abs(np.arange(len(envelope)) / FRAMES_PER_SECOND - toward)
        / BEAT_FOCUS_S
    )
    profile = np.bincount(low, weights * (1 - fraction), bins)
    profile += np.bincount((low + 1) % bins, weights * fraction, bins)
    spread = np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2)
    padded = np.concatenate([profile[-3:], profile, profile[:3]])
    profile = np.convolve(padded, spread / spread.sum(), "valid")
    profile = profile - profile.mean()
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else None


def _sharpest_period(envelope, first, period, toward):
    best = None
    for scale in np.linspace(
        1 - BEAT_REFINE_SPAN, 1 + BEAT_REFINE_SPAN, BEAT_REFINE_STEPS
    ):
        profile = _folded(envelope, first, period * scale, toward)
        if profile is None:
            continue
        sharpness = float(np.max(profile))
        if best is None or sharpness > best[0]:
            best = (sharpness, period * scale)
    return None if best is None else best[1]


def _side_envelope(audio, begin, end):
    first = int(max(0.0, begin) * SAMPLE_RATE)
    last = int(max(first, end * SAMPLE_RATE))
    segment = audio[first:last]
    if len(segment) < BEAT_MIN_WINDOW_S * SAMPLE_RATE:
        return None
    return _onset_envelope(segment), first / SAMPLE_RATE


def _join_delay(audio, previous, following, options):
    lead = options.end_pad_s + options.fade_s
    tail = options.start_pad_s + WINDOW_S / 2
    stop = previous[1] - lead
    before = _side_envelope(audio, max(previous[0], stop - BEAT_WINDOW_S), stop)
    start = following[0] + tail
    after = _side_envelope(
        audio, start, min(following[1], start + BEAT_WINDOW_S)
    )
    if before is None or after is None:
        return 0.0
    periods = [_beat_period(side[0]) for side in (before, after)]
    if None in periods:
        return 0.0
    if abs(periods[0] - periods[1]) > BEAT_PERIOD_TOLERANCE * periods[0]:
        return 0.0
    period = 0.5 * (periods[0] + periods[1])
    toward = (len(before[0]) / FRAMES_PER_SECOND, 0.0)
    refined = [
        _sharpest_period(side[0], side[1], period, edge)
        for side, edge in zip((before, after), toward, strict=True)
    ]
    if None in refined:
        return 0.0
    period = 0.5 * (refined[0] + refined[1])
    profiles = [
        _folded(side[0], side[1], period, edge)
        for side, edge in zip((before, after), toward, strict=True)
    ]
    if profiles[0] is None or profiles[1] is None:
        return 0.0
    bins = len(profiles[0])
    match = np.array(
        [float(profiles[0] @ np.roll(profiles[1], lag)) for lag in range(bins)]
    )
    lag = int(np.argmax(match))
    if match[lag] < BEAT_MIN_MATCH:
        return 0.0
    drift = lag / bins * period
    return float((drift - (previous[1] - following[0])) % period)


def build_timeline(audio, regions, options=None):
    options = options or MusicRegionOptions()
    audio = _mono(audio)
    timeline = []
    cursor = 0.0
    previous = None
    for begin, end in regions:
        delay = 0.0
        if previous is not None and options.align_beats:
            try:
                delay = _join_delay(audio, previous, (begin, end), options)
            except Exception as exc:
                logger.warning("Beat-aligned join failed: %s", exc)
        timeline.append((float(begin), float(end), cursor + delay))
        cursor += delay + (end - begin)
        previous = (begin, end)
    return timeline


def _result(
    regions,
    duration,
    applied,
    reason,
    times=None,
    scores=None,
    timeline=(),
):
    return MusicRegions(
        tuple((float(a), float(b)) for a, b in regions),
        float(duration),
        applied,
        reason,
        times,
        scores,
        tuple((float(a), float(b), float(c)) for a, b, c in timeline),
    )


def _left_whole(duration, reason, times=None, scores=None):
    whole = [(0.0, duration)] if duration > 0 else []
    return _result(whole, duration, False, reason, times, scores)


def _manual_regions(duration, options):
    start = min(options.manual_start_s or 0.0, duration)
    end = min(
        duration if options.manual_end_s is None else options.manual_end_s,
        duration,
    )
    if end <= start:
        return _left_whole(duration, "manual range is outside the audio")
    result = _result(
        [(start, end)],
        duration,
        True,
        f"kept {start:.1f}-{end:.1f} s of {duration:.1f} s (manual)",
        timeline=[(start, end, 0.0)],
    )
    if result.is_identity():
        return _left_whole(duration, "manual range covers the whole audio")
    return result


def _has_manual_range(options):
    return (
        options.manual_start_s is not None or options.manual_end_s is not None
    )


def _analysis_audio(audio, sample_rate):
    import librosa

    if sample_rate == SAMPLE_RATE:
        return audio
    return librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)


def _find_regions(audio, sample_rate, options, on_chunk=None):
    audio = _mono(audio)
    duration = len(audio) / sample_rate
    if not options.enabled:
        return _left_whole(duration, "music detection is off")
    if _has_manual_range(options):
        return _manual_regions(duration, options)
    if duration < options.min_duration_s:
        return _left_whole(
            duration, f"shorter than {options.min_duration_s:g} s, left whole"
        )
    audio = _analysis_audio(audio, sample_rate)
    times, scores = _score_curve(audio, SAMPLE_RATE, on_chunk)
    regions = segment_music(times, scores, duration, options)
    kept = sum(end - start for start, end in regions)
    if not regions:
        return _result([], duration, False, "no music found", times, scores)
    if kept < options.min_music_share * duration:
        share = f"{kept / duration:.0%}"
        reason = f"only {share} of the audio looks like music, left whole"
        return _result([], duration, False, reason, times, scores)
    found = _result(regions, duration, True, "", times, scores)
    if found.is_identity():
        return _left_whole(duration, "all of the audio is music", times, scores)
    if len(regions) == 1:
        where = f"{regions[0][0]:.1f}-{regions[0][1]:.1f} s"
    else:
        where = f"{len(regions)} regions, {kept:.1f} s"
    reason = f"kept {where} of {duration:.1f} s"
    timeline = build_timeline(audio, regions, options)
    return _result(regions, duration, True, reason, times, scores, timeline)


def find_music_regions(
    audio: np.ndarray,
    sample_rate: int,
    options: MusicRegionOptions | None = None,
) -> MusicRegions:
    options = options or MusicRegionOptions()
    try:
        return _find_regions(audio, sample_rate, options)
    except Exception as exc:
        logger.warning("Music detection failed: %s", exc)
        return _result([], 0.0, False, f"music detection failed: {exc}")


def _load_mono(path):
    import librosa

    total = float(librosa.get_duration(path=str(path)))
    parts = []
    for start in range(0, int(np.ceil(total)), LOAD_BLOCK_S):
        lead = min(start, LOAD_OVERLAP_S)
        part, _ = librosa.load(
            str(path),
            sr=SAMPLE_RATE,
            mono=True,
            offset=start - lead,
            duration=lead + LOAD_BLOCK_S + LOAD_OVERLAP_S,
        )
        parts.append(
            part[lead * SAMPLE_RATE : (lead + LOAD_BLOCK_S) * SAMPLE_RATE]
        )
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def find_music_regions_in_file(
    path: str,
    options: MusicRegionOptions | None = None,
    progress=None,
) -> MusicRegions:
    options = options or MusicRegionOptions()

    def report(fraction, text):
        if progress is not None:
            progress(fraction, text)

    try:
        if not options.enabled:
            return _result([], 0.0, False, "music detection is off")
        if _has_manual_range(options):
            import librosa

            duration = float(librosa.get_duration(path=str(path)))
            return _manual_regions(duration, options)
        report(0.0, "Looking for music")
        audio = _load_mono(path)
        report(0.1, "Looking for music")

        def on_chunk(fraction):
            report(0.1 + 0.85 * fraction, "Looking for music")

        result = _find_regions(audio, SAMPLE_RATE, options, on_chunk)
        report(1.0, result.reason)
        return result
    except Exception as exc:
        logger.warning("Music detection failed for %s: %s", path, exc)
        return _result([], 0.0, False, f"music detection failed: {exc}")


def _cut_gain(count, first, total, fade, fade_in, fade_out):
    index = np.arange(first, first + count)
    gain = np.ones(count)
    if fade:
        if fade_in:
            gain = np.minimum(gain, np.clip((index + 0.5) / fade, 0.0, 1.0))
        if fade_out:
            gain = np.minimum(
                gain, np.clip((total - index - 0.5) / fade, 0.0, 1.0)
            )
        gain = 0.5 * (1 - np.cos(np.pi * gain))
    return gain.astype(np.float32)[:, None]


def _cut_spans(timeline, rate, total):
    spans = []
    cursor = 0
    for begin, end, position in timeline:
        first = min(total, max(0, round(begin * rate)))
        last = min(total, max(first, round(end * rate)))
        at = max(cursor, round(position * rate))
        spans.append((first, last, at, begin > 0, last < total))
        cursor = at + (last - first)
    return spans


def cut_outside_regions(
    src: str, dst: str, timeline, fade_s: float = 0.05
) -> str:
    validate_effect_number(fade_s, "Fade length", 0, 5)
    try:
        reader = sf.SoundFile(src)
    except RuntimeError:
        return _cut_decoded(src, dst, timeline, fade_s)
    with reader:
        rate = reader.samplerate
        fade = round(fade_s * rate)
        spans = _cut_spans(timeline, rate, reader.frames)
        with sf.SoundFile(
            dst,
            "w",
            samplerate=rate,
            channels=reader.channels,
            subtype="FLOAT",
            format="WAV",
        ) as writer:
            written = 0
            for first, last, at, fade_in, fade_out in spans:
                if at > written:
                    writer.write(
                        np.zeros((at - written, reader.channels), np.float32)
                    )
                    written = at
                reader.seek(first)
                done = 0
                total = last - first
                while done < total:
                    block = reader.read(
                        min(BLOCK_FRAMES, total - done),
                        dtype="float32",
                        always_2d=True,
                    )
                    if not len(block):
                        break
                    gain = _cut_gain(
                        len(block), done, total, fade, fade_in, fade_out
                    )
                    writer.write((block * gain).astype(np.float32, copy=False))
                    done += len(block)
                    written += len(block)
    return dst


def _cut_decoded(src, dst, timeline, fade_s):
    import librosa

    data, rate = librosa.load(src, sr=None, mono=False)
    block = np.atleast_2d(data).T
    fade = round(fade_s * rate)
    pieces = []
    written = 0
    for first, last, at, fade_in, fade_out in _cut_spans(
        timeline, rate, len(block)
    ):
        if at > written:
            pieces.append(np.zeros((at - written, block.shape[1]), np.float32))
            written = at
        part = block[first:last]
        gain = _cut_gain(len(part), 0, len(part), fade, fade_in, fade_out)
        pieces.append((part * gain).astype(np.float32, copy=False))
        written += len(part)
    if not pieces:
        pieces = [np.zeros((0, block.shape[1]), np.float32)]
    sf.write(dst, np.concatenate(pieces), rate, subtype="FLOAT", format="WAV")
    return dst
