import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from korshmer.audio_processing import music_regions
from korshmer.audio_processing.music_regions import (
    MusicRegionOptions,
    MusicRegions,
    build_timeline,
    cut_outside_regions,
    find_music_regions,
    find_music_regions_in_file,
    music_score_curve,
    segment_music,
)
from tests.fixtures.drum_loops import (
    SAMPLE_RATE as DRUM_RATE,
    make_drum_loop,
)

SR = 22050
INTRO_S = 10.0
BODY_S = 24.0
OUTRO_S = 8.0
EDGE_TOLERANCE_S = 2.5


def speech_like(seconds, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * SR))
    pos = 0.0
    while pos < seconds - 0.8:
        for _ in range(int(rng.integers(2, 6))):
            n = int(rng.uniform(0.09, 0.26) * SR)
            start = int(pos * SR)
            if start + n >= len(out):
                break
            t = np.arange(n) / SR
            f0 = rng.uniform(85, 260)
            glide = f0 * (1 + rng.uniform(-0.25, 0.25) * t * SR / n)
            phase = 2 * np.pi * np.cumsum(glide) / SR
            f1, f2 = rng.uniform(300, 900), rng.uniform(1000, 2800)
            noise = rng.standard_normal(n)
            voice = np.zeros(n)
            for k in range(1, 40):
                if k * f0 > 5000:
                    break
                weight = (
                    np.exp(-(((k * f0 - f1) / 200) ** 2))
                    + 0.6 * np.exp(-(((k * f0 - f2) / 350) ** 2))
                    + 0.03
                )
                voice += weight * np.sin(k * phase)
            if rng.random() < 0.4:
                voice = 1.5 * np.diff(noise, prepend=0.0)
            else:
                voice = voice + 0.3 * noise
            out[start : start + n] += voice * np.hanning(n)
            pos += n / SR + rng.uniform(0.0, 0.06)
        pos += rng.uniform(0.3, 0.8)
    return 0.3 * out / np.abs(out).max()


def fx_whoosh(seconds, seed):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    hit = rng.standard_normal(n) * np.exp(-t * 6)
    swell = rng.standard_normal(n) * np.hanning(n) ** 2
    return 0.25 * (hit + swell) / np.abs(hit + swell).max()


def _harmonic_tone(freq, count, harmonics, rolloff):
    t = np.arange(count) / SR
    return sum(
        np.sin(2 * np.pi * k * freq * t) / k**rolloff
        for k in range(1, harmonics + 1)
    )


def music_body(seconds, seed=2):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    chords = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]
    beat = 0.5
    bar = int(4 * beat * SR)
    for index in range(int(seconds / (4 * beat)) + 1):
        start = index * bar
        chord = chords[index % 4]
        length = bar + int(0.3 * SR)
        ramp = np.arange(length)
        env = np.minimum(1, ramp / (0.03 * SR)) * np.minimum(
            1, (length - ramp) / (0.2 * SR)
        )
        for note in chord:
            freq = 440 * 2 ** ((note - 69) / 12)
            pad = _harmonic_tone(freq, length, 8, 1.2) * env * 0.12
            stop = min(n, start + length)
            out[start:stop] += pad[: stop - start]
        bass = _harmonic_tone(440 * 2 ** ((chord[0] - 93) / 12), bar, 4, 1.0)
        stop = min(n, start + bar)
        out[start:stop] += 0.25 * bass[: stop - start]
        for step in range(4):
            hit = start + int(step * beat * SR)
            m = int(0.15 * SR)
            if hit + m >= n:
                continue
            t = np.arange(m) / SR
            if step % 2 == 0:
                sweep = np.sin(2 * np.pi * (60 + 120 * np.exp(-t * 30)) * t)
                out[hit : hit + m] += 0.5 * sweep * np.exp(-t * 25)
            else:
                snare = rng.standard_normal(m) * np.exp(-t * 30)
                out[hit : hit + m] += 0.3 * snare
        for step in range(8):
            hit = start + int(step * beat / 2 * SR)
            m = int(0.04 * SR)
            if hit + m < n:
                tick = rng.standard_normal(m) * np.exp(-np.arange(m) / 220)
                out[hit : hit + m] += 0.08 * tick
    return 0.5 * out / np.abs(out).max()


def _scene():
    intro = np.concatenate([fx_whoosh(2.0, 5), speech_like(INTRO_S - 2.0, 1)])
    outro = speech_like(OUTRO_S, 3)
    audio = np.concatenate([intro, music_body(BODY_S), outro])
    return audio.astype(np.float32)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    path = tmp_path_factory.mktemp("music_regions") / "scene.wav"
    audio = _scene()
    sf.write(path, audio, SR, subtype="FLOAT")
    return audio, str(path), find_music_regions_in_file(str(path))


def test_intro_and_outro_speech_are_cut_and_music_is_kept(scene, tmp_path):
    audio, path, found = scene
    assert found.applied
    assert len(found.regions) == 1
    start, end = found.regions[0]
    body_end = INTRO_S + BODY_S
    assert INTRO_S - EDGE_TOLERANCE_S <= start <= INTRO_S
    assert body_end <= end <= body_end + EDGE_TOLERANCE_S
    assert found.duration == pytest.approx(len(audio) / SR, abs=0.01)

    assert found.cut_timeline() == ((start, end, 0.0),)
    cut_path = cut_outside_regions(
        path, str(tmp_path / "cut.wav"), found.cut_timeline()
    )
    cut, rate = sf.read(cut_path, dtype="float32")
    assert rate == SR
    first, last = round(start * SR), round(end * SR)
    assert len(cut) == last - first
    fade = round(0.05 * SR)
    assert np.array_equal(cut[fade:-fade], audio[first + fade : last - fade])


def test_pure_music_is_left_whole():
    found = find_music_regions(music_body(BODY_S), SR)
    assert not found.applied
    assert found.is_identity()
    assert found.contains(5.0)


def test_speech_only_is_not_applied():
    found = find_music_regions(speech_like(20.0, 4), SR)
    assert not found.applied
    assert found.regions == ()
    assert found.contains(5.0)


def test_silence_is_not_applied():
    found = find_music_regions(np.zeros(20 * SR, dtype=np.float32), SR)
    assert not found.applied
    assert found.regions == ()


def test_short_input_is_left_whole():
    found = find_music_regions(music_body(8.0), SR)
    assert not found.applied
    assert found.is_identity()
    assert "shorter" in found.reason


def test_disabled_detector_leaves_the_audio_whole():
    options = MusicRegionOptions(enabled=False)
    found = find_music_regions(_scene(), SR, options)
    assert not found.applied
    assert found.is_identity()
    times, scores = music_score_curve(_scene(), SR, options)
    assert len(times) == len(scores) == 0


def test_manual_start_end_override_wins(scene, monkeypatch):
    audio, path, _ = scene

    def fail(*_args, **_kwargs):
        raise AssertionError("the detector must not run")

    monkeypatch.setattr(music_regions, "_score_curve", fail)
    options = MusicRegionOptions(manual_start_s=4.0, manual_end_s=30.0)
    for found in (
        find_music_regions(audio, SR, options),
        find_music_regions_in_file(path, options),
    ):
        assert found.applied
        assert found.regions == ((4.0, 30.0),)
        assert not found.contains(3.9)
        assert found.contains(4.0)
    only_start = find_music_regions(
        audio, SR, MusicRegionOptions(manual_start_s=4.0)
    )
    assert only_start.regions == ((4.0, only_start.duration),)
    whole = find_music_regions(
        audio, SR, MusicRegionOptions(manual_start_s=0.0)
    )
    assert not whole.applied
    assert whole.is_identity()
    short = find_music_regions(
        audio[: 8 * SR], SR, MusicRegionOptions(manual_start_s=2.0)
    )
    assert short.applied
    assert short.regions == ((2.0, short.duration),)


def test_chunked_scores_equal_unchunked(monkeypatch):
    audio = np.concatenate(
        [speech_like(40.0, 6), music_body(100.0), speech_like(40.0, 7)]
    ).astype(np.float32)
    monkeypatch.setattr(music_regions, "CHUNK_S", 1e6)
    assert len(music_regions._chunks(audio)) == 1
    times, whole = music_score_curve(audio, SR)
    monkeypatch.setattr(music_regions, "CHUNK_S", 25.0)
    assert len(music_regions._chunks(audio)) > 5
    chunk_times, chunked = music_score_curve(audio, SR)
    assert len(times) > 300
    assert np.array_equal(times, chunk_times)
    assert np.allclose(chunked, whole, atol=1e-3)
    assert whole.min() < 0.1 < 0.9 < whole.max()


def test_scores_accept_other_rates_and_channel_layouts():
    audio = music_body(14.0)
    wide = resample_poly(audio, 2, 1)
    pair = np.stack([wide, wide])
    _, rows = music_score_curve(pair, 44100)
    _, columns = music_score_curve(pair.T, 44100)
    _, native = music_score_curve(audio, SR)
    assert np.array_equal(rows, columns)
    assert np.allclose(rows, native, atol=0.05)


def test_find_music_regions_in_file_never_raises(tmp_path):
    garbage = tmp_path / "garbage.wav"
    garbage.write_bytes(b"not audio at all" * 100)
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    for path in (tmp_path / "missing.wav", garbage, empty, tmp_path):
        found = find_music_regions_in_file(str(path))
        assert isinstance(found, MusicRegions)
        assert not found.applied
        assert found.reason


def test_find_music_regions_in_file_reports_progress(tmp_path):
    path = tmp_path / "music.wav"
    sf.write(path, music_body(14.0).astype(np.float32), SR)
    calls = []
    found = find_music_regions_in_file(
        str(path),
        progress=lambda fraction, text: calls.append((fraction, text)),
    )
    assert not found.applied
    fractions = [fraction for fraction, _ in calls]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert calls[-1][1] == found.reason


def test_segment_music_hysteresis_bridges_short_gaps_and_drops_islands():
    times = np.arange(0, 100, 0.5) + 1.0
    scores = np.full(len(times), 0.05)

    def put(first_s, last_s, value):
        scores[(times >= first_s) & (times < last_s)] = value

    put(10, 20, 0.35)
    put(20, 40, 0.9)
    put(40, 44, 0.35)
    put(44, 50, 0.9)
    put(54, 70, 0.9)
    put(80, 83, 0.9)
    options = MusicRegionOptions(
        smooth_windows=1, enter=0.40, exit=0.30, bridge_gap_s=5.0
    )
    assert segment_music(times, scores, 100.0, options) == [(18.25, 71.0)]

    apart = MusicRegionOptions(
        smooth_windows=1,
        enter=0.40,
        exit=0.30,
        bridge_gap_s=2.0,
        min_music_s=20.0,
    )
    assert segment_music(times, scores, 100.0, apart) == [(18.25, 51.0)]

    touching = MusicRegionOptions(
        smooth_windows=1, enter=0.40, exit=0.30, bridge_gap_s=0.0
    )
    scores[:] = 0.05
    put(20, 30, 0.9)
    put(31, 45, 0.9)
    assert segment_music(times, scores, 100.0, touching) == [(18.25, 46.0)]

    assert segment_music(times, scores * 0.0, 100.0, options) == []
    assert segment_music([], [], 100.0, options) == []


def test_cuts_shorter_than_two_seconds_at_a_file_edge_snap_to_the_edge():
    times = np.arange(0, 60, 0.5) + 1.0
    scores = np.where((times > 2) & (times < 52), 0.9, 0.0)
    assert segment_music(times, scores, 54.0) == [(0.0, 54.0)]
    assert segment_music(times, scores, 60.0) == [(0.0, 53.0)]


def test_segment_music_runs_to_the_end_and_clips_to_the_audio():
    times = np.arange(0, 30, 0.5) + 1.0
    scores = np.where(times > 10, 0.9, 0.0)
    assert segment_music(times, scores, 31.0) == [(10.5 - 1.75, 31.0)]
    scores[:] = 0.9
    assert segment_music(times, scores, 31.0) == [(0.0, 31.0)]


def test_segment_music_median_filter_ignores_single_window_spikes():
    times = np.arange(0, 60, 0.5) + 1.0
    scores = np.zeros(len(times))
    scores[40] = 1.0
    scores[60:100] = 1.0
    regions = segment_music(times, scores, 61.0)
    assert len(regions) == 1
    assert regions[0][0] > 25.0


def test_regions_helpers():
    whole = MusicRegions(((0.2, 99.8),), 100.0, True, "")
    assert whole.is_identity()
    assert not whole.is_identity(tolerance=0.1)
    assert not MusicRegions(((0.0, 50.0),), 100.0, True, "").is_identity()
    assert not MusicRegions(
        ((0.0, 40.0), (60.0, 100.0)), 100.0, True, ""
    ).is_identity()
    cut = MusicRegions(((10.0, 20.0), (30.0, 40.0)), 50.0, True, "")
    assert cut.contains(10.0) and cut.contains(35.0) and cut.contains(40.0)
    assert not cut.contains(25.0) and not cut.contains(5.0)
    untouched = MusicRegions((), 50.0, False, "")
    assert untouched.contains(25.0)
    assert MusicRegions(((1.0, 2.0),), 3.0, True, "a") == MusicRegions(
        ((1.0, 2.0),), 3.0, True, "a", times=np.zeros(1)
    )


def test_options_are_validated():
    assert MusicRegionOptions().enter == 0.30
    bad = [
        {"min_duration_s": -1},
        {"enter": 1.5},
        {"exit": -0.1},
        {"enter": 0.3, "exit": 0.4},
        {"min_music_s": float("nan")},
        {"bridge_gap_s": -2},
        {"start_pad_s": 100},
        {"end_pad_s": -1},
        {"min_music_share": 2},
        {"smooth_windows": 0},
        {"smooth_windows": 2.5},
        {"fade_s": -0.1},
        {"manual_start_s": -3.0},
        {"manual_end_s": "ten"},
        {"manual_start_s": 20.0, "manual_end_s": 10.0},
    ]
    for kwargs in bad:
        with pytest.raises(ValueError, match="got"):
            MusicRegionOptions(**kwargs)
    with pytest.raises(ValueError, match="Fade length"):
        cut_outside_regions("a.wav", "b.wav", [], fade_s=-1)
    with pytest.raises(ValueError, match="Sample rate"):
        music_score_curve(np.zeros(SR), 0)


def drum_only(seconds, bpm=100.0):
    loop, _ = make_drum_loop(bpm=bpm, bars=int(seconds * bpm / 240) + 1)
    audio = resample_poly(loop, SR, DRUM_RATE)
    return audio[: int(seconds * SR)]


def kick_thumps(seconds, onsets):
    out = np.zeros(int(seconds * SR))
    count = int(0.25 * SR)
    t = np.arange(count) / SR
    thump = np.sin(2 * np.pi * (45 + 90 * np.exp(-t * 28)) * t)
    thump *= np.exp(-t * 14)
    for onset in onsets:
        start = int(onset * SR)
        if start + count >= len(out):
            break
        out[start : start + count] += thump
    return out


def vocals_over_thumps(seconds, onsets):
    return 1.5 * speech_like(seconds, 7) + 0.5 * kick_thumps(seconds, onsets)


def test_a_drum_only_stretch_between_music_is_not_cut_out():
    parts = [
        speech_like(INTRO_S, 1),
        music_body(20.0),
        drum_only(24.0),
        music_body(20.0, seed=4),
        speech_like(OUTRO_S, 3),
    ]
    found = find_music_regions(np.concatenate(parts).astype(np.float32), SR)
    drums_from = INTRO_S + 20.0
    drums_to = drums_from + 24.0
    assert found.applied
    assert len(found.regions) == 1
    start, end = found.regions[0]
    assert start < drums_from - 10.0
    assert end > drums_to + 20.0


def test_rhythm_lifts_the_score_of_a_beat_without_pitch():
    audio = np.concatenate([music_body(10.0), drum_only(24.0)])
    times, scores = music_score_curve(audio.astype(np.float32), SR)
    drums = times > 10.0 + 4.0
    assert drums.any()
    assert scores[drums].min() >= music_regions.RHYTHM_SCORE * 0.9


def test_vocals_over_a_kick_pattern_are_not_cut_out():
    seconds = 30.0
    rap = vocals_over_thumps(seconds, np.arange(0.0, seconds, 60.0 / 90.0))
    parts = [
        speech_like(INTRO_S, 1),
        music_body(20.0),
        rap,
        music_body(20.0, seed=4),
        speech_like(OUTRO_S, 3),
    ]
    found = find_music_regions(np.concatenate(parts).astype(np.float32), SR)
    rap_from = INTRO_S + 20.0
    rap_to = rap_from + seconds
    assert found.applied
    assert len(found.regions) == 1
    start, end = found.regions[0]
    assert start < rap_from - 10.0
    assert end > rap_to + 20.0


def test_vocals_over_irregular_thumps_are_still_cut():
    seconds = 30.0
    rng = np.random.default_rng(1)
    rap = vocals_over_thumps(seconds, np.cumsum(rng.uniform(0.3, 1.5, 100)))
    parts = [
        speech_like(INTRO_S, 1),
        music_body(20.0),
        rap,
        music_body(20.0, seed=4),
        speech_like(OUTRO_S, 3),
    ]
    found = find_music_regions(np.concatenate(parts).astype(np.float32), SR)
    rap_from = INTRO_S + 20.0
    rap_to = rap_from + seconds
    assert len(found.regions) == 2
    assert found.regions[0][1] < rap_from + 4.0
    assert found.regions[1][0] > rap_to - 4.0


def test_timeline_without_beat_alignment_closes_the_gaps():
    audio = np.zeros(20 * SR, dtype=np.float32)
    options = MusicRegionOptions(align_beats=False)
    timeline = build_timeline(audio, [(1.0, 3.0), (6.0, 9.0)], options)
    assert timeline == [(1.0, 3.0, 0.0), (6.0, 9.0, 2.0)]
    plain = MusicRegions(((1.0, 3.0), (6.0, 9.0)), 20.0, True, "")
    assert plain.cut_timeline() == ((1.0, 3.0, 0.0), (6.0, 9.0, 2.0))


def test_cut_removes_the_gap_and_keeps_the_music_bit_exact(tmp_path):
    rate = 48000
    rng = np.random.default_rng(0)
    data = (0.3 * rng.standard_normal((6 * rate + 77, 2))).astype(np.float32)
    src = tmp_path / "stereo.wav"
    sf.write(src, data, rate, subtype="FLOAT")
    timeline = [(0.0, 1.0, 0.0), (3.0, 4.5, 1.0)]
    dst = cut_outside_regions(str(src), str(tmp_path / "cut.wav"), timeline)
    info = sf.info(dst)
    assert (info.samplerate, info.channels, info.subtype) == (
        rate,
        2,
        "FLOAT",
    )
    assert info.frames == round(2.5 * rate)
    out, _ = sf.read(dst, dtype="float32")
    fade = round(0.05 * rate)
    assert np.array_equal(out[: rate - fade], data[: rate - fade])
    assert np.array_equal(
        out[rate + fade : 2 * rate],
        data[3 * rate + fade : 4 * rate],
    )
    ramp = np.abs(out[rate : rate + fade, 0])
    assert ramp[0] < 0.01 and ramp[-1] > 0.0


def test_cut_keeps_the_requested_silence_between_segments(tmp_path):
    rate = 22050
    data = np.full((5 * rate, 1), 0.5, dtype=np.float32)
    src = tmp_path / "mono.wav"
    sf.write(src, data, rate, subtype="FLOAT")
    timeline = [(0.0, 1.0, 0.0), (3.0, 4.0, 1.3)]
    dst = cut_outside_regions(
        str(src), str(tmp_path / "cut.wav"), timeline, 0.0
    )
    out, _ = sf.read(dst, dtype="float32")
    assert len(out) == round(2.3 * rate)
    assert not out[rate : round(1.3 * rate)].any()
    assert out[: rate - 1].min() == 0.5
    assert out[round(1.3 * rate) :].min() == 0.5


def test_cut_decoded_fallback_matches_the_streaming_path(tmp_path):
    rate = 22050
    rng = np.random.default_rng(1)
    data = (0.3 * rng.standard_normal((4 * rate, 2))).astype(np.float32)
    src = tmp_path / "stereo.wav"
    sf.write(src, data, rate, subtype="FLOAT")
    timeline = [(0.5, 1.5, 0.0), (2.5, 3.5, 1.2)]
    streamed = cut_outside_regions(str(src), str(tmp_path / "a.wav"), timeline)
    decoded = music_regions._cut_decoded(
        str(src), str(tmp_path / "b.wav"), timeline, 0.05
    )
    first, _ = sf.read(streamed, dtype="float32")
    second, _ = sf.read(decoded, dtype="float32")
    assert np.array_equal(first, second)


def _pulse_times(path):
    import librosa

    audio, rate = sf.read(path, dtype="float32")
    return librosa.onset.onset_detect(
        y=audio, sr=rate, units="time", backtrack=False
    )


def _beat_phase(times, period=0.5):
    angle = 2 * np.pi * (times % period) / period
    mean = np.angle(np.exp(1j * angle).mean())
    return mean / (2 * np.pi) * period


def _phase_jump(path, join):
    times = _pulse_times(path)
    before = _beat_phase(times[(times > 8.0) & (times < join - 2.0)])
    after = _beat_phase(times[(times > join + 3.0) & (times < join + 20.0)])
    return abs((after - before + 0.25) % 0.5 - 0.25)


def test_the_cut_keeps_the_music_in_step_with_the_beat(tmp_path):
    song = music_body(80.0)
    first, second = int(30.0 * SR), int(43.4 * SR)
    song[first:second] = speech_like((second - first) / SR + 0.1, 9)[
        : second - first
    ]
    src = tmp_path / "song.wav"
    sf.write(src, song, SR, subtype="FLOAT")
    regions = [(0.0, 30.0), (second / SR, 80.0)]

    aligned = build_timeline(song.astype(np.float32), regions)
    plain = build_timeline(
        song.astype(np.float32), regions, MusicRegionOptions(align_beats=False)
    )
    assert plain[1][2] == pytest.approx(30.0)
    delay = aligned[1][2] - plain[1][2]
    gap = second / SR - 30.0
    assert 0.0 <= delay < 1.5
    assert abs((delay - gap + 0.25) % 0.5 - 0.25) < 0.03

    joined = cut_outside_regions(str(src), str(tmp_path / "a.wav"), aligned)
    stuck = cut_outside_regions(str(src), str(tmp_path / "b.wav"), plain)
    assert _phase_jump(joined, 30.0) < 0.03
    assert _phase_jump(stuck, 30.0) > 0.08


def test_unclear_rhythm_falls_back_to_a_plain_join():
    rng = np.random.default_rng(3)
    noise = (0.2 * rng.standard_normal(40 * SR)).astype(np.float32)
    timeline = build_timeline(noise, [(0.0, 15.0), (25.0, 40.0)])
    assert timeline == [(0.0, 15.0, 0.0), (25.0, 40.0, 15.0)]
