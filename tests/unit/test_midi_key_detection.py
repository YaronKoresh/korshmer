import numpy as np
import pretty_midi
import pytest

from korshmer.analysis import (
    DETECTABLE_SCALES,
    detect_midi_key,
    export_scale_stats,
    rank_scales,
)
from korshmer.constants import NOTE_NAMES, SCALES


def _steps(count):
    tune = [0, *range(1, count), 0, 2 % count, 4 % count, 0]
    tune += [0, *reversed(range(1, count)), 0, 4 % count, 0]
    return tune


def _midi(root, scale, tmp_path):
    intervals = SCALES[scale]
    inst = pretty_midi.Instrument(0)
    clock = 0.0
    for step in _steps(len(intervals)):
        length = 1.0 if step == 0 else 0.4
        pitch = 60 + root + intervals[step]
        inst.notes.append(pretty_midi.Note(90, pitch, clock, clock + length))
        clock += length if step == 0 else 0.5
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(inst)
    path = tmp_path / f"{scale}_{root}.mid"
    pm.write(str(path))
    return str(path)


def _chroma(rng, root, scale):
    chroma = np.zeros(12)
    for interval in SCALES[scale]:
        weight = 2.5
        if interval == 0:
            weight = 6.0
        elif interval == 7:
            weight = 4.0
        elif interval in (3, 4):
            weight = 3.2
        chroma[(root + interval) % 12] = weight * rng.uniform(0.8, 1.2)
    return chroma + rng.uniform(0, 0.15, 12)


@pytest.mark.parametrize(
    ("root", "scale"),
    [
        (2, "Lydian"),
        (9, "Dorian"),
        (4, "Phrygian"),
        (7, "Mixolydian"),
        (0, "Arabian (Maqam Hijaz)"),
        (9, "Harmonic Minor"),
        (0, "Pentatonic Minor"),
        (0, "Blues"),
        (5, "Major"),
        (9, "Minor"),
    ],
)
def test_melodies_report_their_own_scale(root, scale, tmp_path):
    assert detect_midi_key(_midi(root, scale, tmp_path)) == (
        NOTE_NAMES[root],
        scale,
    )


def test_every_detectable_scale_is_a_known_scale_name():
    assert len(set(DETECTABLE_SCALES)) == len(DETECTABLE_SCALES)
    assert set(DETECTABLE_SCALES) <= set(SCALES)
    assert {"Lydian", "Dorian", "Arabian (Maqam Hijaz)"} <= set(
        DETECTABLE_SCALES
    )


@pytest.mark.parametrize("scale", DETECTABLE_SCALES)
def test_scale_is_found_in_nine_of_ten_noisy_profiles(scale):
    rng = np.random.default_rng(11)
    good = 0
    for _ in range(100):
        root = int(rng.integers(12))
        top = rank_scales(_chroma(rng, root, scale), 1)[0]
        good += top[0] == root and top[1] == scale
    assert good >= 90


def test_ranking_is_sorted_and_limited():
    ranked = rank_scales(_chroma(np.random.default_rng(1), 2, "Lydian"), 4)
    assert len(ranked) == 4
    scores = [score for _root, _name, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0][:2] == (2, "Lydian")


def test_silence_falls_back_to_c_major():
    assert rank_scales(np.zeros(12), 3) == [(0, "Major", 0.0)]


def test_scale_stats_report_names_the_real_scale(tmp_path):
    pm = pretty_midi.PrettyMIDI(_midi(2, "Lydian", tmp_path))
    out = export_scale_stats(pm, str(tmp_path / "stats.txt"))
    text = open(out, encoding="utf-8").read()
    assert "Top Estimation: D Lydian" in text
    assert "Also possible:" in text


@pytest.mark.parametrize(
    ("root", "scale"),
    [(2, "Lydian"), (0, "Arabian (Maqam Hijaz)"), (9, "Dorian")],
)
def test_clip_key_line_names_the_real_scale(root, scale, tmp_path):
    from korshmer.video.music_clip import _midi_key

    assert _midi_key(_midi(root, scale, tmp_path)) == (
        f"{NOTE_NAMES[root]} {scale}"
    )


def _tone_audio(root, scale, tmp_path, sr=22050):
    import soundfile as sf

    intervals = SCALES[scale]
    pieces = []
    for step in _steps(len(intervals)) * 4:
        freq = 440 * 2 ** ((60 + root + intervals[step] - 69) / 12)
        t = np.arange(int(sr * (1.2 if step == 0 else 0.45))) / sr
        wave = sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3))
        pieces.append(wave * np.exp(-2 * t))
    audio = np.concatenate(pieces).astype(np.float32)
    path = tmp_path / f"audio_{scale}_{root}.wav"
    sf.write(str(path), audio / np.abs(audio).max() * 0.5, sr)
    return str(path)


@pytest.mark.parametrize(
    ("root", "scale"),
    [
        (2, "Lydian"),
        (9, "Dorian"),
        (4, "Phrygian"),
        (0, "Arabian (Maqam Hijaz)"),
        (9, "Harmonic Minor"),
        (5, "Major"),
        (9, "Minor"),
    ],
)
def test_audio_key_detection_reports_the_scale_too(root, scale, tmp_path):
    from korshmer.analysis import detect_key_and_mode

    assert detect_key_and_mode(_tone_audio(root, scale, tmp_path)) == (
        root,
        scale,
    )


def test_audio_key_detection_still_falls_back_for_unreadable_audio(tmp_path):
    from korshmer.analysis import detect_key_and_mode

    bad = tmp_path / "missing.wav"
    assert detect_key_and_mode(str(bad)) == (0, "Major")


TUNES = {
    "twinkle": (
        [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0, 7, 7, 5, 5, 4, 4, 2]
        + [7, 7, 5, 5, 4, 4, 2, 0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0],
        (0, "Major"),
    ),
    "ode": (
        [4, 4, 5, 7, 7, 5, 4, 2, 0, 0, 2, 4, 4, 2, 2, 4, 4, 5, 7, 7, 5, 4]
        + [2, 0, 0, 2, 4, 2, 0, 0, 2, 2, 4, 0, 2, 4, 5, 4, 0, 2, 4, 5, 4],
        (0, "Major"),
    ),
    "scarborough": (
        [2, 2, 9, 9, 9, 0, 11, 9, 7, 5, 2, 2, 9, 9, 0, 0, 2, 0, 11, 9, 7]
        + [5, 2, 4, 5, 5, 7, 9, 7, 5, 4, 2, 2],
        (2, "Dorian"),
    ),
    "hijaz": (
        [2, 3, 6, 7, 9, 10, 9, 7, 6, 3, 2, 2, 3, 6, 7, 9, 10, 2],
        (2, "Arabian (Maqam Hijaz)"),
    ),
    "natural minor": (
        [9, 0, 2, 4, 2, 0, 11, 9, 7, 9, 11, 0, 9, 9, 7, 5, 4, 5, 7, 9, 9]
        + [0, 2, 4, 5, 4, 2, 0, 11, 9, 7, 5, 4, 9],
        (9, "Minor"),
    ),
}


@pytest.mark.parametrize("name", sorted(TUNES))
def test_well_known_tunes_keep_their_usual_scale(name):
    notes, (root, scale) = TUNES[name]
    counts = np.bincount(np.array(notes) % 12, minlength=12).astype(float)
    assert rank_scales(counts, 1)[0][:2] == (root, scale)
