import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing import vocal_melody
from korshmer.audio_processing.vocal_melody import (
    VocalMelodyOptions,
    singing_share,
    transcribe_vocal_melody,
)

HOP_S = 256 / 22050
FAST_LEAPS = [60, 67, 59, 66, 62, 69, 57, 64]


def _sing(notes, sr=22050, vibrato_cents=30.0, glide=0.04, lead=0.2):
    total = lead + sum(d + g for _, d, g, _ in notes) + 0.2
    t = np.arange(int(total * sr)) / sr
    pitch = np.zeros(len(t))
    envelope = np.zeros(len(t))
    onsets = []
    position = lead
    previous = notes[0][0]
    for index, (midi, duration, gap, gain) in enumerate(notes):
        first = int(position * sr)
        last = int((position + duration) * sr)
        pitch[first:last] = midi
        if index:
            ramp = int(glide * sr)
            pitch[first : first + ramp] = np.linspace(previous, midi, ramp)
        shape = np.ones(last - first)
        fade = int(0.02 * sr)
        shape[:fade] = np.linspace(0, 1, fade)
        shape[-fade:] = np.linspace(1, 0, fade)
        envelope[first:last] = shape * gain
        onsets.append(position)
        previous = midi
        position += duration + gap
    held = np.maximum.accumulate(np.where(pitch > 0, np.arange(len(t)), 0))
    pitch = pitch[held]
    cents = vibrato_cents / 100 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(440 * 2 ** ((pitch + cents - 69) / 12)) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 7))
    return 0.5 * voice / np.abs(voice).max() * envelope, onsets


def _melody(pitches, duration=0.4):
    return [
        (m, duration + 0.05 * (i % 3), 0.03 + 0.01 * (i % 5), 1.0)
        for i, m in enumerate(pitches)
    ]


def _write(tmp_path, audio, sr=22050, subtype="PCM_16", channels=1):
    path = tmp_path / "voice.wav"
    data = np.stack([audio] * channels, axis=1) if channels > 1 else audio
    sf.write(path, data, sr, subtype=subtype)
    return str(path)


@pytest.mark.parametrize(
    "sr,subtype,channels",
    [(22050, "PCM_16", 1), (44100, "FLOAT", 2), (44100, "PCM_24", 2)],
)
def test_sung_melody_becomes_one_note_per_pitch(
    tmp_path, sr, subtype, channels
):
    pitches = [60, 62, 64, 65, 67, 65, 64, 62, 60, 64]
    audio, onsets = _sing(_melody(pitches), sr=sr)
    notes = transcribe_vocal_melody(
        _write(tmp_path, audio, sr, subtype, channels)
    )
    assert [n.pitch for n in notes] == pitches
    assert all(
        abs(n.start - o) < 0.06 for n, o in zip(notes, onsets, strict=False)
    )


def test_repeated_pitch_is_split_at_each_gap(tmp_path):
    audio, onsets = _sing([(62, 0.3, 0.04, 1.0)] * 5)
    notes = transcribe_vocal_melody(_write(tmp_path, audio))
    assert [n.pitch for n in notes] == [62] * 5
    assert all(
        abs(n.start - o) < 0.06 for n, o in zip(notes, onsets, strict=False)
    )


def test_portamento_between_distant_notes_adds_no_note(tmp_path):
    pitches = [60, 67, 60, 72, 55]
    notes_in = [(m, 0.45, 0.0, 1.0) for m in pitches]
    audio, _ = _sing(notes_in, glide=0.08)
    notes = transcribe_vocal_melody(_write(tmp_path, audio))
    assert [n.pitch for n in notes] == pitches


def test_long_input_is_processed_in_chunks(tmp_path, monkeypatch):
    pitches = [60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 66, 68]
    audio, _ = _sing(_melody(pitches))
    path = _write(tmp_path, audio)
    whole = transcribe_vocal_melody(path)
    monkeypatch.setattr(vocal_melody, "CHUNK_SECONDS", 1.5)
    chunked = transcribe_vocal_melody(path)
    assert [n.pitch for n in chunked] == pitches
    assert chunked == whole


def test_vibrato_does_not_split_a_note(tmp_path):
    audio, _ = _sing([(60.35, 2.0, 0.0, 1.0)], vibrato_cents=40.0)
    notes = transcribe_vocal_melody(_write(tmp_path, audio))
    assert len(notes) == 1
    assert notes[0].pitch == 60
    assert notes[0].duration > 1.8


def test_octave_is_not_confused_for_low_male_range(tmp_path):
    low = [40, 43, 45, 47, 45, 43, 40]
    audio, _ = _sing(_melody(low), sr=44100)
    notes = transcribe_vocal_melody(_write(tmp_path, audio, 44100))
    assert [n.pitch for n in notes] == low
    high = [69, 72, 74, 76, 74, 72, 69]
    audio, _ = _sing(_melody(high))
    notes = transcribe_vocal_melody(_write(tmp_path, audio))
    assert [n.pitch for n in notes] == high


def test_contour_without_plateaus_is_dropped(tmp_path):
    sr = 22050
    t = np.arange(6 * sr) / sr
    pitch = 55 + 6 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(440 * 2 ** ((pitch - 69) / 12)) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 7))
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)
    path = _write(tmp_path, 0.3 * voice / np.abs(voice).max() * syllables)
    assert transcribe_vocal_melody(path, VocalMelodyOptions(speech_gate=False))
    assert transcribe_vocal_melody(path) == []


@pytest.mark.parametrize("duration,count", [(0.2, 24), (0.15, 32)])
def test_fast_leaping_phrase_survives_the_speech_gate(
    tmp_path, duration, count
):
    pitches = (FAST_LEAPS * 4)[:count]
    audio, onsets = _sing([(m, duration, 0.0, 1.0) for m in pitches])
    notes = transcribe_vocal_melody(_write(tmp_path, audio))
    assert [n.pitch for n in notes] == pitches
    assert all(
        abs(n.start - o) < 0.06 for n, o in zip(notes, onsets, strict=False)
    )


def test_noise_and_silence_give_no_notes(tmp_path):
    rng = np.random.default_rng(3)
    noise = _write(tmp_path, 0.1 * rng.standard_normal(3 * 22050))
    assert transcribe_vocal_melody(noise) == []
    assert transcribe_vocal_melody(_write(tmp_path, np.zeros(3 * 22050))) == []


def test_quiet_bleed_below_the_phrase_is_dropped(tmp_path):
    phrase = [(m, 0.45, 0.04, 1.0) for m in (60, 64, 67, 64)]
    bleed = [(m, 0.45, 0.04, 0.01) for m in (62, 65, 69)]
    path = _write(tmp_path, _sing(phrase + bleed)[0])
    notes = transcribe_vocal_melody(path)
    assert [n.pitch for n in notes] == [60, 64, 67, 64]
    kept = transcribe_vocal_melody(
        path, VocalMelodyOptions(max_level_drop_db=60.0)
    )
    assert len(kept) == 7


def test_notes_are_sorted_non_overlapping_and_in_range(tmp_path):
    rng = np.random.default_rng(5)
    pitches = list(rng.integers(40, 80, size=14))
    notes_in = [
        (int(m), 0.2 + 0.3 * rng.random(), 0.02 + 0.1 * rng.random(), 1.0)
        for m in pitches
    ]
    audio, _ = _sing(notes_in)
    audio = audio + 0.002 * rng.standard_normal(len(audio))
    notes = transcribe_vocal_melody(_write(tmp_path, audio))
    assert notes
    for note in notes:
        assert isinstance(note.pitch, int)
        assert 24 <= note.pitch <= 100
        assert 0 <= note.velocity <= 127
        assert note.start < note.end
    for first, second in zip(notes, notes[1:], strict=False):
        assert first.end <= second.start


def test_options_are_validated():
    assert VocalMelodyOptions().speech_gate is True
    for bad in (
        {"hop_length": 0},
        {"hop_length": 256.5},
        {"sample_rate": True},
        {"fmin": 500.0, "fmax": 100.0},
        {"fmin": float("nan")},
        {"switch_cost": 0.0},
        {"smooth_frames": 0},
        {"gate_min_share": 1.5},
        {"gate_step_s": 0.0},
        {"gate_step_tolerance": 0.0},
        {"min_note_s": -0.1},
        {"max_level_drop_db": float("inf")},
    ):
        with pytest.raises(ValueError, match="got"):
            VocalMelodyOptions(**bad)


def test_short_file_returns_empty(tmp_path):
    assert transcribe_vocal_melody(_write(tmp_path, np.zeros(0))) == []
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(1000) / 22050)
    assert transcribe_vocal_melody(_write(tmp_path, tone)) == []


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        transcribe_vocal_melody(str(tmp_path / "missing.wav"))


def test_velocity_follows_level(tmp_path):
    notes_in = [(60, 0.5, 0.05, g) for g in (1.0, 0.4, 0.15, 0.06)]
    notes = transcribe_vocal_melody(_write(tmp_path, _sing(notes_in)[0]))
    velocities = [n.velocity for n in notes]
    assert len(velocities) == 4
    assert velocities == sorted(velocities, reverse=True)
    assert velocities[0] == 110
    assert min(velocities) >= 45
    assert len(set(velocities)) == 4


def test_progress_reports_fractions_and_text(tmp_path):
    audio, _ = _sing(_melody([60, 64, 67]))
    calls = []
    transcribe_vocal_melody(
        _write(tmp_path, audio),
        progress=lambda f, text: calls.append((f, text)),
    )
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert all(isinstance(text, str) and text for _, text in calls)


def test_stop_between_chunks_raises_after_progress(tmp_path, monkeypatch):
    audio, _ = _sing(_melody([60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 66, 68]))
    path = _write(tmp_path, audio)
    monkeypatch.setattr(vocal_melody, "CHUNK_SECONDS", 1.5)
    fractions = []
    with pytest.raises(InterruptedError, match="Stopped by user"):
        transcribe_vocal_melody(
            path,
            progress=lambda f, text: fractions.append(f),
            should_stop=lambda: len(fractions) >= 2,
        )
    assert len(fractions) == 2
    assert fractions[0] == 0.0 < fractions[1] < 0.9


def test_should_stop_is_polled_once_per_chunk(tmp_path, monkeypatch):
    audio, _ = _sing(_melody([60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 66, 68]))
    path = _write(tmp_path, audio)
    monkeypatch.setattr(vocal_melody, "CHUNK_SECONDS", 1.5)
    fractions = []
    expected = transcribe_vocal_melody(
        path, progress=lambda f, text: fractions.append(f)
    )
    polls = []
    notes = transcribe_vocal_melody(
        path, should_stop=lambda: polls.append(True) or False
    )
    assert notes == expected
    assert len(polls) == len([f for f in fractions if 0.0 < f < 1.0]) > 1


def test_singing_share_tells_steady_notes_from_glides():
    t = np.arange(int(8 / HOP_S)) * HOP_S
    steps = 60 + 2 * (np.floor(t / 0.5) % 3)
    glide = 55 + 6 * np.sin(2 * np.pi * 1.5 * t)
    options = VocalMelodyOptions()
    assert np.nanmin(singing_share(steps, HOP_S, options)) > 0.9
    assert np.nanmax(singing_share(glide, HOP_S, options)) < (
        options.gate_min_share
    )


def test_singing_share_counts_fast_leaps_as_steady():
    t = np.arange(int(8 / HOP_S)) * HOP_S
    leaps = 60 + 7 * (np.floor(t / 0.15) % 2)
    assert np.nanmin(singing_share(leaps, HOP_S)) > 0.9


def test_singing_share_is_undefined_without_enough_voiced_time():
    contour = np.full(int(4 / HOP_S), np.nan)
    contour[100:130] = 60.0
    assert np.isnan(singing_share(contour, HOP_S)).all()


def test_a_fast_tonal_sweep_is_not_taken_for_a_melody(tmp_path):
    sr = 22050
    t = np.arange(6 * sr) / sr
    pitch = 36 * ((t * 11 / 36) % 1.0)
    phase = 2 * np.pi * np.cumsum(110 * 2 ** (pitch / 12)) / sr
    path = _write(tmp_path, 0.3 * np.sin(phase))
    assert transcribe_vocal_melody(path, VocalMelodyOptions(speech_gate=False))
    assert transcribe_vocal_melody(path) == []


def test_high_notes_are_not_dropped_an_octave(tmp_path):
    audio, _ = _sing([(89, 0.7, 0.0, 1.0)])
    path = _write(tmp_path, audio)
    notes = transcribe_vocal_melody(path, VocalMelodyOptions(speech_gate=False))
    assert notes
    top = max(notes, key=lambda n: n.end - n.start)
    assert top.pitch >= 85
