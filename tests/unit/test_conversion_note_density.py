import os

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer.midi.transcription_cleanup import (
    level_flux,
    merge_note_variants,
    merge_sustained_splits,
    merge_unattacked_splits,
    pitch_band_level,
)

SAMPLE_RATE = 22050
TEMPO = 120.0


def render(notes, seconds):
    audio = np.zeros(int(seconds * SAMPLE_RATE))
    for pitch, start, end, velocity in notes:
        t = np.arange(int((end - start) * SAMPLE_RATE)) / SAMPLE_RATE
        hz = 440.0 * 2 ** ((pitch - 69) / 12)
        tone = sum(np.sin(2 * np.pi * hz * k * t) / k for k in (1, 2, 3))
        decay = 0.5 * np.exp(-30 * t) + 0.5 * np.exp(-3 * t)
        tone = tone * decay * (velocity / 127)
        tone[:64] *= np.linspace(0, 1, 64)
        first = int(start * SAMPLE_RATE)
        audio[first : first + len(tone)] += tone[: len(audio) - first]
    return 0.8 * audio / np.abs(audio).max()


def as_midi(notes):
    midi = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(v, p, s, e) for p, s, e, v in notes]
    midi.instruments.append(inst)
    return midi


@pytest.fixture
def convert(monkeypatch, tmp_path):
    from korshmer import track_processor as tp

    def run(notes, second_variant=None, stem="piano", seconds=6.0):
        audio = render(notes, seconds)
        source = str(tmp_path / "song.wav")
        sf.write(source, audio, SAMPLE_RATE)

        def separate(path, job_dir):
            folder = os.path.join(job_dir, "htdemucs_6s", "song")
            os.makedirs(folder)
            sf.write(os.path.join(folder, f"{stem}.wav"), audio, SAMPLE_RATE)

        def predict(path, midi_tempo=TEMPO, **settings):
            use_second = second_variant is not None and path.endswith("_1.wav")
            return None, as_midi(second_variant if use_second else notes), None

        replacements = {
            "DATA_DIR": str(tmp_path / "data"),
            "trim_audio_silence": lambda path, job_dir: path,
            "analyse_tempo": lambda path: TEMPO,
            "normalize_detected_tempo": lambda tempo: tempo,
            "detect_key_and_mode": lambda path: (0, "Major"),
            "_apply_demucs_patches": lambda: None,
            "_apply_torchaudio_patches": lambda: None,
            "run_demucs": separate,
            "_get_basic_pitch_predict": lambda: predict,
        }
        for name, value in replacements.items():
            monkeypatch.setattr(tp, name, value)
        files = tp.process_track(
            source,
            "Original",
            "Auto",
            "Auto (Detect)",
            "1/16",
            selected_stems=[stem],
        )
        return [
            note
            for path in files
            if path.endswith(".mid")
            for inst in pretty_midi.PrettyMIDI(path).instruments
            for note in inst.notes
        ]

    return run


REPEATS = [(60, 0.5 + i * 0.125, 0.61 + i * 0.125, 90) for i in range(16)]
HELD = [(60, 0.5, 2.5, 90)]
CHORDS = [
    (root + step, 3.0 + i * 0.5, 3.45 + i * 0.5, 85)
    for i, root in enumerate((40, 41, 43, 45))
    for step in (0, 7, 12, 16, 19, 22)
]


def count(midi):
    return sum(len(inst.notes) for inst in midi.instruments)


def test_restruck_notes_are_confirmed_by_a_sustained_variant():
    merged = merge_note_variants([as_midi(REPEATS), as_midi(HELD)])
    assert count(merged) == len(REPEATS)


def test_unconfirmed_note_of_typical_loudness_is_kept():
    first = [(60, 0.0, 0.2, 80), (64, 0.5, 0.7, 80), (67, 1.0, 1.2, 80)]
    merged = merge_note_variants([as_midi(first), as_midi(first[:2])])
    assert count(merged) == 3


def test_unconfirmed_short_quiet_note_is_dropped():
    first = [(60, 0.0, 0.2, 80), (64, 0.5, 0.7, 80), (90, 1.0, 1.1, 20)]
    merged = merge_note_variants([as_midi(first), as_midi(first[:2])])
    assert [n.pitch for n in merged.instruments[0].notes] == [60, 64]


def test_strict_voting_drops_every_unconfirmed_note():
    first = [(60, 0.0, 0.2, 80), (64, 0.5, 0.7, 80), (67, 1.0, 1.2, 80)]
    merged = merge_note_variants(
        [as_midi(first), as_midi(first[:2])], keep_seconds=None
    )
    assert count(merged) == 2


def test_fast_repeated_notes_survive_conversion(convert):
    notes = convert(REPEATS + CHORDS, second_variant=HELD + CHORDS)
    repeats = [n for n in notes if n.start < 2.75]
    assert len(repeats) > len(REPEATS) // 2


def test_full_chords_survive_conversion(convert):
    notes = convert(CHORDS)
    assert len(notes) == len(CHORDS)


def test_split_without_a_new_attack_is_still_joined():
    held = [(57, 0.2, 1.4, 90)]
    plucks = [(60 + i, 1.6 + i * 0.3, 1.85 + i * 0.3, 90) for i in range(8)]
    level, rate = pitch_band_level(render(held + plucks, 4.5), SAMPLE_RATE)
    inst = pretty_midi.Instrument(0)
    inst.notes = [
        pretty_midi.Note(90, 57, 0.2, 0.8),
        pretty_midi.Note(90, 57, 0.8, 1.4),
        *as_midi(plucks).instruments[0].notes,
    ]
    joins = merge_sustained_splits(inst, level, rate, flux=level_flux(level))
    assert joins == 1
    assert len(inst.notes) == 1 + len(plucks)


def test_level_check_does_not_join_notes_with_a_measured_attack():
    level, rate = pitch_band_level(render(REPEATS + CHORDS, 6.0), SAMPLE_RATE)
    flux = level_flux(level)
    inst = as_midi(REPEATS + CHORDS).instruments[0]
    merge_unattacked_splits(inst, flux, rate)
    struck = len(inst.notes)
    assert merge_sustained_splits(inst, level, rate, flux=flux) == 0
    assert len(inst.notes) == struck > len(CHORDS) + len(REPEATS) // 2
