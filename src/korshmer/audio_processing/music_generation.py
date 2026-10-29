import random

import numpy as np
import pretty_midi

from korshmer.midi_processing.note_validation import (
    validate_chord_progression,
)
from korshmer.utils.tempo import MAX_BPM, MIN_BPM

from .validation import validate_effect_integer, validate_effect_number

MAX_GENERATED_BARS = 512

RHYTHM_PATTERNS = {
    "steady": [1.0],
    "walking": [0.25, 0.25, 0.25, 0.25],
    "syncopated": [0.5, 0.25, 0.25],
    "pumping": [0.5, 0.5],
}


def _validate_tempo_and_bars(tempo, duration_bars):
    validate_effect_number(tempo, "Tempo", MIN_BPM, MAX_BPM)
    validate_effect_integer(
        duration_bars, "Duration bars", 1, MAX_GENERATED_BARS
    )


def _validate_choice(value, name, choices):
    if value not in choices:
        raise ValueError(
            f"Unknown {name}: {value!r}; use one of {', '.join(sorted(choices))}"
        )


class MusicGenerator:
    def __init__(self):
        self.scale_patterns = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "phrygian": [0, 1, 3, 5, 7, 8, 10],
            "lydian": [0, 2, 4, 6, 7, 9, 11],
            "mixolydian": [0, 2, 4, 5, 7, 9, 10],
            "locrian": [0, 1, 3, 5, 6, 8, 10],
            "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
            "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
        }

        self.chord_progressions = {
            "pop": [[0, 4, 7], [5, 9, 0], [7, 11, 2], [0, 4, 7]],
            "jazz": [[0, 4, 7, 11], [2, 5, 9, 0], [5, 9, 0, 4], [0, 4, 7, 11]],
            "blues": [[0, 4, 7], [5, 9, 0], [0, 4, 7], [0, 4, 7]],
            "rock": [[0, 4, 7], [7, 11, 2], [5, 9, 0], [0, 4, 7]],
        }

    def generate_melody_from_chords(
        self,
        chord_progression: list[list[int]],
        key: int = 60,
        scale: str = "major",
        tempo: float = 120.0,
        duration_bars: int = 4,
    ) -> pretty_midi.PrettyMIDI:
        chord_progression = validate_chord_progression(chord_progression)
        _validate_tempo_and_bars(tempo, duration_bars)
        validate_effect_integer(key, "Key", 0, 127)
        _validate_choice(scale, "scale", self.scale_patterns)
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        melody_instrument = pretty_midi.Instrument(program=0)

        scale_notes = [
            key + interval for interval in self.scale_patterns[scale]
        ]

        bar_duration = 60.0 / tempo * 4
        chord_duration = bar_duration * duration_bars / len(chord_progression)

        current_time = 0.0
        note_durations = [0.25, 0.5, 0.75, 1.0]

        for chord in chord_progression:
            chord_notes = [key + interval for interval in chord]
            chord_end = current_time + chord_duration

            while current_time < chord_end:
                duration = random.choice(note_durations) * (bar_duration / 4)

                if current_time + duration > chord_end:
                    duration = chord_end - current_time

                available_notes = [
                    n
                    for n in scale_notes
                    if any(abs((n % 12) - (c % 12)) <= 1 for c in chord_notes)
                ]
                if not available_notes:
                    available_notes = scale_notes

                pitch = random.choice(available_notes)

                octave_adjust = random.choice([0, 12, -12])
                pitch = np.clip(pitch + octave_adjust, 48, 84)

                velocity = random.randint(70, 100)

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=int(pitch),
                    start=current_time,
                    end=current_time + duration,
                )
                melody_instrument.notes.append(note)

                current_time += duration

        midi.instruments.append(melody_instrument)
        return midi

    def generate_harmony(
        self,
        melody_midi: pretty_midi.PrettyMIDI,
        interval: int = 3,
        key: int = 60,
        scale: str = "major",
    ) -> pretty_midi.PrettyMIDI:
        tempo_times, tempos = melody_midi.get_tempo_changes()
        if len(tempos) > 0:
            initial_tempo = float(tempos[0])
        else:
            initial_tempo = 120.0
        harmony_midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

        scale_notes = self.scale_patterns[scale]

        for instrument in melody_midi.instruments:
            harmony_instrument = pretty_midi.Instrument(
                program=instrument.program
            )

            for note in instrument.notes:
                pitch_class = note.pitch % 12
                relative_pitch = (pitch_class - key) % 12

                scale_degree = None
                for i, interval_val in enumerate(scale_notes):
                    if interval_val == relative_pitch:
                        scale_degree = i
                        break

                if scale_degree is not None:
                    harmony_degree = (scale_degree + interval) % len(
                        scale_notes
                    )
                    harmony_interval = scale_notes[harmony_degree]
                    harmony_pitch = (
                        (note.pitch // 12) * 12 + (key % 12) + harmony_interval
                    )

                    harmony_note = pretty_midi.Note(
                        velocity=max(1, note.velocity - 10),
                        pitch=int(np.clip(harmony_pitch, 0, 127)),
                        start=note.start,
                        end=note.end,
                    )
                    harmony_instrument.notes.append(harmony_note)

            harmony_midi.instruments.append(harmony_instrument)

        return harmony_midi

    def generate_bass_line(
        self,
        chord_progression: list[list[int]],
        key: int = 48,
        tempo: float = 120.0,
        duration_bars: int = 4,
        rhythm_pattern: str = "steady",
    ) -> pretty_midi.PrettyMIDI:
        chord_progression = validate_chord_progression(chord_progression)
        _validate_tempo_and_bars(tempo, duration_bars)
        validate_effect_integer(key, "Key", 0, 127)
        _validate_choice(rhythm_pattern, "rhythm pattern", RHYTHM_PATTERNS)
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        bass_instrument = pretty_midi.Instrument(program=33)

        bar_duration = 60.0 / tempo * 4
        chord_duration = bar_duration * duration_bars / len(chord_progression)

        pattern = RHYTHM_PATTERNS[rhythm_pattern]

        current_time = 0.0

        for chord in chord_progression:
            root_note = key + chord[0]
            fifth_note = key + chord[min(2, len(chord) - 1)]

            chord_end = current_time + chord_duration
            pattern_idx = 0

            while current_time < chord_end:
                duration_ratio = pattern[pattern_idx % len(pattern)]
                duration = duration_ratio * bar_duration

                if current_time + duration > chord_end:
                    duration = chord_end - current_time

                pitch = root_note if pattern_idx % 2 == 0 else fifth_note

                note = pretty_midi.Note(
                    velocity=90,
                    pitch=int(np.clip(pitch, 24, 60)),
                    start=current_time,
                    end=current_time + duration * 0.9,
                )
                bass_instrument.notes.append(note)

                current_time += duration
                pattern_idx += 1

        midi.instruments.append(bass_instrument)
        return midi

    def generate_drum_pattern(
        self,
        genre: str = "rock",
        tempo: float = 120.0,
        duration_bars: int = 4,
        intensity: float = 0.7,
    ) -> pretty_midi.PrettyMIDI:
        _validate_tempo_and_bars(tempo, duration_bars)
        validate_effect_number(intensity, "Intensity", 0.0, 1.0)
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)

        drum_map = {
            "kick": 36,
            "snare": 38,
            "hihat_closed": 42,
            "hihat_open": 46,
            "crash": 49,
            "ride": 51,
        }

        genre_patterns = {
            "rock": {
                "kick": [1, 0, 0, 0, 1, 0, 0, 0],
                "snare": [0, 0, 1, 0, 0, 0, 1, 0],
                "hihat_closed": [1, 1, 1, 1, 1, 1, 1, 1],
            },
            "electronic": {
                "kick": [1, 0, 0, 0, 1, 0, 0, 0],
                "snare": [0, 0, 0, 0, 1, 0, 0, 0],
                "hihat_closed": [1, 0, 1, 0, 1, 0, 1, 0],
            },
            "jazz": {
                "kick": [1, 0, 0, 1, 0, 0, 1, 0],
                "snare": [0, 1, 0, 1, 0, 1, 0, 1],
                "ride": [1, 1, 1, 1, 1, 1, 1, 1],
            },
        }

        _validate_choice(genre, "genre", genre_patterns)
        pattern = genre_patterns[genre]

        bar_duration = 60.0 / tempo * 4
        step_duration = bar_duration / 8

        for bar in range(duration_bars):
            bar_start = bar * bar_duration

            for drum_name, drum_pattern in pattern.items():
                if drum_name in drum_map:
                    drum_pitch = drum_map[drum_name]

                    for step, hit in enumerate(drum_pattern):
                        if hit == 1:
                            velocity = int(70 + intensity * 50)

                            if random.random() < 0.1:
                                velocity = int(velocity * 0.7)

                            note = pretty_midi.Note(
                                velocity=velocity,
                                pitch=drum_pitch,
                                start=bar_start + step * step_duration,
                                end=bar_start
                                + step * step_duration
                                + step_duration * 0.5,
                            )
                            drum_instrument.notes.append(note)

        midi.instruments.append(drum_instrument)
        return midi

    def create_variation(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        variation_type: str = "rhythmic",
        amount: float = 0.5,
    ) -> pretty_midi.PrettyMIDI:
        _tempo_times, tempos = midi_data.get_tempo_changes()
        initial_tempo = float(tempos[0]) if len(tempos) > 0 else 120.0
        varied_midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

        for instrument in midi_data.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program, is_drum=instrument.is_drum
            )

            for note in instrument.notes:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )

                if variation_type == "rhythmic":
                    time_shift = (random.random() - 0.5) * 0.1 * amount
                    new_note.start += time_shift
                    new_note.end += time_shift

                elif variation_type == "melodic":
                    if random.random() < amount and not instrument.is_drum:
                        pitch_shift = random.choice([-2, -1, 1, 2])
                        new_note.pitch = np.clip(
                            note.pitch + pitch_shift, 0, 127
                        )

                elif variation_type == "dynamic":
                    velocity_change = int((random.random() - 0.5) * 40 * amount)
                    new_note.velocity = np.clip(
                        note.velocity + velocity_change, 1, 127
                    )

                new_instrument.notes.append(new_note)

            varied_midi.instruments.append(new_instrument)

        return varied_midi

    def expand_motif(
        self,
        motif_midi: pretty_midi.PrettyMIDI,
        expansion_bars: int = 4,
        development_type: str = "sequence",
    ) -> pretty_midi.PrettyMIDI:
        _tempo_times, tempos = motif_midi.get_tempo_changes()
        initial_tempo = float(tempos[0]) if len(tempos) > 0 else 120.0
        expanded_midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

        for instrument in motif_midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program, is_drum=instrument.is_drum
            )

            if len(instrument.notes) == 0:
                continue

            motif_duration = max(note.end for note in instrument.notes)

            for bar in range(expansion_bars):
                offset = bar * motif_duration

                for note in instrument.notes:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start + offset,
                        end=note.end + offset,
                    )

                    if development_type == "sequence" and bar > 0:
                        if not instrument.is_drum:
                            new_note.pitch = np.clip(
                                note.pitch + bar * 2, 0, 127
                            )

                    elif development_type == "inversion" and bar % 2 == 1:
                        if not instrument.is_drum:
                            center_pitch = 60
                            interval = note.pitch - center_pitch
                            new_note.pitch = np.clip(
                                center_pitch - interval, 0, 127
                            )

                    elif development_type == "augmentation" and bar > 0:
                        duration_scale = 1.5
                        new_note.start = offset + (note.start * duration_scale)
                        new_note.end = offset + (note.end * duration_scale)

                    new_instrument.notes.append(new_note)

            expanded_midi.instruments.append(new_instrument)

        return expanded_midi


class StyleTransfer:
    def __init__(self):
        self.style_features = {}

    def extract_style_features(self, midi_data: pretty_midi.PrettyMIDI) -> dict:
        features = {
            "avg_velocity": [],
            "note_density": 0,
            "pitch_range": [127, 0],
            "rhythm_complexity": 0,
        }

        total_notes = 0
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                features["avg_velocity"].append(note.velocity)
                features["pitch_range"][0] = min(
                    features["pitch_range"][0], note.pitch
                )
                features["pitch_range"][1] = max(
                    features["pitch_range"][1], note.pitch
                )
                total_notes += 1

        if len(features["avg_velocity"]) > 0:
            features["avg_velocity"] = np.mean(features["avg_velocity"])
        else:
            features["avg_velocity"] = 80

        total_duration = midi_data.get_end_time()
        if total_duration > 0:
            features["note_density"] = total_notes / total_duration

        return features

    def apply_style(
        self, content_midi: pretty_midi.PrettyMIDI, style_features: dict
    ) -> pretty_midi.PrettyMIDI:
        _tempo_times, tempos = content_midi.get_tempo_changes()
        initial_tempo = float(tempos[0]) if len(tempos) > 0 else 120.0
        styled_midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

        for instrument in content_midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program, is_drum=instrument.is_drum
            )

            for note in instrument.notes:
                new_note = pretty_midi.Note(
                    velocity=int(style_features["avg_velocity"]),
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
                new_instrument.notes.append(new_note)

            styled_midi.instruments.append(new_instrument)

        return styled_midi
