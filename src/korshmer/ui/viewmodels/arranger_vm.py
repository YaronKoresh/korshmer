import math
import os
import time

import pretty_midi

from korshmer.constants import DATA_DIR
from korshmer.ui.viewmodels.editor_vm import clamp_velocity
from korshmer.ui.viewmodels.piano_roll_vm import NoteData
from korshmer.utils.midi_loading import load_pretty_midi


class ArrangerViewModel:
    def __init__(self):
        self.is_recording = False

    def parse_midi(self, path, tempo, pixels_per_second, grid_size):
        pm = load_pretty_midi(path, initial_tempo=tempo)
        beat_width = (60.0 / tempo) * pixels_per_second
        notes = []
        for inst in pm.instruments:
            for note in inst.notes:
                notes.append(
                    NoteData(
                        pitch=note.pitch,
                        x=note.start * pixels_per_second,
                        width=(note.end - note.start) * pixels_per_second,
                        velocity=note.velocity,
                        grid_x=beat_width,
                        grid_y=grid_size,
                    )
                )
        return notes

    def build_midi(self, note_items, tempo, pixels_per_second, grid_size):
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        inst = pretty_midi.Instrument(program=0, is_drum=False, name="Editor")

        for item in note_items:
            start_time = item["x"] / pixels_per_second
            duration = item["width"] / pixels_per_second
            pitch = int(127 - (math.ceil(item["y"]) // grid_size))
            pitch = max(0, min(127, pitch))
            velocity = clamp_velocity(item.get("velocity", 100))

            inst.notes.append(
                pretty_midi.Note(
                    velocity, pitch, start_time, start_time + duration
                )
            )

        pm.instruments.append(inst)
        return pm

    def recording_filename(self):
        return os.path.join(DATA_DIR, f"Rec_{int(time.time())}.wav")

    def export_midi_filename(self):
        return os.path.join(DATA_DIR, f"Editor_{int(time.time())}.mid")

    def ensure_midi_suffix(self, path):
        if not path:
            return path
        lowered = path.lower()
        if lowered.endswith(".mid") or lowered.endswith(".midi"):
            return path
        return f"{path}.mid"
