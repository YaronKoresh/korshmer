import logging

from korshmer.midi_processing.score_notation import (
    Articulation,
    Clef,
    Dynamic,
    DynamicsMapper,
    Ornament,
)

_logger = logging.getLogger(__name__)


class ScoreNotationViewModel:
    def __init__(self):
        self.measures = []
        self.clef = Clef.TREBLE
        self._dynamics_mapper = DynamicsMapper()

    def set_measures(self, measures):
        self.measures = measures

    def transpose_notes(self, notes, semitones):
        modified = []
        for note in notes:
            new_pitch = note.pitch + semitones
            if 0 <= new_pitch <= 127:
                note.pitch = new_pitch
                modified.append(note)
        if modified:
            _logger.info(
                f"Transposed {len(modified)} notes by {semitones} semitones"
            )
        return modified

    def add_articulation(self, notes, articulation):
        modified = []
        if isinstance(articulation, str):
            articulation = Articulation(articulation)
        for note in notes:
            if articulation not in note.articulations:
                note.articulations.append(articulation)
                modified.append(note)
        if modified:
            _logger.info(f"Added {articulation.value} to {len(modified)} notes")
        return modified

    def change_dynamic(self, notes, dynamic):
        if isinstance(dynamic, str):
            dynamic = Dynamic(dynamic)
        new_velocity = self._dynamics_mapper.dynamic_to_velocity(dynamic)
        for note in notes:
            note.velocity = new_velocity
        _logger.info(
            f"Changed dynamics to {dynamic.value} for {len(notes)} notes"
        )
        return notes

    def add_ornament(self, notes, ornament):
        if isinstance(ornament, str):
            ornament = Ornament(ornament)
        for note in notes:
            note.ornament = ornament
        _logger.info(f"Added {ornament.value} to {len(notes)} notes")
        return notes

    def toggle_tie(self, notes):
        for note in notes:
            note.is_tied = not note.is_tied
        _logger.info(f"Toggled tie for {len(notes)} notes")
        return notes

    def set_clef(self, clef):
        if isinstance(clef, str):
            clef = Clef(clef)
        self.clef = clef
        _logger.info(f"Changed clef to {clef.value}")

    def note_position(self, note, x_offset, _measure_width=200):
        staff_pos = note.get_staff_position(self.clef)
        y_pos = 140 - (staff_pos * 5)
        x_pos = x_offset + (note.start_time * 40)
        return x_pos, y_pos
