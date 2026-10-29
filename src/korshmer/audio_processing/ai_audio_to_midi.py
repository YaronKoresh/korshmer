import librosa
import numpy as np
import pretty_midi

from korshmer.midi_algorithms import normalize_note_velocities
from korshmer.utils.perf_metrics import measure


def _report(progress, fraction, message):
    if progress is None:
        return
    try:
        progress(fraction, message)
    except Exception:
        pass


class EnhancedAudioToMIDI:
    MODEL_FRAME_RATE = 100

    def __init__(self):

        from korshmer.exceptions import AudioToMidiError

        try:
            from basic_pitch.inference import predict as basic_pitch_predict
        except ImportError as err:
            raise AudioToMidiError(
                "basic-pitch is required for AI audio-to-MIDI features. "
                "Please install it with: pip install basic-pitch"
            ) from err

        self.basic_pitch_predict = basic_pitch_predict
        self.confidence_threshold = 0.5
        self.onset_threshold = 0.4
        self.frame_threshold = 0.2
        self.minimum_note_length = 40
        self.velocity_sensitivity = 1.0

    @measure("audio.midi_transcription")
    def transcribe_with_confidence(
        self,
        audio_path: str,
        midi_tempo: float = 120.0,
        confidence_threshold: float = 0.5,
        instrument_type: str = "general",
        progress=None,
    ) -> tuple[pretty_midi.PrettyMIDI, np.ndarray]:
        freq_range = self._get_instrument_frequency_range(instrument_type)
        _report(progress, 0.0, "Starting MIDI transcription")

        model_output, midi_data, note_events = self.basic_pitch_predict(
            audio_path,
            midi_tempo=midi_tempo,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.minimum_note_length,
            minimum_frequency=freq_range[0],
            maximum_frequency=freq_range[1],
        )

        _report(progress, 0.7, "Transcription model finished")
        confidence_scores = self._calculate_confidence_scores(
            model_output, note_events
        )
        filtered_midi = self._filter_by_confidence(
            midi_data, confidence_scores, confidence_threshold
        )
        filtered_midi = normalize_note_velocities(filtered_midi)
        _report(progress, 1.0, "Filtered notes by confidence")

        return filtered_midi, confidence_scores

    def _get_instrument_frequency_range(
        self, instrument_type: str
    ) -> tuple[int, int]:
        ranges = {
            "piano": (27, 4200),
            "guitar": (82, 1319),
            "bass": (41, 392),
            "drums": (30, 8000),
            "vocals": (80, 1100),
            "general": (30, 4000),
        }
        return ranges.get(instrument_type, ranges["general"])

    def _calculate_confidence_scores(
        self, model_output: np.ndarray, note_events: list
    ) -> np.ndarray:
        if len(note_events) == 0:
            return np.array([])

        confidences = []
        for event in note_events:
            start_frame = int(event["start_time_s"] * self.MODEL_FRAME_RATE)
            end_frame = int(event["end_time_s"] * self.MODEL_FRAME_RATE)
            pitch_idx = event["pitch_midi"]

            if (
                start_frame < end_frame
                and start_frame < model_output.shape[0]
                and end_frame <= model_output.shape[0]
                and 0 <= pitch_idx < model_output.shape[1]
            ):
                note_confidence = np.mean(
                    model_output[start_frame:end_frame, pitch_idx]
                )
                confidences.append(note_confidence)
            else:
                confidences.append(0.5)

        return np.array(confidences)

    def _filter_by_confidence(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        confidence_scores: np.ndarray,
        threshold: float,
    ) -> pretty_midi.PrettyMIDI:
        if len(confidence_scores) == 0:
            return midi_data

        filtered_midi = pretty_midi.PrettyMIDI(
            initial_tempo=midi_data.get_tempo_changes()[1][0]
        )

        for instrument in midi_data.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program, is_drum=instrument.is_drum
            )

            for idx, note in enumerate(instrument.notes):
                if (
                    idx < len(confidence_scores)
                    and confidence_scores[idx] >= threshold
                ):
                    new_instrument.notes.append(note)

            if len(new_instrument.notes) > 0:
                filtered_midi.instruments.append(new_instrument)

        return filtered_midi

    def estimate_velocity_from_audio(
        self,
        audio_path: str,
        midi_data: pretty_midi.PrettyMIDI,
        sensitivity: float = 1.0,
    ) -> pretty_midi.PrettyMIDI:
        audio, sr = librosa.load(audio_path, sr=None)

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start_sample = int(note.start * sr)
                end_sample = int(note.end * sr)

                if start_sample < len(audio) and end_sample <= len(audio):
                    note_audio = audio[start_sample:end_sample]
                    rms_energy = np.sqrt(np.mean(note_audio**2))

                    velocity = int(
                        np.clip(rms_energy * 127 * sensitivity, 1, 127)
                    )
                    note.velocity = velocity

        return midi_data

    def refine_note_timing(
        self, audio_path: str, midi_data: pretty_midi.PrettyMIDI
    ) -> pretty_midi.PrettyMIDI:
        audio, sr = librosa.load(audio_path, sr=None)
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        if len(onset_times) == 0:
            return midi_data

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                closest_onset_idx = np.argmin(np.abs(onset_times - note.start))
                closest_onset_time = onset_times[closest_onset_idx]

                if abs(closest_onset_time - note.start) < 0.05:
                    note.start = closest_onset_time

        return midi_data

    def correct_octave_errors(
        self, midi_data: pretty_midi.PrettyMIDI, audio_path: str
    ) -> pretty_midi.PrettyMIDI:
        audio, sr = librosa.load(audio_path, sr=None)

        f0_all, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C8"),
            sr=sr,
        )
        f0_times = librosa.times_like(f0_all, sr=sr)

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note_duration_samples = int((note.end - note.start) * sr)
                if note_duration_samples <= 512:
                    continue

                mask = (f0_times >= note.start) & (f0_times <= note.end)
                if not np.any(mask):
                    continue

                note_f0 = f0_all[mask]
                valid_f0 = note_f0[~np.isnan(note_f0)]
                if len(valid_f0) > 0:
                    median_f0 = np.median(valid_f0)
                    detected_midi = librosa.hz_to_midi(median_f0)

                    current_pitch_class = note.pitch % 12
                    detected_pitch_class = int(detected_midi) % 12

                    if current_pitch_class == detected_pitch_class:
                        octave_diff = round((detected_midi - note.pitch) / 12)
                        if abs(octave_diff) > 0:
                            new_pitch = note.pitch + (octave_diff * 12)
                            if 0 <= new_pitch <= 127:
                                note.pitch = new_pitch

        return midi_data

    def remove_spurious_notes(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        min_duration: float = 0.05,
        max_gap: float = 0.1,
    ) -> pretty_midi.PrettyMIDI:
        for instrument in midi_data.instruments:
            filtered_notes = []

            for note in instrument.notes:
                duration = note.end - note.start
                if duration >= min_duration:
                    filtered_notes.append(note)

            filtered_notes.sort(key=lambda n: n.start)

            merged_notes = []
            if len(filtered_notes) > 0:
                current_note = filtered_notes[0]

                for next_note in filtered_notes[1:]:
                    if (
                        next_note.pitch == current_note.pitch
                        and next_note.start - current_note.end <= max_gap
                    ):
                        current_note.end = next_note.end
                    else:
                        merged_notes.append(current_note)
                        current_note = next_note

                merged_notes.append(current_note)

            instrument.notes = merged_notes

        return midi_data

    def polyphonic_separation(
        self,
        audio_path: str,
        midi_tempo: float = 120.0,
        max_polyphony: int = 4,
        progress=None,
    ) -> pretty_midi.PrettyMIDI:
        _report(progress, 0.0, "Starting polyphonic transcription")
        audio, sr = librosa.load(audio_path, sr=None)

        _, midi_data, _ = self.basic_pitch_predict(
            audio_path,
            midi_tempo=midi_tempo,
            onset_threshold=0.3,
            frame_threshold=0.1,
        )
        _report(progress, 0.7, "Transcription model finished")

        for instrument in midi_data.instruments:
            notes = instrument.notes
            notes.sort(key=lambda n: (n.start, -n.pitch))

            separated_notes = []
            for _i, note in enumerate(notes):
                overlapping = [
                    n
                    for n in notes
                    if n.start <= note.start < n.end and n != note
                ]

                if len(overlapping) < max_polyphony:
                    separated_notes.append(note)

            instrument.notes = separated_notes

        _report(progress, 1.0, "Limited polyphony")
        return midi_data


class EnsembleTranscription:
    def __init__(self):
        self.models = ["basic-pitch"]
        self.weights = [1.0]

    def transcribe_ensemble(
        self, audio_path: str, midi_tempo: float = 120.0, progress=None
    ) -> pretty_midi.PrettyMIDI:
        basic_pitch_midi, _ = EnhancedAudioToMIDI().transcribe_with_confidence(
            audio_path,
            midi_tempo=midi_tempo,
            confidence_threshold=0.3,
            progress=progress,
        )

        return basic_pitch_midi

    def merge_transcriptions(
        self,
        midi_list: list[pretty_midi.PrettyMIDI],
        weights: list[float] | None = None,
    ) -> pretty_midi.PrettyMIDI:
        if weights is None:
            weights = [1.0] * len(midi_list)

        if len(midi_list) == 0:
            return pretty_midi.PrettyMIDI()

        merged = pretty_midi.PrettyMIDI()

        all_notes = []
        for idx, midi_obj in enumerate(midi_list):
            for instrument in midi_obj.instruments:
                for note in instrument.notes:
                    all_notes.append(
                        {
                            "pitch": note.pitch,
                            "start": note.start,
                            "end": note.end,
                            "velocity": note.velocity,
                            "weight": weights[idx],
                        }
                    )

        all_notes.sort(key=lambda n: (n["start"], n["pitch"]))

        merged_instrument = pretty_midi.Instrument(program=0)

        i = 0
        while i < len(all_notes):
            current = all_notes[i]
            similar_notes = [current]

            j = i + 1
            while j < len(all_notes):
                next_note = all_notes[j]
                if (
                    abs(next_note["start"] - current["start"]) < 0.05
                    and abs(next_note["pitch"] - current["pitch"]) <= 1
                ):
                    similar_notes.append(next_note)
                    j += 1
                else:
                    break

            if len(similar_notes) >= len(midi_list) / 2:
                avg_pitch = int(np.average([n["pitch"] for n in similar_notes]))
                avg_start = np.average([n["start"] for n in similar_notes])
                avg_end = np.average([n["end"] for n in similar_notes])
                avg_velocity = int(
                    np.average([n["velocity"] for n in similar_notes])
                )

                note = pretty_midi.Note(
                    velocity=avg_velocity,
                    pitch=avg_pitch,
                    start=avg_start,
                    end=avg_end,
                )
                merged_instrument.notes.append(note)

            i = j

        merged.instruments.append(merged_instrument)
        return merged
