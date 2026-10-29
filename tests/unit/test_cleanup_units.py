import os
import tempfile

import pretty_midi


def _make_midi_with_notes(*durations_ms):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    for dur_ms in durations_ms:
        dur_sec = dur_ms / 1000.0
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=t, end=t + dur_sec)
        )
        t += dur_sec + 0.5
    pm.instruments.append(inst)
    return pm


class TestProcessCleanupUnits:
    def test_cleanup_treats_argument_as_milliseconds(self):
        from korshmer.midi_processors import process_cleanup

        pm = _make_midi_with_notes(50, 200, 500)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            pm.write(f.name)
            tmp = f.name
        try:
            result_path = process_cleanup(tmp, 100)
            try:
                result = pretty_midi.PrettyMIDI(result_path)
                remaining = len(result.instruments[0].notes)
                assert remaining == 2, (
                    f"Expected 2 notes (>=100ms), got {remaining}. "
                    "ms value may be treated as seconds."
                )
            finally:
                if os.path.exists(result_path):
                    os.unlink(result_path)
        finally:
            os.unlink(tmp)

    def test_cleanup_keeps_all_when_threshold_is_zero(self):
        from korshmer.midi_processors import process_cleanup

        pm = _make_midi_with_notes(10, 20, 30)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            pm.write(f.name)
            tmp = f.name
        try:
            result_path = process_cleanup(tmp, 0)
            try:
                result = pretty_midi.PrettyMIDI(result_path)
                assert len(result.instruments[0].notes) == 3
            finally:
                if os.path.exists(result_path):
                    os.unlink(result_path)
        finally:
            os.unlink(tmp)
