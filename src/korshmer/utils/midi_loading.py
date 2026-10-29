import pretty_midi


def load_pretty_midi(path, **kwargs):
    try:
        return pretty_midi.PrettyMIDI(path, **kwargs)
    except (EOFError, OSError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Invalid MIDI file: {exc!r}") from exc
