import io
import pathlib
import re
import tokenize

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "korshmer"

MODEL_FILES = {
    "audio_processing/ai_audio_to_midi.py",
    "track_processor.py",
    "mixins/pages/tools/__init__.py",
}

WORD = re.compile(r"\bAI\b")
TEXT_TOKENS = {
    tokenize.STRING,
    getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING),
}


def _ai_strings():
    found = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        text = path.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in TEXT_TOKENS and WORD.search(tok.string):
                found.append((rel, tok.string))
    return found


def test_ai_is_only_named_where_a_model_runs():
    stray = [(f, s) for f, s in _ai_strings() if f not in MODEL_FILES]
    assert stray == []


def test_model_tools_keep_their_ai_label():
    labels = [s for f, s in _ai_strings() if f in MODEL_FILES]
    assert any("Transcri" in s for s in labels)


def test_deterministic_continuation_is_not_called_ai():
    from korshmer import midi_algorithms

    source = pathlib.Path(midi_algorithms.__file__).read_text(encoding="utf-8")
    assert "AI Continuation" not in source
    assert 'name="Continuation"' in source
