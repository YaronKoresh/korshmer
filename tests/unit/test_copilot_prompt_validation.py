import json

import pytest

from korshmer.ai.copilot import MusicalContext, MusicCopilot


def strict(text):
    def refuse(token):
        raise ValueError(f"not JSON: {token}")

    return json.loads(text, parse_constant=refuse)


@pytest.mark.parametrize("bpm", [float("nan"), float("inf"), float("-inf")])
def test_prompt_never_carries_a_non_json_number(bpm):
    prompt = MusicCopilot()._build_prompt(MusicalContext(bpm=bpm), "hi")
    assert strict(prompt)["context"]["bpm"] == 120.0


def test_prompt_keeps_a_normal_tempo():
    prompt = MusicCopilot()._build_prompt(MusicalContext(bpm=96.5), "hi")
    assert strict(prompt)["context"]["bpm"] == 96.5


@pytest.mark.parametrize("bad", [None, "C G", 5, ("C", "G")])
def test_chord_progression_must_be_a_list_of_text(bad):
    with pytest.raises(ValueError, match="chord progression"):
        MusicCopilot()._build_prompt(MusicalContext(chord_progression=bad), "x")


def test_chord_progression_items_must_be_text():
    with pytest.raises(ValueError, match="chord progression"):
        MusicCopilot()._build_prompt(
            MusicalContext(chord_progression=["C", None]), "x"
        )


@pytest.mark.parametrize("request_text", [None, 5, ["a"]])
def test_request_must_be_text(request_text):
    with pytest.raises(ValueError, match="request"):
        MusicCopilot()._build_prompt(MusicalContext(), request_text)
