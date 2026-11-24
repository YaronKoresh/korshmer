import numpy as np
import pytest

from korshmer.analysis import rank_scales
from korshmer.constants import SCALES


def _plain_major_notes(root, loudness=1.0):
    notes = np.zeros(12)
    notes[SCALES["Major"]] = loudness
    return np.roll(notes, root)


@pytest.mark.parametrize("root", range(12))
def test_notes_with_no_centre_are_named_by_their_minor_in_every_key(root):
    named = rank_scales(_plain_major_notes(root), 1)[0][:2]

    assert named == ((root + 9) % 12, "Minor")


@pytest.mark.parametrize("loudness", [0.1, 1 / 7, 1 / 3, 1.1, 1.3, 3.0, 100.0])
def test_how_loud_the_notes_are_does_not_change_the_key_name(loudness):
    named = rank_scales(_plain_major_notes(0, loudness), 1)[0][:2]

    assert named == (9, "Minor")
