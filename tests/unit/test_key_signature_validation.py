import numpy as np
import pytest

from korshmer.midi_processing.midi_file_enhancement import (
    MIDIFileEnhancer,
    validate_key_signature,
)


@pytest.mark.parametrize("key", [-7, -1, 0, 3, 7])
def test_valid_keys_are_stored(key):
    e = MIDIFileEnhancer()
    e.add_key_signature(1.0, key, False)
    assert e.get_key_at_time(1.0) == (key, False)


@pytest.mark.parametrize(
    "key",
    [8, -8, 100, 2.5, float("nan"), float("inf"), "x", None, True, [1]],
)
def test_invalid_key_is_rejected_and_not_stored(key):
    e = MIDIFileEnhancer()
    with pytest.raises(ValueError):
        e.add_key_signature(0.0, key)
    assert e.key_signatures == []


@pytest.mark.parametrize("mode", [None, "major", 1, 0.0])
def test_invalid_mode_is_rejected(mode):
    e = MIDIFileEnhancer()
    with pytest.raises(ValueError):
        e.add_key_signature(0.0, 0, mode)
    assert e.key_signatures == []


def test_numpy_values_are_normalized():
    assert validate_key_signature(np.int64(-3), np.bool_(True)) == (-3, True)
    key, mode = validate_key_signature(np.int64(2), True)
    assert type(key) is int and type(mode) is bool


def test_invalid_time_still_skips_silently():
    e = MIDIFileEnhancer()
    e.add_key_signature(float("nan"), 100)
    assert e.key_signatures == []
