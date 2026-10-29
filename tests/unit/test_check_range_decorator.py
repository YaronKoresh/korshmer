import pytest

from korshmer.audio_processing.validation import check_range


@check_range(q=(0.1, 20), gain=(-12, 12))
def apply(audio, q=1.0, gain=0):
    return (audio, q, gain)


def test_value_in_range_runs_function_unchanged():
    assert apply("a", 5, gain=3) == ("a", 5, 3)


def test_bounds_are_inclusive():
    assert apply("a", 0.1)[1] == 0.1
    assert apply("a", 20)[1] == 20


@pytest.mark.parametrize("value", [0.05, 21, -1])
def test_out_of_range_positional_and_keyword(value):
    with pytest.raises(ValueError, match="q must be between 0.1 and 20"):
        apply("a", value)
    with pytest.raises(ValueError, match="q must be between 0.1 and 20"):
        apply("a", q=value)


def test_second_argument_checked():
    with pytest.raises(ValueError, match="gain"):
        apply("a", gain=13)


@pytest.mark.parametrize("value", [True, None, "5", float("nan"), float("inf")])
def test_non_numeric_bool_nan_inf_rejected(value):
    with pytest.raises(ValueError, match="q must be"):
        apply("a", q=value)


def test_unpassed_argument_uses_default_without_check():
    assert apply("a") == ("a", 1.0, 0)


def test_unknown_argument_name_fails_at_decoration():
    with pytest.raises(TypeError):
        check_range(nope=(0, 1))(lambda x: x)


def test_preserves_metadata():
    assert apply.__name__ == "apply"
