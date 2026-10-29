import numpy as np
import pytest

from korshmer.plugins.base import PluginParam


def _int():
    return PluginParam("stages", "Stages", "int", 4, 2, 12)


def _float():
    return PluginParam("width", "Width", "float", 1.0, 0, 3.0)


@pytest.mark.parametrize("value", [2, 7, 12, np.int64(5), 4.0])
def test_int_param_accepts_whole_numbers_in_range(value):
    assert _int().validate(value) is True


@pytest.mark.parametrize("value", [3.5, 1, 13, True, False, float("nan")])
def test_int_param_rejects_fractions_out_of_range_and_bools(value):
    assert _int().validate(value) is False


@pytest.mark.parametrize("value", [0, 1.5, 3, np.float32(2.5)])
def test_float_param_accepts_numbers_in_range(value):
    assert _float().validate(value) is True


@pytest.mark.parametrize(
    "value",
    ["1", None, [1], (1,), {}, float("nan"), float("inf"), -0.1, 3.1, True],
)
def test_float_param_rejects_everything_else_without_raising(value):
    assert _float().validate(value) is False


@pytest.mark.parametrize("value", ["x", None, 1, [], 0.5])
def test_choice_param_checks_membership(value):
    param = PluginParam("shape", "Shape", "choice", "sine", choices=["x"])
    assert param.validate(value) is (value == "x")


def test_bool_param_accepts_only_bools():
    param = PluginParam("on", "On", "bool", True)
    assert param.validate(True) is True
    assert param.validate(0) is False
