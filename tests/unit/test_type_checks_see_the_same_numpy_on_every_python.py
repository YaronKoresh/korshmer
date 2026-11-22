import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[2]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]


def _numpy(requirements):
    return [
        requirement.replace(" ", "")
        for requirement in requirements
        if re.match(r"numpy\b", requirement)
    ]


def test_the_type_check_tools_pin_one_exact_numpy_version():
    pinned = _numpy(PROJECT["optional-dependencies"]["lint"])

    assert len(pinned) == 1
    assert re.fullmatch(r"numpy==\d+\.\d+\.\d+", pinned[0])


def test_the_pinned_numpy_satisfies_the_runtime_requirement():
    pinned = _numpy(PROJECT["optional-dependencies"]["lint"])[0]
    lowest = _numpy(PROJECT["dependencies"])[0].removeprefix("numpy>=")
    version = tuple(int(part) for part in pinned.split("==")[1].split("."))

    assert version >= tuple(int(part) for part in lowest.split("."))
