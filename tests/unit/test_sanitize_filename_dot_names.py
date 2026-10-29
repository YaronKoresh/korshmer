import pytest

from korshmer.ui.utils import file_operations
from korshmer.utils import filesystem

MODULES = [filesystem, file_operations]


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("name", ["..", ".", "...", " ", " . ", ". ."])
def test_names_without_real_characters_become_unnamed(module, name):
    assert module.sanitize_filename(name) == "unnamed"


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "name,expected",
    [("song.", "song"), ("song. . ", "song"), ("a/../b", "a_.._b")],
)
def test_trailing_dots_and_spaces_are_removed(module, name, expected):
    assert module.sanitize_filename(name) == expected


@pytest.mark.parametrize("module", MODULES)
def test_hidden_style_and_regular_names_are_kept(module):
    assert module.sanitize_filename(".hidden") == ".hidden"
    assert module.sanitize_filename("my song.wav") == "my song.wav"


@pytest.mark.parametrize("module", MODULES)
def test_truncation_never_leaves_trailing_dot(module):
    name = "a" * 254 + "." + "b" * 10
    result = module.sanitize_filename(name)
    assert len(result) <= 255
    assert not result.endswith((".", " "))


@pytest.mark.parametrize("module", MODULES)
def test_reserved_name_with_trailing_dot_is_prefixed_on_windows(
    monkeypatch, module
):
    monkeypatch.setattr(module.sys, "platform", "win32")
    assert module.sanitize_filename("CON.") == "file_CON"
