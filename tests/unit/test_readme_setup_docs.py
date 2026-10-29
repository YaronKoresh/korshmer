import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _minimum_python():
    match = re.search(r'requires-python\s*=\s*">=(\d+\.\d+)"', PYPROJECT)
    assert match
    return match.group(1)


def test_readme_python_version_matches_pyproject():
    minimum = _minimum_python()
    versions = re.findall(r"Python (\d+\.\d+)(?:\+| or higher)", README)
    assert versions
    assert set(versions) == {minimum}


def test_changelog_python_version_matches_pyproject():
    versions = re.findall(
        r"Primary Language:\*\* Python \(>= (\d+\.\d+)\)", CHANGELOG
    )
    assert versions == [_minimum_python()]


def test_readme_scripts_exist():
    names = set(re.findall(r"scripts/([\w.]+\.(?:bat|py))", README))
    assert names
    for name in names:
        assert (ROOT / "scripts" / name).is_file(), name


def test_readme_quality_tools_are_project_tools():
    assert "black" not in README
    assert "flake8" not in README
    assert "ruff format" in README
    assert "ruff check" in README
    assert '"ruff"' in PYPROJECT
