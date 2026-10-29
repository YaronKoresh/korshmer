import re
from pathlib import Path

from korshmer import __version__

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src" / "korshmer"


class TestVersionString:
    def test_version_is_string(self):
        assert isinstance(__version__, str)

    def test_version_follows_semver(self):
        assert re.match(r"^\d+\.\d+\.\d+$", __version__), (
            f"Version {__version__!r} does not follow semver (X.Y.Z)"
        )

    def test_version_not_empty(self):
        assert len(__version__) > 0


class TestVersionConsistencyInPyproject:
    def test_pyproject_version_matches(self):
        pyproject_path = ROOT_DIR / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"(.+?)"', content, re.MULTILINE)
        assert match is not None, "Version not found in pyproject.toml"
        assert match.group(1) == __version__, (
            f"pyproject.toml version {match.group(1)!r} "
            f"does not match __version__ {__version__!r}"
        )


class TestNoHardcodedVersionStrings:
    def _read_py_file(self, path):
        return path.read_text(encoding="utf-8")

    def test_titlebar_uses_dynamic_version(self):
        content = self._read_py_file(SRC_DIR / "ui" / "widgets" / "titlebar.py")
        assert "v0.5.1" not in content, (
            "titlebar.py still contains hardcoded v0.5.1"
        )
        assert "__version__" in content, "titlebar.py should use __version__"

    def test_main_help_uses_dynamic_version(self):
        content = self._read_py_file(SRC_DIR / "__main__.py")
        assert "v0.5.1" not in content, (
            "__main__.py still contains hardcoded v0.5.1"
        )

    def test_logging_setup_uses_dynamic_version(self):
        content = self._read_py_file(
            SRC_DIR / "ui" / "utils" / "logging_setup.py"
        )
        assert '"Version: 0.6.0"' not in content, (
            "logging_setup.py still contains hardcoded 0.6.0"
        )
        assert "__version__" in content

    def test_logging_config_uses_dynamic_version(self):
        content = self._read_py_file(SRC_DIR / "utils" / "logging_config.py")
        assert '"Version: 0.6.0"' not in content, (
            "logging_config.py still contains hardcoded 0.6.0"
        )
        assert "__version__" in content

    def test_project_manager_uses_dynamic_version(self):
        content = self._read_py_file(
            SRC_DIR / "project_management" / "project_manager.py"
        )
        assert '"version": "0.6.0"' not in content, (
            "project_manager.py still contains hardcoded 0.6.0"
        )
        assert "__version__" in content

    def test_migration_action_uses_dynamic_version(self):
        content = self._read_py_file(
            SRC_DIR / "mixins" / "actions" / "project_management.py"
        )
        assert '"0.6.0"' not in content, (
            "project_management action still contains hardcoded 0.6.0"
        )
        assert "__version__" in content
