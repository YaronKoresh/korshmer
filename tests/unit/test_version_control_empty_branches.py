from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from korshmer.project_management.version_control import VersionControlManager


def test_get_branches_skips_empty_strings():
    mgr = VersionControlManager.__new__(VersionControlManager)
    mgr.base_dir = Path("/tmp/fake")
    mgr.repo_dir = Path("/tmp/fake/saved_data")
    mgr.enabled = True

    result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="\n", stderr=""
    )
    with patch("subprocess.run", return_value=result):
        branches = mgr.get_branches()
    assert branches == [], "Empty output should produce no branch entries"
