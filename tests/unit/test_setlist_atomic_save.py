from __future__ import annotations

import inspect

from korshmer.project_management.setlist import Setlist


def test_setlist_save_uses_atomic_replace():
    source = inspect.getsource(Setlist.save)
    assert "os.replace" in source, (
        "Setlist.save should use os.replace for atomic write"
    )
    assert ".tmp" in source
