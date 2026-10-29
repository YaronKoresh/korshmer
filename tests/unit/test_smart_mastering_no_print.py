from __future__ import annotations

import ast
import inspect

from korshmer.audio_processing import smart_mastering


def test_no_bare_print_calls():
    source = inspect.getsource(smart_mastering)
    tree = ast.parse(source)
    prints = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert prints == [], (
        f"Found {len(prints)} bare print() calls in smart_mastering; use logger.debug()"
    )


def test_module_has_logger():
    assert hasattr(smart_mastering, "logger"), (
        "smart_mastering must define a module-level logger"
    )
