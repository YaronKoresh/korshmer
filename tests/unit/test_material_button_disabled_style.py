import re

import pytest

from korshmer.ui.widgets.material.button import (
    DISABLED_BG,
    DISABLED_TEXT,
    MaterialButton,
)

ROLES = ["default", "primary", "success", "danger", "error", "tonal"]


def _luminance(hex_color):
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_disabled_text_is_readable_on_disabled_background():
    assert _contrast(DISABLED_BG, DISABLED_TEXT) >= 4.5


def test_disabled_background_is_not_near_black():
    assert _contrast(DISABLED_BG, "#000000") >= 7.0


@pytest.mark.parametrize("role", ROLES)
def test_stylesheet_uses_disabled_colors(qtbot, role):
    button = MaterialButton("Undo", role)
    qtbot.addWidget(button)
    sheet = button.styleSheet()
    block = re.search(r"QPushButton:disabled\s*\{(.*?)\}", sheet, re.S).group(1)
    assert DISABLED_BG in block
    assert DISABLED_TEXT in block
    assert "#2a2a2a" not in sheet.lower()
    assert "#666666" not in sheet.lower()
