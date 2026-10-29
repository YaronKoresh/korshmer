import os
import sys
from pathlib import Path

PREFERRED_FONTS = (
    "tahoma",
    "tahomabd",
    "segoeui",
    "segoeuib",
    "arial",
    "arialbd",
    "dejavusans",
    "dejavusans-bold",
    "liberationsans-regular",
    "liberationsans-bold",
)


def _font_roots():
    if sys.platform == "win32":
        return (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Fonts",)
    return (
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
    )


def system_font_files():
    found = {}
    for root in _font_roots():
        if root.is_dir():
            for path in sorted(root.rglob("*.ttf")):
                found.setdefault(path.stem.lower(), path)
    preferred = [found[name] for name in PREFERRED_FONTS if name in found]
    return preferred or list(found.values())[:2]


def has_box_glyphs():
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QApplication

    metrics = QFontMetrics(QApplication.font())
    return (
        metrics.height()
        == metrics.horizontalAdvance("i")
        == metrics.horizontalAdvance("M")
    )


def ensure_real_fonts():
    from PySide6.QtGui import QFontDatabase

    if not has_box_glyphs():
        return True
    for path in system_font_files():
        QFontDatabase.addApplicationFont(str(path))
    return not has_box_glyphs()
