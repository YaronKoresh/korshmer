_CONTROL_LABELS = {
    "QAbstractButton": "Button",
    "QSlider": "Slider",
    "QSpinBox": "Number field",
    "QDoubleSpinBox": "Number field",
    "QComboBox": "Selection list",
}


def _derive_accessible_name(widget, kind):
    text = getattr(widget, "text", None)
    candidates = [text() if callable(text) else ""]
    candidates.append(widget.toolTip().split("\n", 1)[0])
    candidates.append(widget.objectName())
    for candidate in candidates:
        cleaned = (candidate or "").replace("&", "").strip()
        if cleaned:
            return cleaned
    return _CONTROL_LABELS[kind]


def apply_accessible_names(root):
    from PySide6.QtWidgets import (
        QAbstractButton,
        QComboBox,
        QDoubleSpinBox,
        QSlider,
        QSpinBox,
    )

    kinds = (
        (QAbstractButton, "QAbstractButton"),
        (QSlider, "QSlider"),
        (QDoubleSpinBox, "QDoubleSpinBox"),
        (QSpinBox, "QSpinBox"),
        (QComboBox, "QComboBox"),
    )
    named = 0
    for cls, kind in kinds:
        for widget in root.findChildren(cls):
            if widget.accessibleName().strip():
                continue
            widget.setAccessibleName(_derive_accessible_name(widget, kind))
            named += 1
    return named


class AccessibilityManager:
    def __init__(self):
        self.screen_reader_enabled = False
        self.high_contrast = False
        self.font_scale = 1.0
        self.audio_feedback = False
        self.animation_speed = 1.0
        self.colorblind_mode = "none"

    def enable_screen_reader(self):
        self.screen_reader_enabled = True

    def enable_high_contrast(self):
        self.high_contrast = True

    def set_font_scale(self, scale):
        self.font_scale = max(0.5, min(3.0, scale))

    def enable_audio_feedback(self):
        self.audio_feedback = True

    def set_animation_speed(self, speed):
        self.animation_speed = max(0.0, min(2.0, speed))

    def set_colorblind_mode(self, mode):
        valid_modes = ["none", "protanopia", "deuteranopia", "tritanopia"]
        if mode in valid_modes:
            self.colorblind_mode = mode

    def get_settings(self):
        return {
            "screen_reader": self.screen_reader_enabled,
            "high_contrast": self.high_contrast,
            "font_scale": self.font_scale,
            "audio_feedback": self.audio_feedback,
            "animation_speed": self.animation_speed,
            "colorblind_mode": self.colorblind_mode,
        }
