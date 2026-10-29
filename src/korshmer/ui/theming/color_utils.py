import math

from PySide6.QtGui import QColor, QGradient, QLinearGradient, QRadialGradient


def argb(rgba_hex):
    if rgba_hex.startswith("#"):
        rgba_hex = rgba_hex[1:]

    if len(rgba_hex) != 8:
        return "#" + rgba_hex

    r = rgba_hex[0:2]
    g = rgba_hex[2:4]
    b = rgba_hex[4:6]
    a = rgba_hex[6:8]

    argb_hex = f"#{a}{r}{g}{b}"

    return argb_hex


def gradient(*stops, deg=-1):
    if deg == -1:
        g = QRadialGradient(0.5, 0.5, 0.5)
    else:
        angle_rad = math.radians((deg - 90) % 360)

        r = 0.7071
        x1 = 0.5 - r * math.cos(angle_rad)
        y1 = 0.5 - r * math.sin(angle_rad)
        x2 = 0.5 + r * math.cos(angle_rad)
        y2 = 0.5 + r * math.sin(angle_rad)

        g = QLinearGradient(x1, y1, x2, y2)

    g.setCoordinateMode(QGradient.ObjectBoundingMode)

    parsed_stops = []
    for s in stops:
        if (
            isinstance(s, (tuple, list))
            and len(s) == 2
            and isinstance(s[0], (int, float))
        ):
            parsed_stops.append({"pos": float(s[0]), "col": QColor(s[1])})
        else:
            col_val = s[1] if (isinstance(s, tuple) and len(s) == 2) else s
            parsed_stops.append({"pos": None, "col": QColor(col_val)})

    if not parsed_stops:
        return g

    if parsed_stops[0]["pos"] is None:
        parsed_stops[0]["pos"] = 0.0
    if parsed_stops[-1]["pos"] is None:
        parsed_stops[-1]["pos"] = 1.0

    i = 0
    while i < len(parsed_stops) - 1:
        curr = parsed_stops[i]

        j = i + 1
        while j < len(parsed_stops) and parsed_stops[j]["pos"] is None:
            j += 1

        if j < len(parsed_stops):
            start_pos = curr["pos"]
            end_pos = parsed_stops[j]["pos"]
            steps = j - i
            step_size = (end_pos - start_pos) / steps

            for k in range(1, steps):
                parsed_stops[i + k]["pos"] = start_pos + (k * step_size)

        i = j

    for item in parsed_stops:
        g.setColorAt(item["pos"], item["col"])

    return g


MAT_TEXT = argb("#111111BB")
MAT_BG = argb("#FFFFFFBB")
MAT_HOVER = "#DDD"
MAT_PRIMARY = "#1A73E8"
MAT_ACCENT = "#00FFFF"
MAT_SUCCESS = "#34A853"
MAT_WARNING = "#FBBC04"
MAT_ERROR = "#EA4335"
MAT_BORDER = "#222"


BPM_SPINBOX_NORMAL = """
    QSpinBox {
        background-color: #1a1a1a;
        color: #00FFFF;
        padding: 5px;
        border-radius: 4px;
        border: 1px solid #333;
        font-size: 14px;
        font-weight: bold;
    }
    QSpinBox:hover { border-color: #00FFFF; }
"""

BPM_SPINBOX_HIGHLIGHTED = """
    QSpinBox {
        background-color: #505050;
        color: #00FFFF;
        padding: 5px;
        border-radius: 4px;
        border: 2px solid #00FFFF;
        font-size: 14px;
        font-weight: bold;
    }
    QSpinBox:hover { border-color: #7c4dff; }
"""

BEAT_INDICATOR_OFF = "background-color: #444; border-radius: 8px;"
BEAT_INDICATOR_ON = f"background-color: {MAT_PRIMARY}; border-radius: 8px;"
