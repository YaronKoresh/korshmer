import pytest
from PIL import Image

from korshmer.video import text_render


@pytest.mark.parametrize(
    ("logical", "visual"),
    [
        ("Hello", "Hello"),
        ("", ""),
        ("פסנתר", "רתנספ"),
        ("שיר חדש (גרסה 2) by Ari", "by Ari (2 הסרג) שדח ריש"),
        ("Piano - פסנתר", "Piano - רתנספ"),
        ("Track: שיר 2 - mix", "Track: 2 ריש - mix"),
        ("מנגינה 120 BPM", "BPM 120 הניגנמ"),
        ("שורה\nline", "הרוש\nline"),
    ],
)
def test_visual_order(logical, visual):
    assert text_render.visual_order(logical) == visual


def test_hebrew_is_drawn_with_real_glyphs_not_boxes(monkeypatch):
    if text_render.hebrew_font_path() is None:
        pytest.skip("no Hebrew-capable font installed")
    font = text_render.load_font(32)
    assert text_render._has_hebrew(font)
    monkeypatch.setattr(text_render, "_native_bidi", lambda: False)

    def render(text):
        image = Image.new("L", (300, 60))
        text_render.draw(image).text((5, 5), text, font=font, fill=255)
        return image.tobytes()

    assert render("אבג") != render("שתק")


def test_drawing_does_not_reorder_when_pillow_has_raqm(monkeypatch):
    seen = []
    monkeypatch.setattr(text_render, "_native_bidi", lambda: True)
    monkeypatch.setattr(
        "PIL.ImageDraw.ImageDraw.text",
        lambda self, xy, text, *a, **k: seen.append(text),
    )
    text_render.draw(Image.new("L", (10, 10))).text((0, 0), "שיר")
    assert seen == ["שיר"]
