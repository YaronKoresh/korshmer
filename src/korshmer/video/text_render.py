
import functools
import os
import re
import sys

from PIL import ImageDraw, ImageFont, features

_HEBREW_FONT_FILES = (
             
    "segoeui.ttf",
    "arial.ttf",
    "tahoma.ttf",
           
    "NotoSansHebrew-Regular.ttf",
    "DejaVuSans.ttf",
    "FreeSans.ttf",
    "LiberationSans-Regular.ttf",
           
    "Arial.ttf",
    "ArialHB.ttc",
)

_RTL_CHARS = "\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\ufb1d-\ufdff\ufe70-\ufeff"
_RTL = re.compile(f"[{_RTL_CHARS}]")
_STRONG_LTR = re.compile(r"[A-Za-z0-9\u00c0-\u024f]")
_MIRROR = str.maketrans("()[]{}<>", ")(][}{><")


def _font_dirs():
    dirs = []
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(os.path.join(windir, "Fonts"))
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    elif sys.platform == "darwin":
        dirs += [
            "/System/Library/Fonts/Supplemental",
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    else:
        dirs += [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.local/share/fonts"),
            os.path.expanduser("~/.fonts"),
        ]
    return [d for d in dirs if os.path.isdir(d)]


def _has_hebrew(font):
                                                                       
                                         
    try:
        letter = font.getmask("\u05d0")
        missing = font.getmask("\U0010fffd")
        return letter.getbbox() is not None and (
            letter.size != missing.size or letter.tobytes() != missing.tobytes()
        )
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def hebrew_font_path():
    wanted = {
        name.lower(): rank for rank, name in enumerate(_HEBREW_FONT_FILES)
    }
    found = []
    for root in _font_dirs():
        for folder, _, files in os.walk(root):
            for name in files:
                rank = wanted.get(name.lower())
                if rank is not None:
                    found.append((rank, os.path.join(folder, name)))
    for _, path in sorted(found):
        try:
            if _has_hebrew(ImageFont.truetype(path, 20)):
                return path
        except OSError:
            continue
    return None


def load_font(size):
    path = hebrew_font_path()
    if path is not None:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


@functools.lru_cache(maxsize=1)
def _native_bidi():
    return bool(features.check("raqm"))


def visual_order(text):
    if not text or not _RTL.search(text):
        return text
    lines = text.split("\n")
    return "\n".join(_visual_line(line) for line in lines)


def _kind(token):
    if _RTL.match(token):
        return "rtl"
    if re.search("[A-Za-z\u00c0-\u024f]", token):
        return "ltr"
    if _STRONG_LTR.search(token):
        return "num"
    return ""


def _visual_line(line):
    if not _RTL.search(line):
        return line
    tokens = re.findall(
        f"[{_RTL_CHARS}]+|[A-Za-z0-9\u00c0-\u024f.,:%'+-]+|.", line
    )
    kinds = [_kind(tok) for tok in tokens]
    paragraph = next((k for k in kinds if k in ("rtl", "ltr")), "rtl")
    if paragraph == "ltr":
        return _ltr_paragraph(tokens, kinds)
                                                                       
                                                                        
    strong = ["rtl" if k == "num" else k for k in kinds]
    for i, kind in enumerate(kinds):
        if kind:
            continue
        before = next((k for k in reversed(strong[:i]) if k), "rtl")
        after = next((k for k in strong[i + 1 :] if k), "rtl")
        kinds[i] = "ltr" if before == after == "ltr" else "rtl"
    runs = []
    for kind, tok in zip(kinds, tokens, strict=True):
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(tok)
        else:
            runs.append((kind, [tok]))
    out = []
    for kind, toks in reversed(runs):
        if kind in ("ltr", "num"):
            out.append("".join(toks))
        else:
            out.append(
                "".join(
                    t[::-1].translate(_MIRROR)
                    for t in reversed(toks)
                )
            )
    return "".join(out)


def _ltr_paragraph(tokens, kinds):
                                                                     
                                                                            
    out = []
    i = 0
    while i < len(tokens):
        if kinds[i] != "rtl":
            out.append(tokens[i])
            i += 1
            continue
        j = i
        while j < len(tokens) and kinds[j] != "ltr":
            j += 1
        while kinds[j - 1] not in ("rtl", "num"):
            j -= 1
        out.append(_visual_line("".join(tokens[i:j])))
        i = j
    return "".join(out)


class BidiDraw(ImageDraw.ImageDraw):
    def text(self, xy, text, *args, **kwargs):
        if not _native_bidi():
            text = visual_order(text)
        return super().text(xy, text, *args, **kwargs)


def draw(image):
    return BidiDraw(image)
