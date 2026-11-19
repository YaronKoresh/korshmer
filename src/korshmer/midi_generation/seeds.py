import hashlib
import secrets
import unicodedata

RANDOM_WORDS = ("", "random")
FRESH_SEED_DIGITS = 18
TEXT_SEED_BYTES = 8
DIGIT_SEPARATORS = " ,._'"
MAX_SEED_TEXT = 200


def fresh_seed():
    low = 10 ** (FRESH_SEED_DIGITS - 1)
    return low + secrets.randbelow(9 * low)


def _clean(text):
    text = unicodedata.normalize("NFKC", str(text))
    return " ".join(text.split()).casefold()


def _digits(text):
    digits = text
    for separator in DIGIT_SEPARATORS:
        digits = digits.replace(separator, "")
    return digits if digits.isascii() and digits.isdigit() else None


def seed_number(seed):
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise ValueError("The seed must be a whole number or a text.")
    if isinstance(seed, int):
        return abs(seed)
    if len(seed) > MAX_SEED_TEXT:
        raise ValueError(
            f"The seed can be at most {MAX_SEED_TEXT} characters long."
        )
    text = _clean(seed)
    if text in RANDOM_WORDS:
        return None
    digits = _digits(text)
    if digits is not None:
        return int(digits) or None
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:TEXT_SEED_BYTES], "big") or 1


def seed_label(seed):
    number = seed_number(seed)
    if number is None:
        return "random"
    if isinstance(seed, str) and _digits(_clean(seed)) is None:
        shown = " ".join(seed.split())
        return f'"{shown}" = {number}'
    return str(number)
