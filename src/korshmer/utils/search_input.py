from __future__ import annotations

from korshmer.utils.text_input import MAX_TEXT_INPUT_LENGTH

MAX_SEARCH_QUERY_LENGTH: int = MAX_TEXT_INPUT_LENGTH


def sanitize_search_query(query: str) -> str:
    if not isinstance(query, str):
        return ""

    cleaned = "".join(ch for ch in query if ord(ch) >= 32 and ch != "\x7f")
    cleaned = cleaned.strip()
    return cleaned[:MAX_SEARCH_QUERY_LENGTH]
