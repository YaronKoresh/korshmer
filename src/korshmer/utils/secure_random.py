from __future__ import annotations

import secrets
import tempfile
import uuid
from pathlib import Path

DEFAULT_TOKEN_BYTES = 16


def token_bytes(nbytes: int = DEFAULT_TOKEN_BYTES) -> bytes:
    if nbytes < 1:
        raise ValueError("nbytes must be at least 1")
    return secrets.token_bytes(nbytes)


def token_hex(nbytes: int = DEFAULT_TOKEN_BYTES) -> str:
    if nbytes < 1:
        raise ValueError("nbytes must be at least 1")
    return secrets.token_hex(nbytes)


def token_urlsafe(nbytes: int = DEFAULT_TOKEN_BYTES) -> str:
    if nbytes < 1:
        raise ValueError("nbytes must be at least 1")
    return secrets.token_urlsafe(nbytes)


def new_id() -> str:
    return str(uuid.UUID(bytes=secrets.token_bytes(16), version=4))


def make_private_dir(parent, prefix: str) -> Path:
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
