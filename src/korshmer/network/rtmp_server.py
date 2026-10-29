from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import urlsplit

from korshmer.network.youtube_live import YouTubeLiveStreamer

DEFAULT_PORT = 1935
DEFAULT_HOST = "127.0.0.1"
MAX_URL_LENGTH = 2048

_BAD_CHARS = re.compile(r"[\s\x00-\x1f\x7f]")
_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+$")
_PATH_PART_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def valid_rtmp_url(url: str) -> bool:
    if not isinstance(url, str) or not url or len(url) > MAX_URL_LENGTH:
        return False
    if _BAD_CHARS.search(url):
        return False
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    if parts.scheme not in ("rtmp", "rtmps"):
        return False
    if not parts.hostname or parts.username or parts.password:
        return False
    if parts.query or parts.fragment:
        return False
    if port is not None and not 1 <= port <= 65535:
        return False
    return True


class RtmpDestination(YouTubeLiveStreamer):
    def __init__(
        self,
        url: str,
        stream_key: str = "",
        on_status: Callable[[str], None] | None = None,
        **kwargs,
    ):
        url = (url or "").rstrip("/")
        if not valid_rtmp_url(url):
            raise ValueError("destination must be an rtmp:// or rtmps:// URL")
        super().__init__(
            stream_key,
            on_status=on_status,
            ingest_url=url,
            label="RTMP",
            allow_empty_key=True,
            **kwargs,
        )


class LocalRtmpServer(YouTubeLiveStreamer):
    def __init__(
        self,
        port: int = DEFAULT_PORT,
        host: str = DEFAULT_HOST,
        app: str = "live",
        stream: str = "korshmer",
        on_status: Callable[[str], None] | None = None,
        **kwargs,
    ):
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not (1 <= port <= 65535)
        ):
            raise ValueError("port must be between 1 and 65535")
        if not host or not _HOST_RE.match(host) or host.startswith("-"):
            raise ValueError("invalid host")
        if not _PATH_PART_RE.match(app or "") or not _PATH_PART_RE.match(
            stream or ""
        ):
            raise ValueError("invalid application or stream name")
        self.host = host
        self.port = port
        self.app = app
        self.stream = stream
        super().__init__(
            stream,
            on_status=on_status,
            ingest_url=f"rtmp://{host}:{port}/{app}",
            label="RTMP server",
            listen=True,
            **kwargs,
        )

    @property
    def url(self) -> str:
        return f"rtmp://{self.host}:{self.port}/{self.app}/{self.stream}"
