from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from korshmer.utils.audit_log import OUTCOME_FAILED, audit
from korshmer.utils.log_stream import LogStreamHandler, parse_stream_level
from korshmer.utils.perf_metrics import metrics
from korshmer.utils.tracing import tracer

_logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_STALE_SECONDS = 5.0
PORT_ENV_VAR = "KORSHMER_HEALTH_PORT"

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_ERROR = "error"

LIVE_PATH = "/health"
STATE_PATH = "/health/state"
METRICS_PATH = "/health/metrics"
TRACES_PATH = "/health/traces"
LOGS_PATH = "/health/logs"
LOG_STREAM_PATH = "/health/logs/stream"

MAX_LOG_LIMIT = 1000
STREAM_KEEPALIVE_SECONDS = 15.0
STREAM_POLL_SECONDS = 1.0

CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)
SECURITY_HEADERS = (
    ("Content-Security-Policy", CONTENT_SECURITY_POLICY),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)

Check = Callable[[], Any]


def _reuse_address_allowed() -> bool:
    return sys.platform != "win32"


class _HealthHTTPServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        self.allow_reuse_address = _reuse_address_allowed()
        super().server_bind()


def port_from_env(environ: Mapping[str, str]) -> int | None:
    raw = (environ.get(PORT_ENV_VAR) or "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        _logger.warning("Ignoring %s=%r: not a port number", PORT_ENV_VAR, raw)
        return None
    if not 0 <= port <= 65535:
        _logger.warning("Ignoring %s=%r: out of range", PORT_ENV_VAR, raw)
        return None
    return port


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _check_result(result: Any) -> dict[str, Any]:
    if isinstance(result, bool):
        return {"status": STATUS_OK if result else STATUS_ERROR}
    if isinstance(result, dict):
        data = _jsonable(result)
        if data.get("status") not in (STATUS_OK, STATUS_DEGRADED, STATUS_ERROR):
            data["status"] = STATUS_OK
        return data
    return {"status": STATUS_OK, "value": _jsonable(result)}


class HealthServer:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = 0,
        stale_after: float = DEFAULT_STALE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        log_stream: LogStreamHandler | None = None,
    ):
        if (
            isinstance(stale_after, bool)
            or not isinstance(stale_after, (int, float))
            or not math.isfinite(stale_after)
            or stale_after <= 0
        ):
            raise ValueError(
                f"stale_after must be a finite positive number, got "
                f"{stale_after!r}"
            )
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 <= port <= 65535
        ):
            raise ValueError(
                f"port must be a whole number from 0 to 65535, got {port!r}"
            )
        self.host = host
        self.port = port
        self.stale_after = stale_after
        self._clock = clock
        self.log_stream = log_stream
        self._stopping = threading.Event()
        self._started_at = clock()
        self._lock = threading.Lock()
        self._checks: dict[str, Check] = {}
        self._state: dict[str, Any] = {}
        self._last_beat: float | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def register_check(self, name: str, check: Check) -> None:
        with self._lock:
            self._checks[name] = check

    def unregister_check(self, name: str) -> None:
        with self._lock:
            self._checks.pop(name, None)

    def beat(self, **state: Any) -> None:
        with self._lock:
            self._last_beat = self._clock()
            self._state.update(state)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            checks = dict(self._checks)
            state = _jsonable(self._state)
            last_beat = self._last_beat
        now = self._clock()
        components: dict[str, Any] = {}
        if last_beat is None:
            components["main_thread"] = {
                "status": STATUS_DEGRADED,
                "detail": "no heartbeat yet",
            }
        else:
            age = now - last_beat
            components["main_thread"] = {
                "status": STATUS_OK
                if age <= self.stale_after
                else STATUS_ERROR,
                "heartbeat_age_seconds": round(age, 3),
            }
        for name, check in checks.items():
            try:
                components[name] = _check_result(check())
            except Exception as e:
                _logger.debug("health check %s failed", name, exc_info=True)
                components[name] = {
                    "status": STATUS_ERROR,
                    "detail": f"{type(e).__name__}: {e}",
                }
        statuses = {c["status"] for c in components.values()}
        if STATUS_ERROR in statuses:
            overall = STATUS_ERROR
        elif STATUS_DEGRADED in statuses:
            overall = STATUS_DEGRADED
        else:
            overall = STATUS_OK
        return {
            "status": overall,
            "uptime_seconds": round(now - self._started_at, 3),
            "components": components,
            "state": state,
        }

    def _log_query(
        self, query: dict[str, list[str]], last_event_id: str | None = None
    ) -> tuple[dict[str, Any] | None, str | None]:
        def first(name: str) -> str | None:
            values = query.get(name)
            return values[0] if values else None

        raw_since = first("since") or last_event_id or "0"
        try:
            since = int(raw_since)
            limit = int(first("limit") or MAX_LOG_LIMIT)
        except ValueError:
            return None, "since and limit must be integers"
        if since < 0 or limit < 1:
            return None, "since must not be negative and limit must be positive"
        level, invalid = parse_stream_level(first("level"))
        if invalid:
            return None, "unknown level"
        return {
            "since": since,
            "limit": min(limit, MAX_LOG_LIMIT),
            "level": level,
            "component": (first("component") or "").strip() or None,
        }, None

    def _logs_response(self, query: dict[str, list[str]]):
        if self.log_stream is None:
            return 404, {
                "status": STATUS_ERROR,
                "detail": "log streaming is off",
            }
        params, error = self._log_query(query)
        if params is None:
            return 400, {"status": STATUS_ERROR, "detail": error}
        entries, last_seq, dropped = self.log_stream.read(**params)
        return 200, {
            "status": STATUS_OK,
            "last_seq": last_seq,
            "dropped": dropped,
            "logs": entries,
        }

    def handle(self, path: str) -> tuple[int, dict[str, Any]]:
        parts = urlsplit(path)
        query = parse_qs(parts.query)
        path = parts.path.rstrip("/") or "/"
        if path == LOGS_PATH:
            return self._logs_response(query)
        if path == LIVE_PATH:
            return 200, {
                "status": STATUS_OK,
                "uptime_seconds": round(self._clock() - self._started_at, 3),
            }
        if path == STATE_PATH:
            snap = self.snapshot()
            return (503 if snap["status"] == STATUS_ERROR else 200), snap
        if path == METRICS_PATH:
            return 200, {"status": STATUS_OK, "metrics": metrics.snapshot()}
        if path == TRACES_PATH:
            return 200, {"status": STATUS_OK, "traces": tracer.snapshot()}
        return 404, {"status": STATUS_ERROR, "detail": "not found"}

    def stream_logs(self, handler: BaseHTTPRequestHandler) -> None:
        parts = urlsplit(handler.path)
        if self.log_stream is None:
            handler.send_error(404, "log streaming is off")
            return
        params, error = self._log_query(
            parse_qs(parts.query), handler.headers.get("Last-Event-ID")
        )
        if params is None:
            handler.send_error(400, error)
            return
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        since = params.pop("since")
        stream = self.log_stream
        idle = 0.0
        try:
            while not self._stopping.is_set() and not stream.closed:
                entries, last_seq, _ = stream.read(since=since, **params)
                if entries:
                    chunks = [
                        f"id: {e['seq']}\ndata: {json.dumps(e)}\n\n"
                        for e in entries
                    ]
                    handler.wfile.write("".join(chunks).encode("utf-8"))
                    handler.wfile.flush()
                    idle = 0.0
                since = last_seq
                if not entries:
                    stream.wait(since, STREAM_POLL_SECONDS)
                    idle += STREAM_POLL_SECONDS
                    if idle >= STREAM_KEEPALIVE_SECONDS:
                        handler.wfile.write(b": keepalive\n\n")
                        handler.wfile.flush()
                        idle = 0.0
        except OSError:
            return

    def start(self) -> bool:
        if self.running:
            return True
        health = self

        class Handler(BaseHTTPRequestHandler):
            def end_headers(self) -> None:
                for name, value in SECURITY_HEADERS:
                    self.send_header(name, value)
                super().end_headers()

            def do_GET(self) -> None:
                if urlsplit(self.path).path.rstrip("/") == LOG_STREAM_PATH:
                    health.stream_logs(self)
                    return
                code, body = health.handle(self.path)
                self._send(code, body)

            def do_HEAD(self) -> None:
                if urlsplit(self.path).path.rstrip("/") == LOG_STREAM_PATH:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
                code, body = health.handle(self.path)
                self._send(code, body, include_body=False)

            def _send(self, code, body, include_body=True) -> None:
                payload = json.dumps(_jsonable(body), allow_nan=False).encode(
                    "utf-8"
                )
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if include_body:
                    self.wfile.write(payload)

            def log_message(self, format, *args) -> None:
                _logger.debug(
                    "health %s - %s", self.address_string(), format % args
                )

        try:
            server = _HealthHTTPServer((self.host, self.port), Handler)
        except OSError as e:
            _logger.error(
                "Health server cannot listen on %s:%s: %s",
                self.host,
                self.port,
                e,
            )
            audit(
                "health_server.start",
                OUTCOME_FAILED,
                host=self.host,
                port=self.port,
                error=type(e).__name__,
            )
            return False
        server.daemon_threads = True
        self._stopping.clear()
        self.port = server.server_address[1]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="korshmer-health",
            daemon=True,
        )
        self._thread.start()
        _logger.info("Health server listening on %s:%s", self.host, self.port)
        audit("health_server.start", host=self.host, port=self.port)
        return True

    def stop(self) -> None:
        self._stopping.set()
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(2)
        if server is not None:
            audit("health_server.stop", host=self.host, port=self.port)
