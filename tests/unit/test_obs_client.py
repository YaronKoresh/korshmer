import asyncio
import json
import socket
import threading

import pytest

from korshmer.network.obs_client import (
    SUBPROTOCOL,
    ObsClient,
    auth_string,
)

websockets = pytest.importorskip("websockets")


class FakeObs:
    def __init__(self, password=None):
        self.password = password
        self.received = []
        self.messages = []
        self.subprotocol = None
        self.identify = None
        self.fail_requests = False
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.port = None

    def __enter__(self):
        self._thread.start()
        assert self._ready.wait(5)
        return self

    def __exit__(self, *exc):
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(5)

    def _run(self):
        asyncio.run(self._main())

    async def _handler(self, ws):
        self.subprotocol = ws.subprotocol
        hello = {"obsWebSocketVersion": "5.0.0", "rpcVersion": 1}
        if self.password:
            hello["authentication"] = {"challenge": "chal", "salt": "salt"}
        await ws.send(json.dumps({"op": 0, "d": hello}))
        self.identify = json.loads(await ws.recv())
        if self.password:
            expected = auth_string(self.password, "salt", "chal")
            if self.identify["d"].get("authentication") != expected:
                await ws.close(4009, "auth failed")
                return
        await ws.send(json.dumps({"op": 2, "d": {"negotiatedRpcVersion": 1}}))
        async for raw in ws:
            msg = json.loads(raw)
            self.messages.append(msg["d"])
            self.received.append(msg["d"]["requestType"])
            await ws.send(
                json.dumps(
                    {
                        "op": 7,
                        "d": {
                            "requestType": msg["d"]["requestType"],
                            "requestId": msg["d"]["requestId"],
                            "requestStatus": {
                                "result": not self.fail_requests,
                                "code": 100,
                                "comment": "nope",
                            },
                        },
                    }
                )
            )

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with websockets.serve(
            self._handler,
            "127.0.0.1",
            0,
            subprotocols=[SUBPROTOCOL],
        ) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}"


def wait_for(predicate, timeout=3.0):
    done = threading.Event()
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return True
        done.wait(0.02)
    return predicate()


def test_identify_sent_and_record_requested_once():
    with FakeObs() as server:
        client = ObsClient(server.url, timeout=3)
        try:
            assert client.start() is True
            assert server.identify["op"] == 1
            assert server.identify["d"]["rpcVersion"] == 1
            assert "authentication" not in server.identify["d"]
            assert server.subprotocol == SUBPROTOCOL

            assert client.start_recording() is True
            assert client.start_recording() is False
            assert wait_for(lambda: server.received == ["StartRecord"])

            assert client.stop_recording() is True
            assert client.stop_recording() is False
            assert wait_for(
                lambda: server.received == ["StartRecord", "StopRecord"]
            )
        finally:
            client.stop()


def test_password_authentication():
    with FakeObs(password="secret") as server:
        client = ObsClient(server.url, password="secret", timeout=3)
        try:
            assert client.start() is True
        finally:
            client.stop()


def test_wrong_password_returns_false():
    with FakeObs(password="secret") as server:
        messages = []
        client = ObsClient(
            server.url, password="wrong", on_status=messages.append, timeout=3
        )
        assert client.start() is False
        assert client.start_recording() is False
        assert not any("wrong" in m for m in messages)
        client.stop()


def test_missing_password_returns_false():
    with FakeObs(password="secret") as server:
        client = ObsClient(server.url, timeout=3)
        assert client.start() is False
        client.stop()


def test_unavailable_server_returns_false_without_raising():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    messages = []
    client = ObsClient(
        f"ws://127.0.0.1:{port}", on_status=messages.append, timeout=2
    )
    assert client.start() is False
    assert client.connected is False
    assert client.start_recording() is False
    assert any("could not be reached" in m for m in messages)
    client.stop()


def test_failed_start_request_allows_retry():
    with FakeObs() as server:
        server.fail_requests = True
        messages = []
        client = ObsClient(server.url, on_status=messages.append, timeout=3)
        try:
            assert client.start() is True
            assert client.start_recording() is True
            assert wait_for(lambda: client.recording is False)
            assert any("StartRecord failed" in m for m in messages)
        finally:
            client.stop()


def test_stop_without_start_is_safe():
    ObsClient().stop()


def test_default_url_is_loopback():
    assert ObsClient().url.startswith("ws://127.0.0.1:")


def test_auth_string_follows_the_protocol_steps():
    import base64
    import hashlib

    secret = base64.b64encode(hashlib.sha256(b"psalt").digest()).decode()
    expected = base64.b64encode(
        hashlib.sha256((secret + "chal").encode()).digest()
    ).decode()
    assert auth_string("p", "salt", "chal") == expected


def test_switch_scene_sends_request_data():
    with FakeObs() as server:
        client = ObsClient(server.url, timeout=3)
        try:
            assert client.switch_scene("B") is False
            assert client.start() is True
            assert client.switch_scene("B") is True
            assert wait_for(
                lambda: server.received == ["SetCurrentProgramScene"]
            )
            assert server.messages[0]["requestData"] == {"sceneName": "B"}
        finally:
            client.stop()


def test_switch_scene_rejects_bad_names():
    with FakeObs() as server:
        client = ObsClient(server.url, timeout=3)
        try:
            assert client.start() is True
            for bad in ("", "  ", None, 5, "a\x00b", "x" * 300):
                assert client.switch_scene(bad) is False
            assert server.received == []
        finally:
            client.stop()
