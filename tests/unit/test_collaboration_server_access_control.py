import asyncio
from types import SimpleNamespace

import pytest

from korshmer.collaboration.client import CollaborationClient
from korshmer.collaboration.operations import Operation, OpType
from korshmer.collaboration.server import CollaborationServer

REFUSED = 1008


class Visitor:
    def __init__(self, messages, path="/", headers=None):
        self.messages = messages
        self.request = SimpleNamespace(path=path, headers=headers or {})
        self.sent = []
        self.closed_with = None

    def __aiter__(self):
        async def gen():
            for m in self.messages:
                yield m

        return gen()

    async def send(self, message):
        self.sent.append(message)

    async def close(self, code=1000, reason=""):
        self.closed_with = code


def cursor_move(session_id="s1"):
    return Operation(
        op_type=OpType.CURSOR_MOVE.value, session_id=session_id, peer_id="x"
    ).to_json()


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def network_server(**options):
    try:
        server = CollaborationServer(host="0.0.0.0", **options)
    except TypeError:
        server = CollaborationServer(host="0.0.0.0")
    server.create_session("s1")
    return server


def visit(server, **visitor_options):
    visitor = Visitor([cursor_move()], **visitor_options)
    run(server._handle_client(visitor))
    return visitor, len(server.sessions["s1"]["operations"])


def test_the_server_listens_only_on_this_computer_by_default():
    assert CollaborationServer().host == "127.0.0.1"


def test_the_default_port_differs_from_the_readme_health_example_as_a_safeguard():
    assert CollaborationServer().port != 8787


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::", ""])
def test_opening_the_server_to_the_network_needs_a_secret(host):
    with pytest.raises(ValueError, match="token"):
        CollaborationServer(host=host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_a_local_server_needs_no_secret_as_a_safeguard(host):
    assert CollaborationServer(host=host).host == host


@pytest.mark.parametrize("path", ["/", "/?token=", "/?token=wrong", "/?t=1"])
def test_a_visitor_without_the_right_secret_is_turned_away(path):
    visitor, stored = visit(network_server(token="open sesame"), path=path)
    assert visitor.closed_with == REFUSED
    assert stored == 0


def test_a_visitor_with_the_secret_in_the_address_joins_as_a_safeguard():
    server = network_server(token="open sesame")
    visitor, stored = visit(server, path="/?token=open%20sesame")
    assert (visitor.closed_with, stored) == (None, 1)


def test_a_visitor_with_the_secret_in_a_header_joins_as_a_safeguard():
    server = network_server(token="open sesame")
    visitor, stored = visit(
        server, headers={"Authorization": "Bearer open sesame"}
    )
    assert (visitor.closed_with, stored) == (None, 1)


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example.com",
        "http://127.0.0.1.evil.example.com",
        "null",
        "file://",
    ],
)
def test_a_web_page_from_another_site_cannot_connect(origin):
    server = CollaborationServer()
    server.create_session("s1")
    visitor, stored = visit(server, headers={"Origin": origin})
    assert (visitor.closed_with, stored) == (REFUSED, 0)


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:3000", "http://127.0.0.1", "https://[::1]:8443"],
)
def test_a_page_served_from_this_computer_may_connect_as_a_safeguard(origin):
    server = CollaborationServer()
    server.create_session("s1")
    visitor, stored = visit(server, headers={"Origin": origin})
    assert (visitor.closed_with, stored) == (None, 1)


def test_the_session_link_never_contains_the_secret():
    server = network_server(token="open sesame")
    assert "sesame" not in server.get_session_url("s1")


def test_the_client_presents_its_secret_to_a_real_server():
    pytest.importorskip("websockets")

    async def scenario():
        server = CollaborationServer(port=0, token="s3cret")
        await server.start()
        port = server.server.sockets[0].getsockname()[1]
        server.create_session("s1")
        joined = asyncio.Event()
        original = server.broadcast

        async def broadcast(session_id, operation, exclude_peer=None):
            await original(session_id, operation, exclude_peer)
            joined.set()

        server.broadcast = broadcast
        client = CollaborationClient(
            f"ws://127.0.0.1:{port}", "s1", token="s3cret"
        )
        try:
            await client.connect()
            await client.send_operation(
                Operation(
                    op_type=OpType.CURSOR_MOVE.value,
                    session_id="s1",
                    peer_id=client.peer_id,
                )
            )
            await asyncio.wait_for(joined.wait(), 3)
            return len(server.peers["s1"])
        finally:
            await client.disconnect()
            await server.stop()

    assert run(scenario()) == 1


def test_a_client_without_the_secret_is_disconnected_by_a_real_server():
    websockets = pytest.importorskip("websockets")

    async def scenario():
        server = CollaborationServer(port=0, token="s3cret")
        await server.start()
        port = server.server.sockets[0].getsockname()[1]
        try:
            bad = await websockets.connect(f"ws://127.0.0.1:{port}")
            with pytest.raises(websockets.exceptions.ConnectionClosed) as end:
                await asyncio.wait_for(bad.recv(), 3)
            return end.value.rcvd.code
        finally:
            await server.stop()

    assert run(scenario()) == REFUSED
