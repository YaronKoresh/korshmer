import asyncio
import json

import pytest

from korshmer.collaboration.operations import Operation, OpType
from korshmer.collaboration.server import CollaborationServer


class TestFromJson:
    def test_unknown_field_is_ignored(self):
        op = Operation.from_json(
            '{"op_type":"x","session_id":"s","peer_id":"p","extra":1}'
        )
        assert op.op_type == "x"
        assert not hasattr(op, "extra")

    def test_missing_required_fields_raise_value_error(self):
        with pytest.raises(ValueError):
            Operation.from_json('{"op_type":"x"}')

    def test_non_object_raises_value_error(self):
        with pytest.raises(ValueError):
            Operation.from_json("[1, 2]")

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError):
            Operation.from_json("not json")

    def test_null_data_becomes_empty_dict(self):
        op = Operation.from_json(
            '{"op_type":"x","session_id":"s","peer_id":"p","data":null}'
        )
        assert op.data == {}

    def test_non_dict_data_raises(self):
        with pytest.raises(ValueError):
            Operation.from_json(
                '{"op_type":"x","session_id":"s","peer_id":"p","data":[1]}'
            )

    def test_roundtrip(self):
        op = Operation(op_type="x", session_id="s", peer_id="p", data={"a": 1})
        assert Operation.from_json(op.to_json()) == op


class _WS:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self._messages:
            yield m


class TestServerRepliesWithError:
    def test_invalid_message_returns_error_to_sender(self):
        server = CollaborationServer()
        ws = _WS(['{"op_type":"x"}'])
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server._handle_client(ws, "/"))
        finally:
            loop.close()
        assert len(ws.sent) == 1
        reply = Operation.from_json(ws.sent[0])
        assert reply.op_type == OpType.ERROR.value
        assert "session_id" in json.loads(ws.sent[0])["data"]["error"]
