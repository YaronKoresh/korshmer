from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

from korshmer.utils.secure_random import new_id


class OpType(Enum):
    NOTE_ADD = "note_add"
    NOTE_REMOVE = "note_remove"
    NOTE_MODIFY = "note_modify"
    TEMPO_CHANGE = "tempo_change"
    SCALE_CHANGE = "scale_change"
    CURSOR_MOVE = "cursor_move"
    TRACK_ADD = "track_add"
    TRACK_REMOVE = "track_remove"
    TRACK_MUTE = "track_mute"
    SYNC_STATE = "sync_state"
    PEER_JOINED = "peer_joined"
    PEER_LEFT = "peer_left"
    ERROR = "error"


@dataclass
class Operation:
    op_type: str
    session_id: str
    peer_id: str
    timestamp: float = field(default_factory=time.time)
    op_id: str = field(default_factory=lambda: new_id())
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> Operation:
        payload = json.loads(s)
        if not isinstance(payload, dict):
            raise ValueError("Operation message must be a JSON object")

        required = ("op_type", "session_id", "peer_id")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                "Operation message is missing required fields: "
                + ", ".join(missing)
            )
        for name in required:
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError(
                    f"Operation field '{name}' must be a non-empty string"
                )

        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in payload.items() if k in known}

        if kwargs.get("data") is None:
            kwargs["data"] = {}
        elif not isinstance(kwargs["data"], dict):
            raise ValueError("Operation field 'data' must be an object")
        for name in ("timestamp", "op_id"):
            if name in kwargs and kwargs[name] is None:
                del kwargs[name]
        if "timestamp" in kwargs and (
            isinstance(kwargs["timestamp"], bool)
            or not isinstance(kwargs["timestamp"], (int, float))
        ):
            raise ValueError("Operation field 'timestamp' must be a number")
        if "op_id" in kwargs and not isinstance(kwargs["op_id"], str):
            raise ValueError("Operation field 'op_id' must be a string")

        return cls(**kwargs)


def transform(op1: Operation, op2: Operation) -> Operation:
    if (
        op1.op_type == OpType.NOTE_ADD.value
        and op2.op_type == OpType.NOTE_REMOVE.value
    ):
        if op1.data.get("note_id") == op2.data.get("note_id"):
            op1.data["_conflict"] = True
    return op1
