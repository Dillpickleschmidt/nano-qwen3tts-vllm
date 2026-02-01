"""
ZMQ PUB bridge for publishing token outputs by (engine_type, request_id).

Message format (multipart):
  - Frame 0: topic (bytes) = "talker/<request_id>" or "predictor/<request_id>"
  - Frame 1: msg_type (bytes) = "token" or "done"
  - Frame 2: payload (bytes, serialized)

Payload for "token": msgpack-encoded dict with "token_ids" (list[int]) and
optionally "hidden_states" (numpy array serialized as bytes + shape + dtype).
Payload for "done": empty or msgpack dict with optional summary.
"""

import os
from typing import Any, Optional

import numpy as np

try:
    import zmq
except ImportError:
    zmq = None

try:
    import msgpack
except ImportError:
    msgpack = None


def _ensure_deps():
    if zmq is None:
        raise ImportError("pyzmq is required for ZMQ output bridge. Install with: pip install pyzmq")
    if msgpack is None:
        raise ImportError("msgpack is required for ZMQ output bridge. Install with: pip install msgpack")


def serialize_token_payload(token_ids: list[int], hidden_states: Optional[Any] = None) -> bytes:
    """Serialize token output for ZMQ. hidden_states: numpy array or torch tensor (will be converted to numpy)."""
    _ensure_deps()
    obj = {"token_ids": token_ids}
    if hidden_states is not None:
        if hasattr(hidden_states, "cpu"):
            arr = hidden_states.float().detach().cpu().numpy()
        else:
            arr = np.asarray(hidden_states)
        obj["hidden_states"] = arr.tobytes()
        obj["hidden_states_shape"] = list(arr.shape)
        obj["hidden_states_dtype"] = str(arr.dtype)
    return msgpack.packb(obj, use_bin_type=True)


def deserialize_token_payload(payload: bytes) -> dict:
    """Deserialize token payload from ZMQ. Returns dict with 'token_ids' and optionally 'hidden_states' (numpy)."""
    _ensure_deps()
    obj = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    if "hidden_states_shape" in obj:
        arr = np.frombuffer(obj["hidden_states"], dtype=obj["hidden_states_dtype"]).reshape(obj["hidden_states_shape"])
        obj["hidden_states"] = arr
        del obj["hidden_states_shape"]
        del obj["hidden_states_dtype"]
    return obj


def topic_for(engine_type: str, request_id: str) -> str:
    """Build ZMQ topic: e.g. talker/<request_id> or predictor/<request_id>."""
    return f"{engine_type}/{request_id}"


class ZMQOutputBridge:
    """
    Publishes engine outputs over a ZMQ PUB socket.
    Topic = engine_type/request_id so subscribers can filter by request_id.
    """

    def __init__(self, bind_address: Optional[str] = None):
        _ensure_deps()
        self.bind_address = bind_address or os.environ.get("QWEN_TTS_ZMQ_PUB", "tcp://127.0.0.1:9555")
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self.bind_address)

    def publish_token(self, engine_type: str, request_id: str, token_ids: list[int], hidden_states: Optional[Any] = None):
        """Publish a token output. engine_type is 'talker' or 'predictor'."""
        topic = topic_for(engine_type, request_id)
        payload = serialize_token_payload(token_ids, hidden_states)
        self._socket.send_multipart([topic.encode("utf-8"), b"token", payload], flags=zmq.NOBLOCK)

    def publish_done(self, engine_type: str, request_id: str, payload: Optional[bytes] = None):
        """Publish a done message for this request."""
        topic = topic_for(engine_type, request_id)
        self._socket.send_multipart([topic.encode("utf-8"), b"done", payload or b""], flags=zmq.NOBLOCK)

    def close(self):
        self._socket.close()
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
