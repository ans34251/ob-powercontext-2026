# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prefix-reusing context packs for incremental context transfer."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST_PREFIX = "sha256:"


class ContextPackError(ValueError):
    """A context pack or delta violates the wire protocol."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete, chunked, checksummed context body."""

    items: tuple[bytes, ...]
    digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or not all(isinstance(item, bytes) for item in self.items):
            raise ContextPackError("pack items must be a tuple of bytes")
        expected = _pack_digest(self.items)
        if self.digest is None:
            object.__setattr__(self, "digest", expected)
        elif self.digest != expected:
            raise ContextPackError("pack digest does not match its items")

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes], /) -> ContextPack:
        """Build a complete pack from in-memory chunks."""

        return cls(tuple(items))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split UTF-8 content into deterministic chunks no larger than ``chunk_bytes``."""

        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if not isinstance(content, str):
            raise ContextPackError("content must be text")
        return cls(_chunk_text(content, chunk_bytes))

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and validate a complete pack payload."""

        payload = _load_json(data, expected_fields=_PACK_FIELDS)
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unsupported context pack schema")
        byte_count = payload["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count != len(data):
            raise ContextPackError("pack byte_count does not match its payload size")
        digest = payload["digest"]
        if not isinstance(digest, str):
            raise ContextPackError("pack digest must be a string")
        items = _decode_items(payload["items"])
        pack = cls(items, digest)
        if len(pack.to_bytes()) != len(data):
            raise ContextPackError("pack is not canonically encoded")
        return pack

    @property
    def byte_count(self) -> int:
        """Serialized size of this pack, including every protocol field."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Encode the pack canonically with a self-referential byte count."""

        unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": _encode_items(self.items)}
        digest = _pack_digest(self.items)
        byte_count = 0
        for _ in range(8):
            payload = {**unsigned, "digest": digest, "byte_count": byte_count}
            encoded = _canonical(payload)
            if len(encoded) == byte_count:
                return encoded
            byte_count = len(encoded)
        raise ContextPackError("pack byte_count did not converge")


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A reusable base prefix plus the suffix needed to rebuild the target."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Reuse the longest shared chunk prefix and carry the remaining suffix."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be context packs")
        prefix_items = 0
        for base_item, target_item in zip(base.items, target.items):
            if base_item != target_item:
                break
            prefix_items += 1
        return cls(
            base_digest=_require_digest(base.digest, "base"),
            target_digest=_require_digest(target.digest, "target"),
            prefix_items=prefix_items,
            appended=target.items[prefix_items:],
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and validate a delta payload."""

        payload = _load_json(data, expected_fields=_DELTA_FIELDS)
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unsupported context pack delta schema")
        prefix_items = payload["prefix_items"]
        if isinstance(prefix_items, bool) or not isinstance(prefix_items, int) or prefix_items < 0:
            raise ContextPackError("delta prefix_items must be a non-negative integer")
        return cls(
            base_digest=_require_digest(payload["base_digest"], "base"),
            target_digest=_require_digest(payload["target_digest"], "target"),
            prefix_items=prefix_items,
            appended=_decode_items(payload["appended"]),
        )

    @property
    def byte_count(self) -> int:
        """Serialized size of this delta, including every protocol field."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Encode the delta canonically."""

        if isinstance(self.prefix_items, bool) or not isinstance(self.prefix_items, int) or self.prefix_items < 0:
            raise ContextPackError("delta prefix_items must be a non-negative integer")
        if not isinstance(self.appended, tuple) or not all(isinstance(item, bytes) for item in self.appended):
            raise ContextPackError("delta appended must be a tuple of bytes")
        return _canonical(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": _require_digest(self.base_digest, "base"),
                "target_digest": _require_digest(self.target_digest, "target"),
                "prefix_items": self.prefix_items,
                "appended": _encode_items(self.appended),
            }
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from a base pack and a delta."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta application requires a pack and a delta")
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the provided pack")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base pack item count")
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack digest does not match the delta target")
    return target


def _chunk_text(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    chunks: list[bytes] = []
    pending = bytearray()
    for character in content:
        encoded = character.encode("utf-8")
        if len(pending) + len(encoded) > chunk_bytes:
            if not pending:
                raise ContextPackError("chunk_bytes is too small for one character")
            chunks.append(bytes(pending))
            pending = bytearray()
        pending.extend(encoded)
    if pending:
        chunks.append(bytes(pending))
    return tuple(chunks)


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": _encode_items(items)}
    return _DIGEST_PREFIX + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(value: Any) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("chunk collection must be a list")
    decoded: list[bytes] = []
    for chunk in value:
        if not isinstance(chunk, str):
            raise ContextPackError("chunk must be a base64 string")
        try:
            decoded.append(base64.b64decode(chunk, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError("chunk is not valid base64") from error
    return tuple(decoded)


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX) or len(value) != len(_DIGEST_PREFIX) + 64:
        raise ContextPackError(f"{label} digest must be a sha256 string")
    return value


def _load_json(data: bytes | bytearray, *, expected_fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)):
        raise ContextPackError("wire payload must be bytes")
    try:
        payload = json.loads(bytes(data))
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise ContextPackError("wire payload is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ContextPackError("wire payload has missing or unexpected fields")
    return payload


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
