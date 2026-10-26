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

"""Capture one selected Pi session branch as immutable raw event Sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
    Source,
    SourceMaterialization,
    TextEvidence,
)

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
_PI_SESSION_VERSION = 3
_CHECKPOINT_VERSION = 1
_CHECKPOINT_EVENTS_KEY = "events"
_CHECKPOINT_VERSION_KEY = "version"
_COMPACT_JSON_SEPARATORS = (",", ":")


class PiSessionInput(BaseModel):
    """One complete UTF-8 JSONL Pi session plus the selected branch leaf."""

    model_config = ConfigDict(extra="forbid")

    content: bytes
    leaf_id: str | None = None

    @field_validator("leaf_id")
    @classmethod
    def validate_leaf_id(cls, value: str | None) -> str | None:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("leaf_id must be a non-empty string")  # noqa: TRY003
        return value


class PiEventInput(BaseModel):
    """One raw visible Pi event that still carries its complete input record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]

    @field_validator("session_id", "event_id", "event_type", "role")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Pi event identity fields must be non-empty")  # noqa: TRY003
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Pi event content must be non-empty")  # noqa: TRY003
        return value


class PiEventSource(Source):
    """One captured visible Pi event addressable by session and event identity."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable identity of one Pi event from its session and event IDs."""

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")  # noqa: TRY003
    material = json.dumps([session_id, event_id], ensure_ascii=False, separators=_COMPACT_JSON_SEPARATORS)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{PI_EVENT_SOURCE_NAME}:{digest}"


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse the full ancestor chain of the selected leaf into visible events."""

    records = _read_records(value.content)
    session_id = _session_id(records[0])
    nodes = _session_nodes(records[1:])
    if value.leaf_id is None:
        if not nodes:
            return ()
        leaf_id = next(reversed(nodes))
    else:
        if value.leaf_id not in nodes:
            raise ValueError(f"Pi session leaf {value.leaf_id!r} is missing")  # noqa: TRY003
        leaf_id = value.leaf_id
    events: list[PiEventInput] = []
    for event_id in _ancestor_chain(nodes, leaf_id):
        event = _visible_event(session_id, nodes[event_id])
        if event is not None:
            events.append(event)
    return tuple(events)


def _read_records(content: bytes) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session is not valid UTF-8") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session line {line_number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"Pi session line {line_number} is not a JSON object")  # noqa: TRY003
        records.append(record)
    if not records:
        raise ValueError("Pi session is empty")  # noqa: TRY003
    return records


def _session_id(header: dict[str, Any]) -> str:
    if header.get("type") != "session":
        raise ValueError("Pi session must start with a session header")  # noqa: TRY003
    version = header.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version != _PI_SESSION_VERSION:
        raise ValueError(f"Pi session version must be {_PI_SESSION_VERSION}")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Pi session header id must be a non-empty string")  # noqa: TRY003
    return session_id


def _session_nodes(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for record in records:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("Pi session record id must be a non-empty string")  # noqa: TRY003
        if event_id in nodes:
            raise ValueError(f"Pi session record id {event_id!r} is duplicated")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError(f"Pi session record {event_id!r} is missing parentId")  # noqa: TRY003
        parent_id = record["parentId"]
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError(f"Pi session record {event_id!r} has an invalid parentId")  # noqa: TRY003
            if parent_id not in nodes:
                raise ValueError(f"Pi session record {event_id!r} references a missing parent")  # noqa: TRY003
        nodes[event_id] = record
    return nodes


def _ancestor_chain(nodes: dict[str, dict[str, Any]], leaf_id: str) -> tuple[str, ...]:
    chain: list[str] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            raise ValueError("Pi session contains a parent cycle")  # noqa: TRY003
        seen.add(current)
        chain.append(current)
        parent = nodes[current]["parentId"]
        current = parent if parent else None
    chain.reverse()
    return tuple(chain)


def _visible_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    record_type = record.get("type")
    if not isinstance(record_type, str) or not record_type:
        return None
    if record_type == "message":
        rendered = _message_event(record.get("message"))
    elif record_type in {"compaction", "branch_summary"}:
        body = _text(record.get("summary"))
        rendered = (record_type, body) if body.strip() else None
    elif record_type == "custom_message":
        body = _text_only(record.get("content"))
        rendered = ("custom", body) if body.strip() else None
    else:
        return None
    if rendered is None:
        return None
    role, body = rendered
    if not body.strip():
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=record["parentId"],
        event_type=record_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_event(message: object) -> tuple[str, str] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    if role == "toolResult":
        tool_name = _identifier(message.get("toolName"))
        tool_call_id = _identifier(message.get("toolCallId"))
        return role, f"[tool_result {tool_name} {tool_call_id}] {_text_only(message.get('content'))}"
    if role == "bashExecution":
        return role, f"$ {_text(message.get('command'))}\n{_text(message.get('output'))}"
    return role, _message_text(message.get("content"))


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall":
            parts.append(_tool_call(block))
    return "\n".join(parts)


def _text_only(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _tool_call(block: dict[str, Any]) -> str:
    name = _identifier(block.get("name"))
    call_id = _identifier(block.get("id"))
    arguments = block.get("arguments", {})
    return f"[tool_call {name} {call_id}] {json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=_COMPACT_JSON_SEPARATORS)}"


def _identifier(value: object) -> str:
    return value if isinstance(value, str) and value else "unknown"


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


class PiEventSourceAdapter:
    """Resolve and read one Pi event Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=pi_event_source_id(value.session_id, value.event_id),
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one raw Pi event as its complete lossless text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class PiSessionConnector:
    """Import every not-yet-captured visible event of one selected branch."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        if leaf_id is not None and (not isinstance(leaf_id, str) or not leaf_id.strip()):
            raise ValueError("leaf_id must be a non-empty string")  # noqa: TRY003
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))
        imported = _checkpoint_events(session.checkpoint)
        digests = {
            pi_event_source_id(event.session_id, event.event_id): _record_digest(event.record) for event in events
        }
        for identity, digest in digests.items():
            known = imported.get(identity)
            if known is not None and known != digest:
                raise ValueError(f"Pi event {identity!r} changed after it was imported")  # noqa: TRY003
        next_events = dict(imported)
        for event in events:
            identity = pi_event_source_id(event.session_id, event.event_id)
            if identity in imported:
                continue
            outcome = await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                return ConnectorRunCompletion(status=ConnectorRunStatus.INCOMPLETE)
            next_events[identity] = digests[identity]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={_CHECKPOINT_VERSION_KEY: _CHECKPOINT_VERSION, _CHECKPOINT_EVENTS_KEY: next_events},
        )


def _checkpoint_events(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or value.get(_CHECKPOINT_VERSION_KEY) != _CHECKPOINT_VERSION:
        raise ValueError("Pi session checkpoint is not supported")  # noqa: TRY003
    events = value.get(_CHECKPOINT_EVENTS_KEY)
    if not isinstance(events, dict):
        raise ValueError("Pi session checkpoint events are missing")  # noqa: TRY003
    validated: dict[str, str] = {}
    for identity, digest in events.items():
        if not isinstance(identity, str) or not identity:
            raise ValueError("Pi session checkpoint identity is invalid")  # noqa: TRY003
        if not isinstance(digest, str) or not _is_digest(digest):
            raise ValueError("Pi session checkpoint digest is invalid")  # noqa: TRY003
        validated[identity] = digest
    return validated


def _is_digest(value: str) -> bool:
    if not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError:
        return False
    return value == value.lower()


def _record_digest(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=_COMPACT_JSON_SEPARATORS).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    projections=(PiEventTextEvidenceProjection(),),
)

__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
