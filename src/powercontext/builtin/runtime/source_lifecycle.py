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

"""Propagate Source lifecycle events through registered evidence dependencies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.errors import ArtifactNotFoundError
from powercontext.builtin.persistence.source_lifecycle import (
    LifecycleNode,
    load_lifecycle_nodes,
    replace_lifecycle_nodes,
)
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.persistence.database import AsyncDatabase
    from powercontext.builtin.runtime.application import BuiltinRuntime

LifecycleFamily = Literal["source", "memory", "topic"]
LifecyclePolicy = Literal["all", "any"]
LifecyclePurpose = Literal["generation", "audit"]

_MAX_NODE_ID_LENGTH = 256


class SourceLifecycleError(ValueError):
    """Reject an invalid lifecycle registration, event, or resolution."""


class SourceLifecycleEvent(BaseModel):
    """One explicit availability change for a registered Source node."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: int
    sequence: int
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")  # noqa: TRY003
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be a positive integer")  # noqa: TRY003
        return value


def _event_digest(event: SourceLifecycleEvent) -> str:
    payload = json.dumps(
        {"available": event.available, "reason": event.reason, "revision": event.revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_node_id(node_id: object) -> str:
    if not isinstance(node_id, str) or not node_id.strip() or len(node_id) > _MAX_NODE_ID_LENGTH:
        raise SourceLifecycleError("node-id")
    return node_id


def _validate_revision(revision: object) -> int:
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise SourceLifecycleError("revision")
    return revision


def _validate_policy(policy: object) -> LifecyclePolicy:
    if policy not in ("all", "any"):
        raise SourceLifecycleError("policy")
    return policy


def _validate_parents(parents: object, node_id: str) -> tuple[str, ...]:
    if isinstance(parents, (str, bytes)) or not isinstance(parents, Iterable):
        raise SourceLifecycleError("parents")
    values: list[str] = []
    for parent in parents:
        if not isinstance(parent, str) or not parent.strip():
            raise SourceLifecycleError("parent")
        if parent == node_id:
            raise SourceLifecycleError("self-dependency")
        if parent in values:
            raise SourceLifecycleError("duplicate-parent")
        values.append(parent)
    if not values:
        raise SourceLifecycleError("parents")
    return tuple(sorted(values))


def _source_payload(reference: SourceRef) -> dict[str, Any]:
    return {"source_type": reference.source_type, "source_id": reference.source_id}


def _memory_payload(citation: MemoryCitation) -> dict[str, Any]:
    return {
        "memory_ref": {
            "family": citation.memory_ref.family,
            "artifact_id": citation.memory_ref.artifact_id,
            "revision": citation.memory_ref.revision,
        },
        "entry_id": citation.entry_id,
        "entry_version_id": citation.entry_version_id,
    }


def _topic_payload(reference: ArtifactRef) -> dict[str, Any]:
    return {
        "family": reference.family,
        "artifact_id": reference.artifact_id,
        "revision": reference.revision,
    }


def _reference(node: LifecycleNode) -> SourceRef | MemoryCitation | ArtifactRef:
    payload = node.reference
    if node.family == "source":
        return SourceRef.model_validate(payload)
    if node.family == "memory":
        return MemoryCitation.model_validate(payload)
    return ArtifactRef.model_validate(payload)


def _reference_key(node: LifecycleNode) -> str:
    return json.dumps(dict(node.reference), sort_keys=True, separators=(",", ":"))


def _compute_available(nodes: Mapping[str, LifecycleNode], node_id: str, memo: dict[str, bool]) -> bool:
    if node_id in memo:
        return memo[node_id]
    node = nodes[node_id]
    if node.family == "source":
        value = node.available
    else:
        support = tuple(_compute_available(nodes, parent, memo) for parent in node.parents)
        value = all(support) if node.policy == "all" else any(support)
    memo[node_id] = value
    return value


def _active_nodes(nodes: Mapping[str, LifecycleNode]) -> dict[str, bool]:
    memo: dict[str, bool] = {}
    return {node_id: _compute_available(nodes, node_id, memo) for node_id in nodes}


def _with_availability(nodes: Mapping[str, LifecycleNode], active: Mapping[str, bool]) -> tuple[LifecycleNode, ...]:
    return tuple(
        LifecycleNode(
            node_id=node.node_id,
            family=node.family,
            reference=node.reference,
            parents=node.parents,
            policy=node.policy,
            revision=node.revision,
            available=active[node.node_id],
            last_sequence=node.last_sequence,
            last_event_digest=node.last_event_digest,
        )
        for node in nodes.values()
    )


class ScopedSourceLifecycle:
    """Register evidence dependencies and apply Source lifecycle events in one Scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id

    @property
    def _database(self) -> AsyncDatabase:
        database = self._runtime._database
        if database is None:
            raise SourceLifecycleError("lifecycle-unavailable")
        return database

    async def register_source(
        self,
        node_id: str,
        source_ref: SourceRef,
        revision: int,
    ) -> None:
        node = _validate_node_id(node_id)
        registered_revision = _validate_revision(revision)
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source-ref")
        await self._require_source(source_ref)
        candidate = LifecycleNode(
            node_id=node,
            family="source",
            reference=_source_payload(source_ref),
            parents=(),
            policy="all",
            revision=registered_revision,
            available=True,
        )
        async with self._runtime._scope_operation(self.scope_id):
            async with self._runtime._locked(self.scope_id):
                nodes = await self._load()
                existing = nodes.get(node)
                if existing is not None:
                    if self._is_same_registration(existing, candidate):
                        return
                    raise SourceLifecycleError("node-conflict")
                nodes[node] = candidate
                await self._save(nodes)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        parents: Iterable[str],
        policy: str = "all",
    ) -> None:
        node = _validate_node_id(node_id)
        parent_ids = _validate_parents(parents, node)
        selected_policy = _validate_policy(policy)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory-citation")
        await self._require_memory(citation)
        candidate = LifecycleNode(
            node_id=node,
            family="memory",
            reference=_memory_payload(citation),
            parents=parent_ids,
            policy=selected_policy,
            revision=0,
            available=True,
        )
        await self._register_derived(candidate)

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        parents: Iterable[str],
        policy: str = "all",
    ) -> None:
        node = _validate_node_id(node_id)
        parent_ids = _validate_parents(parents, node)
        selected_policy = _validate_policy(policy)
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("topic-ref")
        await self._require_topic(artifact_ref)
        candidate = LifecycleNode(
            node_id=node,
            family="topic",
            reference=_topic_payload(artifact_ref),
            parents=parent_ids,
            policy=selected_policy,
            revision=0,
            available=True,
        )
        await self._register_derived(candidate)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event")
        async with self._runtime._context(self.scope_id) as context:
            async with self._runtime._locked(self.scope_id):
                nodes = await self._load()
                selected = nodes.get(event.node_id)
                if selected is None or selected.family != "source":
                    raise SourceLifecycleError("unknown-source")
                if event.revision != selected.revision:
                    raise SourceLifecycleError("revision-mismatch")
                digest = _event_digest(event)
                if selected.last_sequence is not None:
                    if event.sequence < selected.last_sequence:
                        raise SourceLifecycleError("sequence-regression")
                    if event.sequence == selected.last_sequence:
                        if digest == selected.last_event_digest:
                            return sorted(node_id for node_id, value in _active_nodes(nodes).items() if value)
                        raise SourceLifecycleError("sequence-conflict")
                updated = LifecycleNode(
                    node_id=selected.node_id,
                    family=selected.family,
                    reference=selected.reference,
                    parents=selected.parents,
                    policy=selected.policy,
                    revision=selected.revision,
                    available=event.available,
                    last_sequence=event.sequence,
                    last_event_digest=digest,
                )
                nodes[selected.node_id] = updated
                active = _active_nodes(nodes)
                await self._converge_memory(context, nodes, active)
                await self._save(nodes, active)
                return sorted(node_id for node_id, value in active.items() if value)

    async def resolve(self, node_id: str, purpose: str = "generation") -> SourceRef | MemoryCitation | ArtifactRef:
        node = _validate_node_id(node_id)
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError("purpose")
        nodes = await self._load()
        selected = nodes.get(node)
        if selected is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == "generation" and not selected.available:
            raise SourceLifecycleError("unavailable")
        return _reference(selected)

    async def _register_derived(self, candidate: LifecycleNode) -> None:
        async with self._runtime._context(self.scope_id) as context:
            async with self._runtime._locked(self.scope_id):
                nodes = await self._load()
                existing = nodes.get(candidate.node_id)
                if existing is not None:
                    if self._is_same_registration(existing, candidate):
                        return
                    raise SourceLifecycleError("node-conflict")
                for parent in candidate.parents:
                    if parent not in nodes:
                        raise SourceLifecycleError("missing-parent")
                nodes[candidate.node_id] = candidate
                active = _active_nodes(nodes)
                await self._converge_memory(context, nodes, active)
                await self._save(nodes, active)

    async def _converge_memory(
        self,
        context: Any,
        nodes: Mapping[str, LifecycleNode],
        active: Mapping[str, bool],
    ) -> None:
        service = context.artifacts.memory
        try:
            head, entries = await service.head_entries(context.artifacts.memory_artifact_id)
        except ArtifactNotFoundError:
            return
        states = {item.entry_id: item.state for item in head.content.manifest.entries}
        by_entry = {entry.entry_id: entry for entry in entries}
        deactivate: list[Any] = []
        reactivate: list[Any] = []
        for node in nodes.values():
            if node.family != "memory":
                continue
            entry_id = node.reference.get("entry_id")
            if not isinstance(entry_id, str):
                continue
            entry = by_entry.get(entry_id)
            state = states.get(entry_id)
            if entry is None or state is None:
                continue
            if active[node.node_id] and state != "active":
                reactivate.append(entry)
            elif not active[node.node_id] and state == "active":
                deactivate.append(entry)
        if deactivate:
            head = await service.forget(head, entries=tuple(deactivate), reason="source-lifecycle")
        if reactivate:
            await service.reactivate(head, entries=tuple(reactivate), reason="source-lifecycle")

    async def _load(self) -> dict[str, LifecycleNode]:
        async with self._database.transaction() as connection:
            return await load_lifecycle_nodes(connection, self.scope_id)

    async def _save(
        self,
        nodes: Mapping[str, LifecycleNode],
        active: Mapping[str, bool] | None = None,
    ) -> None:
        resolved = _active_nodes(nodes) if active is None else active
        async with self._database.transaction() as connection:
            await replace_lifecycle_nodes(connection, self.scope_id, _with_availability(nodes, resolved))

    @staticmethod
    def _is_same_registration(existing: LifecycleNode, candidate: LifecycleNode) -> bool:
        return (
            existing.family == candidate.family
            and existing.parents == candidate.parents
            and existing.policy == candidate.policy
            and existing.revision == candidate.revision
            and _reference_key(existing) == _reference_key(candidate)
        )

    async def _require_source(self, reference: SourceRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(
                reference.source_type, reference.source_id
            )
        except Exception as error:
            raise SourceLifecycleError("source-not-readable") from error

    async def _require_memory(self, citation: MemoryCitation) -> None:
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        except Exception as error:
            raise SourceLifecycleError("memory-not-readable") from error

    async def _require_topic(self, reference: ArtifactRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                reference.family, reference.artifact_id, reference.revision
            )
        except Exception as error:
            raise SourceLifecycleError("topic-not-readable") from error


class SourceLifecycleApplication:
    """Select the lifecycle service for one Scope."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, scope_id)


__all__ = [
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "ScopedSourceLifecycle",
]
