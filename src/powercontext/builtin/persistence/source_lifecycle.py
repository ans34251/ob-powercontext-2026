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

"""Persist per-scope Source lifecycle dependency graphs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, false, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE


@dataclass(frozen=True, slots=True)
class LifecycleNode:
    """One registered lifecycle node, independent of its typed reference."""

    node_id: str
    family: str
    reference: Mapping[str, Any]
    parents: tuple[str, ...]
    policy: str
    revision: int
    available: bool
    last_sequence: int | None = None
    last_event_digest: str | None = None


async def load_lifecycle_nodes(connection: AsyncConnection, scope_id: str, /) -> dict[str, LifecycleNode]:
    rows = (
        await connection.execute(
            select(SOURCE_LIFECYCLE_NODES_TABLE).where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
        )
    ).mappings()
    nodes: dict[str, LifecycleNode] = {}
    for row in rows:
        node_id = str(row["node_id"])
        nodes[node_id] = LifecycleNode(
            node_id=node_id,
            family=str(row["family"]),
            reference=json.loads(str(row["reference"])),
            parents=tuple(str(parent) for parent in json.loads(str(row["parents"]))),
            policy=str(row["policy"]),
            revision=int(row["revision"]),
            available=bool(row["available"]),
            last_sequence=None if row["last_sequence"] is None else int(row["last_sequence"]),
            last_event_digest=None if row["last_event_digest"] is None else str(row["last_event_digest"]),
        )
    return nodes


async def replace_lifecycle_nodes(
    connection: AsyncConnection,
    scope_id: str,
    nodes: Iterable[LifecycleNode],
    /,
) -> None:
    table = SOURCE_LIFECYCLE_NODES_TABLE
    await connection.execute(delete(table).where(table.c.scope_id == scope_id))
    values = tuple(
        {
            "scope_id": scope_id,
            "node_id": node.node_id,
            "family": node.family,
            "reference": json.dumps(dict(node.reference), sort_keys=True, separators=(",", ":")),
            "parents": json.dumps(list(node.parents), separators=(",", ":")),
            "policy": node.policy,
            "revision": node.revision,
            "available": node.available,
            "last_sequence": node.last_sequence,
            "last_event_digest": node.last_event_digest,
        }
        for node in nodes
    )
    if values:
        await connection.execute(insert(table), values)


async def unavailable_topic_ids(connection: AsyncConnection, scope_id: str, /) -> frozenset[str]:
    """Return Topic artifact identities whose lifecycle support is currently withdrawn."""

    table = SOURCE_LIFECYCLE_NODES_TABLE
    rows = (
        await connection.execute(
            select(table.c.reference).where(
                table.c.scope_id == scope_id,
                table.c.family == "topic",
                table.c.available == false(),
            )
        )
    ).scalars()
    identities: set[str] = set()
    for payload in rows:
        reference = json.loads(str(payload))
        artifact_id = reference.get("artifact_id")
        if isinstance(artifact_id, str):
            identities.add(artifact_id)
    return frozenset(identities)


__all__ = [
    "LifecycleNode",
    "load_lifecycle_nodes",
    "replace_lifecycle_nodes",
    "unavailable_topic_ids",
]
