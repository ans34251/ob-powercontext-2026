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

"""HTTP scoring-service reranking for coarse Memory candidates."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import (
    MemoryRerankDecision,
    _validate_bounds,
)
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score candidates through an HTTP service and select from the whole pool."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once, then select the globally best ranks."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = tuple(enumerate(candidates[start : start + self._batch_size], start=start + 1))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [{"rank": rank, "text": hit.text} for rank, hit in batch],
                },
            )
            response.raise_for_status()
            requests += 1
            scores.update(_parse_scores(response, expected_ranks=frozenset(rank for rank, _ in batch)))
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _ in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _parse_scores(response: httpx.Response, *, expected_ranks: frozenset[int]) -> dict[int, float]:
    payload: Any = response.json()
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly the 'scores' field")  # noqa: TRY003
    entries = payload["scores"]
    if not isinstance(entries, list):
        raise ValueError("rerank response field 'scores' must be a list")  # noqa: TRY003
    scores: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
            raise ValueError("each rerank score must contain exactly 'rank' and 'score'")  # noqa: TRY003
        rank = entry["rank"]
        score = entry["score"]
        if type(rank) is not int or rank not in expected_ranks or rank in scores:
            raise ValueError("rerank score ranks must cover this batch exactly once")  # noqa: TRY003
        if type(score) is not int and type(score) is not float:
            raise ValueError("rerank score must be a finite number")  # noqa: TRY003
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("rerank score must be a finite number")  # noqa: TRY003
        scores[rank] = numeric_score
    if set(scores) != expected_ranks:
        raise ValueError("rerank score ranks must cover this batch exactly once")  # noqa: TRY003
    return scores


__all__ = ["HTTPMemoryReranker"]
