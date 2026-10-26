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

"""Cross-call Embedding request coalescing with an LRU vector cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_COALESCE_MS = 5.0
DEFAULT_CACHE_SIZE = 1024


@dataclass(slots=True)
class _Batch:
    texts: tuple[str, ...]
    waiters: tuple[_Waiter, ...]
    usage: InferenceUsage = field(default_factory=lambda: InferenceUsage(requests=0))


@dataclass(slots=True)
class _Waiter:
    future: asyncio.Future[EmbeddingVector]
    batch: _Batch | None = None


class CoalescingEmbeddingModel:
    """Share in-flight Embedding work across calls and cache successful vectors."""

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be at least one")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must not be negative")  # noqa: TRY003
        self.profile = model.profile
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1_000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, _Waiter] = {}
        self._queued: list[str] = []
        self._scheduler: asyncio.Task[None] | None = None

    async def embed(self, texts: Any) -> EmbeddingResult:
        """Return vectors in the caller's original order with duplicate positions kept."""

        requested = tuple(texts)
        if not requested:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        if not all(isinstance(text, str) for text in requested):
            raise ValueError("embedding texts must be strings")  # noqa: TRY003
        unique = tuple(dict.fromkeys(requested))
        waiters: dict[str, _Waiter] = {}
        queued: list[str] = []
        for text in unique:
            cached = self._cached(text)
            if cached is not None:
                waiters[text] = _Waiter(_resolved(cached))
                continue
            inflight = self._inflight.get(text)
            if inflight is not None:
                waiters[text] = inflight
                continue
            waiter = _Waiter(asyncio.get_running_loop().create_future())
            self._inflight[text] = waiter
            waiters[text] = waiter
            queued.append(text)
        if queued:
            self._queued.extend(queued)
            self._ensure_scheduler()
        try:
            await asyncio.gather(*(asyncio.shield(waiters[text].future) for text in unique))
        except BaseException:
            for waiter in waiters.values():
                if waiter.future.done() and not waiter.future.cancelled():
                    waiter.future.exception()
            raise
        batches = {id(waiter.batch): waiter.batch for waiter in waiters.values() if waiter.batch is not None}
        return EmbeddingResult(
            vectors=tuple(waiters[text].future.result() for text in requested),
            usage=_combine_usage(batch.usage for batch in batches.values() if batch is not None),
        )

    def _cached(self, text: str) -> EmbeddingVector | None:
        if self._cache_size == 0:
            return None
        vector = self._cache.get(text)
        if vector is not None:
            self._cache.move_to_end(text)
        return vector

    def _remember(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _ensure_scheduler(self) -> None:
        if self._scheduler is None or self._scheduler.done():
            self._scheduler = asyncio.get_running_loop().create_task(self._dispatch())

    async def _dispatch(self) -> None:
        while self._queued:
            if self._coalesce_seconds > 0:
                await asyncio.sleep(self._coalesce_seconds)
            while self._queued:
                chunk = tuple(self._queued[: self._max_batch_size])
                del self._queued[: self._max_batch_size]
                batch = _Batch(texts=chunk, waiters=tuple(self._inflight[text] for text in chunk))
                for waiter in batch.waiters:
                    waiter.batch = batch
                asyncio.get_running_loop().create_task(self._run(batch))

    async def _run(self, batch: _Batch) -> None:
        try:
            result = await self._model.embed(batch.texts)
            vectors = tuple(result.vectors)
            if len(vectors) != len(batch.texts):
                raise ValueError("embedding provider returned the wrong number of vectors")  # noqa: TRY003
            batch.usage = result.usage
        except BaseException as error:
            for text in batch.texts:
                self._inflight.pop(text, None)
            for waiter in batch.waiters:
                if not waiter.future.done():
                    waiter.future.set_exception(error)
            return
        for text, vector in zip(batch.texts, vectors, strict=True):
            self._remember(text, vector)
            waiter = self._inflight.pop(text, None)
            if waiter is not None and not waiter.future.done():
                waiter.future.set_result(vector)


def _resolved(vector: EmbeddingVector) -> asyncio.Future[EmbeddingVector]:
    future: asyncio.Future[EmbeddingVector] = asyncio.get_running_loop().create_future()
    future.set_result(vector)
    return future


def _combine_usage(usages: Any) -> InferenceUsage:
    collected = tuple(usages)
    if not collected:
        return InferenceUsage(requests=0)
    return InferenceUsage(
        requests=sum(usage.requests for usage in collected),
        input_tokens=_sum_optional(usage.input_tokens for usage in collected),
        output_tokens=_sum_optional(usage.output_tokens for usage in collected),
    )


def _sum_optional(values: Any) -> int | None:
    total = 0
    observed = False
    for value in values:
        if value is not None:
            total += value
            observed = True
    return total if observed else None


__all__ = ["CoalescingEmbeddingModel"]
