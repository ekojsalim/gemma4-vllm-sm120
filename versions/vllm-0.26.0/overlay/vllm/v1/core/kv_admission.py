# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transactional admission credits for hybrid KV allocation.

The scheduler remains the sole owner of this ledger. The ledger does not own
or trim cache blocks; it conservatively couples future persistent admission,
live physical ownership, and every submitted-but-unsettled batch upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def get_request_admission_length(
    *, prompt_tokens: int, max_tokens: int, max_model_len: int
) -> int:
    """Return the declared request extent that must remain satisfiable."""
    if prompt_tokens <= 0 or max_tokens <= 0 or max_model_len <= 0:
        raise ValueError("admission lengths must be positive")
    return min(prompt_tokens + max_tokens, max_model_len)


def validate_hybrid_kv_admission(
    *,
    has_connector: bool,
    is_encoder_decoder: bool,
    has_mamba_layers: bool,
    prefix_caching: bool,
    cache_groups_supported: bool,
    max_concurrent_batches: int,
) -> None:
    """Fail closed outside supported global hybrid-cache configurations."""
    if has_connector:
        raise ValueError("hybrid KV admission does not support connectors")
    if is_encoder_decoder or has_mamba_layers:
        raise ValueError(
            "hybrid KV admission supports decoder-only attention caches"
        )
    if prefix_caching:
        raise ValueError("hybrid KV admission requires prefix caching disabled")
    if not cache_groups_supported:
        raise ValueError(
            "hybrid KV admission supports only SWA/full-attention groups"
        )
    if max_concurrent_batches > 2:
        raise ValueError("hybrid KV admission supports at most two GPU batches")


@dataclass(frozen=True)
class AdmissionReservation:
    batch_id: int
    request_id: str
    previous_persistent_blocks: int | None
    previous_physical_blocks: int | None
    previous_batch_upper_bound: int | None


@dataclass
class _BatchReservation:
    physical_upper_bounds: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _RetiredReservation:
    blocks: int
    blocking_batches: frozenset[int]


class HybridKVAdmissionLedger:
    """Fail-closed credits over one global hybrid KV block pool.

    For each active request the charged amount is::

        max(persistent, live physical, every active batch upper bound)

    Persistent capacity can therefore absorb bounded transient ownership that
    is already physically inside the request's future reservation, while a
    page crossing beyond that reservation remains charged exactly once.
    """

    def __init__(self, capacity_blocks: int, max_in_flight_batches: int) -> None:
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        if max_in_flight_batches <= 0:
            raise ValueError("max_in_flight_batches must be positive")
        self.capacity_blocks = capacity_blocks
        self.max_in_flight_batches = max_in_flight_batches
        self.persistent: dict[str, int] = {}
        self.physical: dict[str, int] = {}
        self.batches: dict[int, _BatchReservation] = {}
        self.retired: dict[str, _RetiredReservation] = {}

    @property
    def persistent_blocks(self) -> int:
        return sum(self.persistent.values())

    @property
    def physical_blocks(self) -> int:
        return sum(self.physical.values())

    @property
    def provisional_blocks(self) -> int:
        total = 0
        for request_id in self._active_request_ids():
            base = max(
                self.persistent.get(request_id, 0),
                self.physical.get(request_id, 0),
            )
            total += max(0, self._request_batch_bound(request_id) - base)
        return total

    @property
    def stranded_blocks(self) -> int:
        """Physical tail blocks beyond persistence with no active batch."""
        active_batch_requests = {
            request_id
            for batch in self.batches.values()
            for request_id in batch.physical_upper_bounds
        }
        return sum(
            max(0, blocks - self.persistent.get(request_id, 0))
            for request_id, blocks in self.physical.items()
            if request_id not in active_batch_requests
        )

    @property
    def retired_blocks(self) -> int:
        return sum(item.blocks for item in self.retired.values())

    @property
    def charged_blocks(self) -> int:
        active = sum(
            max(
                self.persistent.get(request_id, 0),
                self.physical.get(request_id, 0),
                self._request_batch_bound(request_id),
            )
            for request_id in self._active_request_ids()
        )
        return active + self.retired_blocks

    @property
    def free_blocks(self) -> int:
        return self.capacity_blocks - self.charged_blocks

    def batch_request_bounds(self, batch_id: int) -> dict[str, int]:
        batch = self.batches.get(batch_id)
        return dict(batch.physical_upper_bounds) if batch is not None else {}

    def can_reserve(
        self,
        *,
        batch_id: int,
        request_id: str,
        persistent_blocks: int,
        physical_blocks: int,
        physical_upper_bound: int,
    ) -> bool:
        return self._staged_charge(
            batch_id=batch_id,
            request_id=request_id,
            persistent_blocks=persistent_blocks,
            physical_blocks=physical_blocks,
            physical_upper_bound=physical_upper_bound,
        ) <= self.capacity_blocks

    def try_reserve(
        self,
        *,
        batch_id: int,
        request_id: str,
        persistent_blocks: int,
        physical_blocks: int,
        physical_upper_bound: int,
    ) -> AdmissionReservation | None:
        """Atomically reserve one request's exact allocation upper bound."""
        self._validate_counts(
            persistent_blocks, physical_blocks, physical_upper_bound
        )
        if request_id in self.retired:
            raise ValueError(f"request {request_id!r} is retired")
        batch = self.batches.get(batch_id)
        if batch is None and len(self.batches) >= self.max_in_flight_batches:
            return None
        if batch is not None and request_id in batch.physical_upper_bounds:
            raise ValueError(
                f"request {request_id!r} already reserved in batch {batch_id}"
            )
        if not self.can_reserve(
            batch_id=batch_id,
            request_id=request_id,
            persistent_blocks=persistent_blocks,
            physical_blocks=physical_blocks,
            physical_upper_bound=physical_upper_bound,
        ):
            return None

        previous_persistent = self.persistent.get(request_id)
        previous_physical = self.physical.get(request_id)
        previous_batch_bound = (
            batch.physical_upper_bounds.get(request_id) if batch else None
        )
        self.persistent[request_id] = max(
            persistent_blocks, previous_persistent or 0
        )
        self.physical[request_id] = physical_blocks
        if batch is None:
            batch = self.batches.setdefault(batch_id, _BatchReservation())
        batch.physical_upper_bounds[request_id] = physical_upper_bound
        self._assert_invariants()
        return AdmissionReservation(
            batch_id=batch_id,
            request_id=request_id,
            previous_persistent_blocks=previous_persistent,
            previous_physical_blocks=previous_physical,
            previous_batch_upper_bound=previous_batch_bound,
        )

    def commit_allocation(
        self, reservation: AdmissionReservation, physical_blocks: int
    ) -> None:
        """Bind a successful reservation to the observed physical result."""
        if physical_blocks < 0:
            raise ValueError("physical_blocks must be non-negative")
        batch = self.batches.get(reservation.batch_id)
        if batch is None or reservation.request_id not in batch.physical_upper_bounds:
            raise KeyError((reservation.batch_id, reservation.request_id))
        planned = batch.physical_upper_bounds[reservation.request_id]
        if physical_blocks > planned:
            raise RuntimeError(
                f"physical allocation {physical_blocks} exceeded bound {planned}"
            )
        self.physical[reservation.request_id] = physical_blocks
        batch.physical_upper_bounds[reservation.request_id] = physical_blocks
        self._assert_invariants()

    def rollback(self, reservation: AdmissionReservation) -> None:
        """Roll back one reservation after allocation failed."""
        batch = self.batches.get(reservation.batch_id)
        if batch is None or reservation.request_id not in batch.physical_upper_bounds:
            raise KeyError((reservation.batch_id, reservation.request_id))
        if reservation.previous_batch_upper_bound is None:
            del batch.physical_upper_bounds[reservation.request_id]
        else:
            batch.physical_upper_bounds[reservation.request_id] = (
                reservation.previous_batch_upper_bound
            )
        if not batch.physical_upper_bounds:
            del self.batches[reservation.batch_id]
        self._restore(
            self.persistent,
            reservation.request_id,
            reservation.previous_persistent_blocks,
        )
        self._restore(
            self.physical,
            reservation.request_id,
            reservation.previous_physical_blocks,
        )
        self._assert_invariants()

    def settle_batch(
        self,
        batch_id: int,
        physical_blocks: dict[str, int],
        persistent_blocks: dict[str, int],
    ) -> None:
        """Promote/roll back only after GPU output and rejection are resolved."""
        batch = self.batches.get(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        active_request_ids = [
            request_id
            for request_id in batch.physical_upper_bounds
            if request_id not in self.retired
        ]
        for request_id in active_request_ids:
            if request_id not in physical_blocks or request_id not in persistent_blocks:
                raise ValueError(f"missing settlement state for {request_id!r}")
            physical = physical_blocks[request_id]
            persistent = persistent_blocks[request_id]
            if physical < 0 or persistent <= 0:
                raise ValueError("invalid settlement block counts")

        next_physical = dict(self.physical)
        next_persistent = dict(self.persistent)
        for request_id in active_request_ids:
            next_physical[request_id] = physical_blocks[request_id]
            next_persistent[request_id] = max(
                next_persistent.get(request_id, 0),
                persistent_blocks[request_id],
            )

        self.physical = next_physical
        self.persistent = next_persistent
        del self.batches[batch_id]
        self._drain_retired()
        self._assert_invariants()

    def finish_request(self, request_id: str, physical_blocks: int) -> None:
        """Release now or retire until all submitted batches have settled."""
        if physical_blocks < 0:
            raise ValueError("physical_blocks must be non-negative")
        if request_id in self.retired:
            raise ValueError(f"request {request_id!r} is already retired")
        persistent = self.persistent.pop(request_id, 0)
        recorded_physical = self.physical.pop(request_id, 0)
        blockers = frozenset(
            batch_id
            for batch_id, batch in self.batches.items()
            if request_id in batch.physical_upper_bounds
        )
        charged = max(
            persistent,
            physical_blocks,
            recorded_physical,
            self._request_batch_bound(request_id),
        )
        if charged and blockers:
            self.retired[request_id] = _RetiredReservation(charged, blockers)
        self._drain_retired()
        self._assert_invariants()

    def _staged_charge(
        self,
        *,
        batch_id: int,
        request_id: str,
        persistent_blocks: int,
        physical_blocks: int,
        physical_upper_bound: int,
    ) -> int:
        self._validate_counts(
            persistent_blocks, physical_blocks, physical_upper_bound
        )
        if request_id in self.retired:
            return self.capacity_blocks + 1
        batch = self.batches.get(batch_id)
        if batch is None and len(self.batches) >= self.max_in_flight_batches:
            return self.capacity_blocks + 1

        active_ids = self._active_request_ids() | {request_id}
        total = self.retired_blocks
        for active_id in active_ids:
            persistent = self.persistent.get(active_id, 0)
            physical = self.physical.get(active_id, 0)
            bound = self._request_batch_bound(active_id)
            if active_id == request_id:
                persistent = max(persistent, persistent_blocks)
                physical = physical_blocks
                bound = max(bound, physical_upper_bound)
            total += max(persistent, physical, bound)
        return total

    def _active_request_ids(self) -> set[str]:
        active = self.persistent.keys() | self.physical.keys() | {
            request_id
            for batch in self.batches.values()
            for request_id in batch.physical_upper_bounds
        }
        return set(active).difference(self.retired)

    def _request_batch_bound(self, request_id: str) -> int:
        return max(
            (
                batch.physical_upper_bounds.get(request_id, 0)
                for batch in self.batches.values()
            ),
            default=0,
        )

    def _drain_retired(self) -> None:
        active_batches = self.batches.keys()
        for request_id in [
            request_id
            for request_id, item in self.retired.items()
            if item.blocking_batches.isdisjoint(active_batches)
        ]:
            del self.retired[request_id]

    @staticmethod
    def _validate_counts(
        persistent_blocks: int,
        physical_blocks: int,
        physical_upper_bound: int,
    ) -> None:
        if persistent_blocks <= 0:
            raise ValueError("persistent_blocks must be positive")
        if physical_blocks < 0:
            raise ValueError("physical_blocks must be non-negative")
        if physical_upper_bound < physical_blocks:
            raise ValueError("physical upper bound is below current ownership")

    @staticmethod
    def _restore(mapping: dict[str, int], key: str, value: int | None) -> None:
        if value is None:
            mapping.pop(key, None)
        else:
            mapping[key] = value

    def _assert_invariants(self) -> None:
        if self.free_blocks < 0:
            raise RuntimeError("hybrid KV admission exceeded capacity")
        if len(self.batches) > self.max_in_flight_batches:
            raise RuntimeError("hybrid KV admission exceeded the batch bound")
        if not self.persistent.keys().isdisjoint(self.retired):
            raise RuntimeError("retired request still has persistent credit")
        if not self.physical.keys().isdisjoint(self.retired):
            raise RuntimeError("retired request still has physical ownership")
        if self.charged_blocks < self.physical_blocks + self.retired_blocks:
            raise RuntimeError("logical KV charge undercounts physical ownership")
