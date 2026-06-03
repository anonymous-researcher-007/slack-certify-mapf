"""Slack-certificate value object.

A :class:`Certificate` is the machine-checkable artefact emitted alongside a
slack-certified plan. It is small (a single Pydantic model serialisable to
JSON), self-describing (carries the mode, parameters, and a human-readable
``proof_sketch``), and reproducibly verifiable (the :meth:`verify` method
re-checks the relevant inequality from scratch against a provided plan).

Examples
--------
>>> from datetime import datetime, timezone
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
>>> plan = Plan.from_paths(a, p)
>>> cert = Certificate(
...     mode="bounded",
...     delta=0,
...     p_d=None,
...     epsilon=None,
...     total_wait_inserted=0,
...     per_conflict_waits={},
...     solver_used="trivial-single-agent",
...     plan_hash=Certificate.hash_plan(plan),
...     created_at=datetime(2026, 5, 14, tzinfo=timezone.utc),
...     proof_sketch="Single-agent plan; no inter-agent constraints to check.",
... )
>>> cert.verify(plan)
True
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slackcertify.certify.bounded import is_delta_certified
from slackcertify.certify.probabilistic import is_epsilon_certified
from slackcertify.core.plan import Plan

__all__ = ["Certificate"]


class Certificate(BaseModel):
    """Machine-checkable record of a slack-certification result.

    ``delta`` has a single, unified meaning across both modes:

    * In ``"bounded"`` mode it is the Δ-disjointness margin: the
      certified plan satisfies ``|t_i - t_j| > delta`` on every
      shared-vertex / reverse-edge pair (Definition 1 of §III).
    * In ``"probabilistic"`` mode it is the Δ-window of arrival-time
      coincidence used by the per-conflict risk formula:
      ``|D_i(t_i) - D_j(t_j) - (t_j - t_i)| <= delta`` is the
      *unsafe event* whose probability the union bound aggregates.
      ``delta`` is **optional** here (default ``None`` is treated as
      ``0`` — the point-collision form).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["bounded", "probabilistic"]
    delta: int | None = Field(default=None, ge=0)
    p_d: float | None = Field(default=None, ge=0.0, le=1.0)
    epsilon: float | None = Field(default=None, ge=0.0)
    total_wait_inserted: int = Field(ge=0)
    outer_rounds: int = Field(default=0, ge=0)
    initial_unsafe_count: int = Field(default=0, ge=0)
    cumulative_unsafe_count: int = Field(default=0, ge=0)
    per_conflict_waits: dict[str, int]
    solver_used: str = Field(min_length=1)
    plan_hash: str = Field(min_length=1)
    created_at: datetime
    proof_sketch: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_mode_parameters(self) -> Certificate:
        """Reject mode / delta / p_d / epsilon combinations that don't match a valid certificate."""
        if self.mode == "bounded":
            if self.delta is None:
                raise ValueError("mode='bounded' requires a non-null delta")
            if self.p_d is not None or self.epsilon is not None:
                raise ValueError("mode='bounded' must not set p_d or epsilon")
        else:  # probabilistic
            if self.p_d is None or self.epsilon is None:
                raise ValueError("mode='probabilistic' requires both p_d and epsilon")
            # delta is OPTIONAL in probabilistic mode: it acts as the
            # Δ-margin in the |arrival-difference - signed_gap| ≤ delta
            # unsafe event (single, unified semantics with bounded mode).
            # The field's own ge=0 constraint covers the non-negative
            # check; default None is treated downstream as 0
            # (point-collision).
        for k, v in self.per_conflict_waits.items():
            if v < 0:
                raise ValueError(f"per_conflict_waits[{k!r}] must be non-negative, got {v}")
        return self

    # --------------------------------------------------------------- accessors

    @property
    def nodes(self) -> list[tuple[int, tuple[int, int], int]]:
        """Compatibility shim for :func:`slackcertify.core.validators.validate_certificate`.

        The current Certificate carries aggregate statistics rather than an
        explicit node list, so this returns an empty list. Future revisions
        that record per-conflict repair edges should populate it.
        """
        return []

    # -------------------------------------------------------------- (de)serial

    def to_json(self) -> str:
        """Return a UTF-8 JSON string representation.

        Examples
        --------
        >>> from datetime import datetime, timezone
        >>> Certificate(
        ...     mode="bounded", delta=0, p_d=None, epsilon=None,
        ...     total_wait_inserted=0, per_conflict_waits={},
        ...     solver_used="x", plan_hash="abc",
        ...     created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ...     proof_sketch="ok",
        ... ).to_json()[:9]
        '{"mode":"'
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> Certificate:
        """Parse a :class:`Certificate` from JSON ``payload``.

        Examples
        --------
        >>> from datetime import datetime, timezone
        >>> raw = Certificate(
        ...     mode="bounded", delta=0, p_d=None, epsilon=None,
        ...     total_wait_inserted=0, per_conflict_waits={},
        ...     solver_used="x", plan_hash="abc",
        ...     created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ...     proof_sketch="ok",
        ... ).to_json()
        >>> Certificate.from_json(raw).mode
        'bounded'
        """
        return cls.model_validate_json(payload)

    # ----------------------------------------------------------------- verify

    def verify(self, plan: Plan) -> bool:
        """Re-check the certificate's claim against ``plan`` from scratch.

        For ``mode='bounded'`` this calls
        :func:`slackcertify.certify.bounded.is_delta_certified`; for
        ``mode='probabilistic'`` it calls
        :func:`slackcertify.certify.probabilistic.is_epsilon_certified`.
        Also re-derives ``plan_hash`` and confirms it matches.

        Examples
        --------
        >>> from datetime import datetime, timezone
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
        >>> plan = Plan.from_paths(a, p)
        >>> Certificate(
        ...     mode="bounded", delta=0, p_d=None, epsilon=None,
        ...     total_wait_inserted=0, per_conflict_waits={},
        ...     solver_used="trivial", plan_hash=Certificate.hash_plan(plan),
        ...     created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ...     proof_sketch="single agent",
        ... ).verify(plan)
        True
        """
        if self.hash_plan(plan) != self.plan_hash:
            return False
        if self.mode == "bounded":
            assert self.delta is not None  # invariant from validator
            return is_delta_certified(plan, self.delta)
        # probabilistic
        assert self.p_d is not None and self.epsilon is not None
        delta_window = self.delta if self.delta is not None else 0
        return is_epsilon_certified(plan, self.p_d, self.epsilon, delta_window=delta_window)

    # ------------------------------------------------------------------ hash

    @staticmethod
    def hash_plan(plan: Plan) -> str:
        """Return a stable SHA-256 fingerprint of ``plan``.

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
        >>> len(Certificate.hash_plan(Plan.from_paths(a, p))) == 64
        True
        """
        # model_dump_json produces a canonical representation suitable as
        # input to the digest; sort keys for cross-platform determinism.
        payload = json.dumps(
            json.loads(plan.model_dump_json()), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
