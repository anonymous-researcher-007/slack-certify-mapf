"""Stable hashing utilities for plans and other JSON-serialisable artefacts."""

from __future__ import annotations

import hashlib
import json

from slackcertify.core.plan import Plan

__all__ = ["plan_hash"]


def plan_hash(plan: Plan) -> str:
    """Return a SHA-256 hex digest of ``plan``'s canonical JSON form.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
    >>> len(plan_hash(Plan.from_paths(a, p))) == 64
    True
    """
    payload = json.dumps(
        json.loads(plan.model_dump_json()), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
