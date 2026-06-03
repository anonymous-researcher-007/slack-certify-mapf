"""Unit tests for slackcertify.certify.bounded."""

from __future__ import annotations

import pytest

from slackcertify.certify.bounded import is_delta_certified, unsafe_pairs_bounded
from slackcertify.core.conflict import EdgeConflict, VertexConflict
from slackcertify.core.plan import Agent, Path, Plan


def _shared_vertex_plan() -> Plan:
    """Two agents that cross at (1, 1) at t=2 — a delta=0 vertex conflict."""
    agents = [
        Agent(id=0, start=(0, 0), goal=(2, 1)),
        Agent(id=1, start=(1, 3), goal=(1, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (1, 1), (2, 1)]),
        Path(agent_id=1, vertices=[(1, 3), (1, 2), (1, 1), (1, 0)]),
    ]
    return Plan.from_paths(agents, paths)


def _delta_only_plan() -> Plan:
    """Agents visit (2, 0) at t=1 and t=3 — no delta=0 conflict, but delta=2 is."""
    agents = [
        Agent(id=0, start=(1, 0), goal=(3, 0)),
        Agent(id=1, start=(2, 3), goal=(2, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(1, 0), (2, 0), (3, 0)]),
        Path(agent_id=1, vertices=[(2, 3), (2, 2), (2, 1), (2, 0)]),
    ]
    return Plan.from_paths(agents, paths)


def _head_on_swap_plan() -> Plan:
    agents = [
        Agent(id=0, start=(0, 0), goal=(1, 0)),
        Agent(id=1, start=(1, 0), goal=(0, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
        Path(agent_id=1, vertices=[(1, 0), (0, 0)]),
    ]
    return Plan.from_paths(agents, paths)


def _trivial_single_agent_plan() -> Plan:
    return Plan.from_paths(
        agents=[Agent(id=0, start=(0, 0), goal=(2, 0))],
        paths=[Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])],
    )


@pytest.mark.unit
def test_certified_when_no_conflicts() -> None:
    plan = _trivial_single_agent_plan()
    assert is_delta_certified(plan, delta=0)
    assert is_delta_certified(plan, delta=5)
    assert unsafe_pairs_bounded(plan, delta=0) == []


@pytest.mark.unit
def test_uncertified_at_shared_vertex() -> None:
    plan = _shared_vertex_plan()
    assert not is_delta_certified(plan, delta=0)
    pairs = unsafe_pairs_bounded(plan, delta=0)
    assert any(isinstance(c, VertexConflict) and c.vertex == (1, 1) for c in pairs)


@pytest.mark.unit
def test_delta_widens_certification() -> None:
    plan = _delta_only_plan()
    assert is_delta_certified(plan, delta=0)
    assert not is_delta_certified(plan, delta=2)


@pytest.mark.unit
def test_uncertified_on_head_on_swap() -> None:
    plan = _head_on_swap_plan()
    assert not is_delta_certified(plan, delta=0)
    assert any(isinstance(c, EdgeConflict) for c in unsafe_pairs_bounded(plan, delta=0))


@pytest.mark.unit
def test_negative_delta_rejected() -> None:
    plan = _trivial_single_agent_plan()
    with pytest.raises(ValueError, match="non-negative"):
        is_delta_certified(plan, delta=-1)
    with pytest.raises(ValueError, match="non-negative"):
        unsafe_pairs_bounded(plan, delta=-1)


@pytest.mark.unit
def test_certificate_round_trip_bounded() -> None:
    from datetime import datetime, timezone

    from slackcertify.certify.certificate import Certificate

    plan = _trivial_single_agent_plan()
    cert = Certificate(
        mode="bounded",
        delta=0,
        p_d=None,
        epsilon=None,
        total_wait_inserted=0,
        per_conflict_waits={},
        solver_used="trivial",
        plan_hash=Certificate.hash_plan(plan),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        proof_sketch="single-agent plan; trivially Delta-certified",
    )
    payload = cert.to_json()
    cert2 = Certificate.from_json(payload)
    assert cert == cert2
    assert cert2.verify(plan) is True

    # Wrong plan -> hash mismatch -> verify returns False.
    other_plan = Plan.from_paths(
        agents=[Agent(id=0, start=(0, 0), goal=(0, 0))],
        paths=[Path(agent_id=0, vertices=[(0, 0)])],
    )
    assert cert.verify(other_plan) is False


@pytest.mark.unit
def test_certificate_rejects_inconsistent_mode_args() -> None:
    """`delta` is required in bounded mode and accepted (but optional) in
    probabilistic mode as the Δ-window for arrival-coincidence."""
    from datetime import datetime, timezone

    from pydantic import ValidationError

    from slackcertify.certify.certificate import Certificate

    base = dict(
        total_wait_inserted=0,
        per_conflict_waits={},
        solver_used="x",
        plan_hash="abc",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        proof_sketch="ok",
    )
    # bounded mode requires `delta`.
    with pytest.raises(ValidationError, match="bounded"):
        Certificate(mode="bounded", delta=None, p_d=None, epsilon=None, **base)
    # bounded mode must not set p_d / epsilon.
    with pytest.raises(ValidationError, match="bounded"):
        Certificate(mode="bounded", delta=0, p_d=0.1, epsilon=None, **base)
    # probabilistic mode requires p_d and epsilon.
    with pytest.raises(ValidationError, match="probabilistic"):
        Certificate(mode="probabilistic", delta=None, p_d=None, epsilon=0.05, **base)
    # probabilistic mode now ACCEPTS `delta` as the Δ-window for the
    # |arrival-difference - signed_gap| <= delta unsafe event.
    cert = Certificate(mode="probabilistic", delta=2, p_d=0.1, epsilon=0.05, **base)
    assert cert.delta == 2
