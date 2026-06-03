"""Plan IO — JSON for archival, Parquet for batch analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from slackcertify.core.plan import Plan

__all__ = ["load_plan", "save_plan", "save_plan_compact"]


def save_plan(plan: Plan, path: str | Path) -> None:
    """Write ``plan`` to ``path`` as UTF-8 JSON.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from slackcertify.core.plan import Agent, Path as APath, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [APath(agent_id=0, vertices=[(0, 0)])]
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "plan.json"
    >>> save_plan(Plan.from_paths(a, p), tmp)
    >>> tmp.read_text()[:9]
    '{"agents"'
    """
    Path(path).write_text(plan.model_dump_json(indent=2), encoding="utf-8")


def load_plan(path: str | Path) -> Plan:
    """Read a :class:`Plan` previously written by :func:`save_plan`.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from slackcertify.core.plan import Agent, Path as APath, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [APath(agent_id=0, vertices=[(0, 0)])]
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "plan.json"
    >>> save_plan(Plan.from_paths(a, p), tmp)
    >>> load_plan(tmp).makespan
    0
    """
    return Plan.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_plan_compact(plan: Plan, path: str | Path) -> None:
    """Write ``plan`` to ``path`` as a compact Parquet file for batch analytics.

    The schema is one row per ``(agent_id, time_step)`` pair with
    ``vx`` / ``vy`` columns. Plan-level metadata (``makespan``,
    ``sum_of_costs``) is stored in the Parquet metadata so the file is
    fully self-describing.

    Examples
    --------
    >>> import tempfile, pathlib, pandas as pd
    >>> from slackcertify.core.plan import Agent, Path as APath, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
    >>> p = [APath(agent_id=0, vertices=[(0, 0), (1, 0)])]
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "plan.parquet"
    >>> save_plan_compact(Plan.from_paths(a, p), tmp)
    >>> df = pd.read_parquet(tmp)
    >>> sorted(df.columns.tolist())
    ['agent_id', 'time_step', 'vx', 'vy']
    """
    rows: list[dict[str, int]] = []
    for path_obj in plan.paths:
        for t, (x, y) in enumerate(path_obj.vertices):
            rows.append({"agent_id": path_obj.agent_id, "time_step": t, "vx": x, "vy": y})
    df = pd.DataFrame(rows, columns=["agent_id", "time_step", "vx", "vy"])
    df.to_parquet(Path(path), index=False)
