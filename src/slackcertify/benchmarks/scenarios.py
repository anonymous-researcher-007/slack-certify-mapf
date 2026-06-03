"""Scenario registry for the MovingAI benchmark suite.

Centralises the file-system layout of ``benchmarks/{maps,scenarios}/``
and the family-name mapping for the five maps used by §V of the paper.
The registry is deliberately tolerant: missing files emit a warning
rather than raising, so partial benchmark sets are usable for smoke
tests.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import ClassVar

from slackcertify.benchmarks.movingai import make_instance
from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent

__all__ = ["ScenarioRegistry"]


_FETCH_HINT = "Run scripts/fetch_benchmarks.sh to download the MovingAI benchmark suite."


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` looking for the first directory with ``pyproject.toml``."""
    here = start.resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"could not locate repo root (no pyproject.toml above {start})")


class ScenarioRegistry:
    """Resolve MovingAI map / scen file paths and load ``(graph, agents)``.

    Parameters
    ----------
    root
        Directory containing ``maps/`` and ``scenarios/``. Defaults to
        ``<repo_root>/benchmarks/``.
    """

    #: The five maps used by the §V experiments, with their MovingAI families.
    MAP_FAMILIES: ClassVar[dict[str, str]] = {
        "warehouse-10-20-10-2-1": "warehouse",
        "warehouse-20-40-10-2-2": "warehouse",
        "random-32-32-20": "random",
        "room-64-64-16": "room",
        "maze-32-32-4": "maze",
    }

    def __init__(self, root: Path | None = None) -> None:
        """Store the benchmark root path; defaults to ``<repo>/benchmarks/``."""
        if root is None:
            root = _find_repo_root(Path(__file__)) / "benchmarks"
        self._root: Path = Path(root)

    @property
    def root(self) -> Path:
        """Root directory of the registry."""
        return self._root

    # ------------------------------------------------------------- paths

    def map_path(self, map_name: str) -> Path:
        """Return the ``.map`` path for ``map_name`` or raise with a hint."""
        path = self._root / "maps" / f"{map_name}.map"
        if not path.exists():
            raise FileNotFoundError(f"MovingAI map not found: {path}. {_FETCH_HINT}")
        return path

    def scen_path(self, map_name: str, scen_number: int) -> Path:
        """Return the random ``.scen`` path for ``(map_name, scen_number)``."""
        path = self._root / "scenarios" / f"{map_name}-random-{scen_number}.scen"
        if not path.exists():
            raise FileNotFoundError(f"MovingAI scenario not found: {path}. {_FETCH_HINT}")
        return path

    # --------------------------------------------------------- loaders

    def load_instance(
        self,
        map_name: str,
        n_agents: int,
        scen_seed: int,
        scen_index: int = 0,
    ) -> tuple[GridGraph, list[Agent]]:
        """Load a single instance via :func:`make_instance`."""
        return make_instance(
            self.map_path(map_name),
            self.scen_path(map_name, scen_seed),
            n_agents=n_agents,
            scen_index=scen_index,
        )

    def enumerate_instances(
        self,
        maps: Iterable[str],
        agent_counts: Iterable[int],
        scen_seeds: Iterable[int],
        scen_index: int = 0,
    ) -> Iterator[tuple[str, int, int, GridGraph, list[Agent]]]:
        """Yield ``(map_name, n_agents, scen_seed, graph, agents)`` for the cross product.

        Skips any combination whose map or scen file is missing — the
        loop emits a :class:`UserWarning` and continues, so partial
        benchmark sets are still usable for smoke tests.
        """
        maps_list = list(maps)
        agent_counts_list = list(agent_counts)
        scen_seeds_list = list(scen_seeds)
        for map_name in maps_list:
            for n_agents in agent_counts_list:
                for scen_seed in scen_seeds_list:
                    try:
                        graph, agents = self.load_instance(
                            map_name, n_agents, scen_seed, scen_index
                        )
                    except FileNotFoundError as exc:
                        warnings.warn(
                            f"skipping {map_name}/seed={scen_seed}/" f"n_agents={n_agents}: {exc}",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue
                    yield map_name, n_agents, scen_seed, graph, agents
