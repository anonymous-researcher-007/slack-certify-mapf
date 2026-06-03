"""Unit tests for slackcertify.benchmarks.scenarios.ScenarioRegistry."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pytest

from slackcertify.benchmarks.scenarios import ScenarioRegistry

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _populate_tiny_registry(root: Path, scen_seeds: list[int]) -> ScenarioRegistry:
    """Mirror the fixture map / scen under ``root`` for the requested seeds."""
    maps = root / "maps"
    scens = root / "scenarios"
    maps.mkdir(parents=True, exist_ok=True)
    scens.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA_DIR / "tiny_4x4.map", maps / "tiny_4x4.map")
    for s in scen_seeds:
        shutil.copy(
            DATA_DIR / "tiny_4x4-random-1.scen",
            scens / f"tiny_4x4-random-{s}.scen",
        )
    return ScenarioRegistry(root=root)


@pytest.mark.unit
def test_map_path_resolves_correctly(tmp_path: Path) -> None:
    reg = _populate_tiny_registry(tmp_path, scen_seeds=[1])
    assert reg.map_path("tiny_4x4") == tmp_path / "maps" / "tiny_4x4.map"


@pytest.mark.unit
def test_scen_path_resolves_correctly_for_scen_number(tmp_path: Path) -> None:
    reg = _populate_tiny_registry(tmp_path, scen_seeds=[7])
    assert reg.scen_path("tiny_4x4", 7) == (tmp_path / "scenarios" / "tiny_4x4-random-7.scen")


@pytest.mark.unit
def test_enumerate_instances_yields_cross_product(tmp_path: Path) -> None:
    reg = _populate_tiny_registry(tmp_path, scen_seeds=[1, 2])
    instances = list(reg.enumerate_instances(["tiny_4x4"], [2, 3], [1, 2]))
    # 1 map * 2 agent_counts * 2 scen_seeds = 4 instances.
    assert len(instances) == 4
    for map_name, n_agents, scen_seed, graph, agents in instances:
        assert map_name == "tiny_4x4"
        assert n_agents in (2, 3)
        assert scen_seed in (1, 2)
        assert graph.width == 4 and graph.height == 4
        assert len(agents) == n_agents


@pytest.mark.unit
def test_enumerate_instances_skips_missing_files_with_warning(tmp_path: Path) -> None:
    reg = _populate_tiny_registry(tmp_path, scen_seeds=[1])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        instances = list(reg.enumerate_instances(["tiny_4x4"], [2], [1, 99]))
    # seed=99 is missing; enumerator yields the seed=1 instance only.
    assert len(instances) == 1
    assert any("seed=99" in str(w.message) for w in captured)


@pytest.mark.unit
def test_load_instance_raises_file_not_found_with_hint(tmp_path: Path) -> None:
    reg = ScenarioRegistry(root=tmp_path)  # empty
    with pytest.raises(FileNotFoundError, match="fetch_benchmarks.sh"):
        reg.load_instance("nonexistent", n_agents=2, scen_seed=1)


@pytest.mark.unit
def test_default_root_is_repo_benchmarks_dir() -> None:
    # Smoke check: default-constructed registry points at <repo>/benchmarks.
    reg = ScenarioRegistry()
    assert reg.root.name == "benchmarks"
    assert (reg.root.parent / "pyproject.toml").exists()


@pytest.mark.unit
def test_map_families_table_lists_paper_maps() -> None:
    expected = {
        "warehouse-10-20-10-2-1",
        "warehouse-20-40-10-2-2",
        "random-32-32-20",
        "room-64-64-16",
        "maze-32-32-4",
    }
    assert set(ScenarioRegistry.MAP_FAMILIES) == expected
