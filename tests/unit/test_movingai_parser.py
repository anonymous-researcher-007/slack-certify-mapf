"""Unit tests for slackcertify.benchmarks.movingai."""

from __future__ import annotations

from pathlib import Path

import pytest

from slackcertify.benchmarks.movingai import (
    ScenRow,
    make_instance,
    parse_map_file,
    parse_scen_file,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MAP_FIXTURE = DATA_DIR / "tiny_4x4.map"
SCEN_FIXTURE = DATA_DIR / "tiny_4x4-random-1.scen"


@pytest.mark.unit
def test_parse_map_file_returns_correct_grid_graph() -> None:
    grid = parse_map_file(MAP_FIXTURE)
    assert grid.width == 4
    assert grid.height == 4
    # Hand-counted: 16 cells minus 2 obstacles = 14 free.
    assert len(grid) == 14
    assert grid.obstacles == frozenset({(1, 1), (2, 3)})


@pytest.mark.unit
def test_parse_scen_file_returns_expected_rows() -> None:
    rows = parse_scen_file(SCEN_FIXTURE)
    assert len(rows) == 3
    assert isinstance(rows[0], ScenRow)
    r0 = rows[0]
    assert r0.bucket == 0
    assert r0.map_name == "tiny_4x4"
    assert r0.map_width == 4 and r0.map_height == 4
    assert r0.start_x == 0 and r0.start_y == 0
    assert r0.goal_x == 3 and r0.goal_y == 3
    assert r0.optimal_length == pytest.approx(4.24)


@pytest.mark.unit
def test_parse_scen_file_skips_empty_lines(tmp_path: Path) -> None:
    text = SCEN_FIXTURE.read_text(encoding="utf-8").splitlines()
    # Splice in blank lines and a comment.
    munged = [
        text[0],
        "",
        "# this is a comment",
        text[1],
        "",
        text[2],
        text[3],
        "   ",
    ]
    tmp = tmp_path / "munged.scen"
    tmp.write_text("\n".join(munged) + "\n", encoding="utf-8")
    rows = parse_scen_file(tmp)
    assert len(rows) == 3


@pytest.mark.unit
def test_make_instance_returns_first_n_agents() -> None:
    grid, agents = make_instance(MAP_FIXTURE, SCEN_FIXTURE, n_agents=2)
    assert grid.width == 4
    assert [a.id for a in agents] == [0, 1]
    assert agents[0].start == (0, 0) and agents[0].goal == (3, 3)
    assert agents[1].start == (0, 2) and agents[1].goal == (3, 0)


@pytest.mark.unit
def test_make_instance_raises_when_insufficient_rows() -> None:
    with pytest.raises(ValueError, match="3 rows"):
        make_instance(MAP_FIXTURE, SCEN_FIXTURE, n_agents=4)


@pytest.mark.unit
def test_make_instance_raises_when_start_on_obstacle(tmp_path: Path) -> None:
    bad_scen = tmp_path / "bad.scen"
    bad_scen.write_text(
        "version 1\n" "0\ttiny_4x4\t4\t4\t1\t1\t0\t0\t1.41\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="obstacle"):
        make_instance(MAP_FIXTURE, bad_scen, n_agents=1)


@pytest.mark.unit
def test_parse_scen_file_rejects_malformed_row(tmp_path: Path) -> None:
    bad = tmp_path / "bad.scen"
    bad.write_text(
        "version 1\n" "0\ttiny_4x4\t4\t4\t0\t0\n",  # only 6 columns
        encoding="utf-8",
    )
    with pytest.raises(SyntaxError, match="9 tab-separated columns"):
        parse_scen_file(bad)
