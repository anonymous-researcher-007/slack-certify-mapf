"""Unit tests for slackcertify.core.graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from slackcertify.core.graph import GridGraph


@pytest.mark.unit
def test_constructor_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        GridGraph(0, 1, set())
    with pytest.raises(ValueError, match="connectivity"):
        GridGraph(2, 2, set(), connectivity=6)


@pytest.mark.unit
def test_neighbors_4_connected() -> None:
    g = GridGraph(4, 4, {(2, 2)})
    assert sorted(g.neighbors((1, 1))) == [(0, 1), (1, 0), (1, 2), (2, 1)]
    # (2, 2) is an obstacle so it is filtered out of neighbours of (1, 2).
    assert (2, 2) not in g.neighbors((1, 2))


@pytest.mark.unit
def test_neighbors_8_connected() -> None:
    g = GridGraph(3, 3, set(), connectivity=8)
    assert len(g.neighbors((1, 1))) == 8


@pytest.mark.unit
def test_distance_metrics() -> None:
    assert GridGraph.manhattan((0, 0), (3, 4)) == 7
    assert GridGraph.chebyshev((0, 0), (3, 4)) == 4


@pytest.mark.unit
def test_contains_and_iter() -> None:
    g = GridGraph(2, 2, {(1, 1)})
    assert (0, 0) in g
    assert (1, 1) not in g
    assert "not a tuple" not in g  # type: ignore[operator]
    free_cells = list(g)
    assert len(free_cells) == 3
    assert (1, 1) not in free_cells


@pytest.mark.unit
def test_movingai_round_trip(tmp_path: Path) -> None:
    src = "type octile\nheight 3\nwidth 3\nmap\n.@.\n...\n@..\n"
    p = tmp_path / "tiny.map"
    p.write_text(src, encoding="utf-8")

    g = GridGraph.from_movingai(p)
    assert g.width == 3
    assert g.height == 3
    assert g.obstacles == frozenset({(1, 0), (0, 2)})

    # to_dict / from_dict round-trip preserves equality.
    g2 = GridGraph.from_dict(g.to_dict())
    assert g == g2


@pytest.mark.unit
def test_movingai_rejects_bad_header(tmp_path: Path) -> None:
    p = tmp_path / "bad.map"
    p.write_text("not a real header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="octile"):
        GridGraph.from_movingai(p)


@pytest.mark.unit
def test_movingai_rejects_unknown_char(tmp_path: Path) -> None:
    src = "type octile\nheight 1\nwidth 2\nmap\n.X\n"
    p = tmp_path / "weird.map"
    p.write_text(src, encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognised"):
        GridGraph.from_movingai(p)
