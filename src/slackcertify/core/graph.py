"""Grid graph abstraction for one-shot multi-agent path finding.

The :class:`GridGraph` is the canonical environment representation used
throughout :mod:`slackcertify`. It supports 4- and 8-connected neighbourhoods,
common grid distance metrics, and round-tripping with the MovingAI ``.map``
benchmark format.

Examples
--------
>>> obstacles = {(1, 1)}
>>> g = GridGraph(width=3, height=3, obstacles=obstacles)
>>> sorted(g.neighbors((0, 0)))
[(0, 1), (1, 0)]
>>> (0, 0) in g
True
>>> (1, 1) in g
False
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

__all__ = ["Cell", "GridGraph"]

Cell = tuple[int, int]

#: Characters in a MovingAI ``.map`` file that denote *free* cells.
_FREE_CHARS: frozenset[str] = frozenset({".", "G"})

#: Characters in a MovingAI ``.map`` file that denote *blocked* cells.
_OBSTACLE_CHARS: frozenset[str] = frozenset({"@", "O", "T", "W"})


@dataclass(frozen=True, slots=True)
class GridGraph:
    """Rectangular grid graph with optional obstacles.

    Parameters
    ----------
    width
        Number of columns (``x`` extent), ``x ∈ [0, width)``.
    height
        Number of rows (``y`` extent), ``y ∈ [0, height)``.
    obstacles
        Set of ``(x, y)`` cells that are blocked.
    connectivity
        Either ``4`` (von Neumann) or ``8`` (Moore). Defaults to ``4``.

    Examples
    --------
    >>> g = GridGraph(width=2, height=2, obstacles=set(), connectivity=8)
    >>> sorted(g.neighbors((0, 0)))
    [(0, 1), (1, 0), (1, 1)]
    """

    width: int
    height: int
    obstacles: frozenset[Cell] = field(default_factory=frozenset)
    connectivity: int = 4

    _NEIGHBOUR_OFFSETS_4: ClassVar[tuple[Cell, ...]] = ((1, 0), (-1, 0), (0, 1), (0, -1))
    _NEIGHBOUR_OFFSETS_8: ClassVar[tuple[Cell, ...]] = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: set[Cell] | frozenset[Cell],
        connectivity: int = 4,
    ) -> None:
        """Validate dimensions and freeze the obstacle set onto this instance."""
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive, got {width}x{height}")
        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
        # `frozen=True` requires going through ``object.__setattr__``.
        object.__setattr__(self, "width", int(width))
        object.__setattr__(self, "height", int(height))
        object.__setattr__(self, "obstacles", frozenset(obstacles))
        object.__setattr__(self, "connectivity", int(connectivity))

    # ------------------------------------------------------------------ helpers

    def in_bounds(self, v: Cell) -> bool:
        """Return ``True`` if ``v`` lies within the grid extents.

        Examples
        --------
        >>> GridGraph(2, 2, set()).in_bounds((1, 1))
        True
        >>> GridGraph(2, 2, set()).in_bounds((2, 0))
        False
        """
        x, y = v
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, v: Cell) -> bool:
        """Return ``True`` if ``v`` is in-bounds and not an obstacle.

        Examples
        --------
        >>> g = GridGraph(2, 2, {(0, 0)})
        >>> g.is_free((0, 0))
        False
        >>> g.is_free((1, 1))
        True
        """
        return self.in_bounds(v) and v not in self.obstacles

    def neighbors(self, v: Cell) -> list[Cell]:
        """Return free neighbours of ``v`` according to ``connectivity``.

        Examples
        --------
        >>> g = GridGraph(3, 3, {(1, 1)})
        >>> sorted(g.neighbors((0, 0)))
        [(0, 1), (1, 0)]
        """
        offsets = self._NEIGHBOUR_OFFSETS_8 if self.connectivity == 8 else self._NEIGHBOUR_OFFSETS_4
        x, y = v
        out: list[Cell] = []
        for dx, dy in offsets:
            n = (x + dx, y + dy)
            if self.is_free(n):
                out.append(n)
        return out

    @staticmethod
    def manhattan(u: Cell, v: Cell) -> int:
        """L1 (Manhattan) distance between two cells.

        Examples
        --------
        >>> GridGraph.manhattan((0, 0), (3, 4))
        7
        """
        return abs(u[0] - v[0]) + abs(u[1] - v[1])

    @staticmethod
    def chebyshev(u: Cell, v: Cell) -> int:
        """L∞ (Chebyshev) distance between two cells.

        Examples
        --------
        >>> GridGraph.chebyshev((0, 0), (3, 4))
        4
        """
        return max(abs(u[0] - v[0]), abs(u[1] - v[1]))

    # --------------------------------------------------------------- protocols

    def __contains__(self, v: object) -> bool:
        """Return ``True`` iff ``v`` is a free, in-bounds cell tuple."""
        if (
            not isinstance(v, tuple)
            or len(v) != 2
            or not isinstance(v[0], int)
            or not isinstance(v[1], int)
        ):
            return False
        return self.is_free((v[0], v[1]))

    def __iter__(self) -> Iterator[Cell]:
        """Yield every free cell in row-major order."""
        for y in range(self.height):
            for x in range(self.width):
                cell = (x, y)
                if cell not in self.obstacles:
                    yield cell

    def __len__(self) -> int:
        """Return the number of free cells (total minus obstacles)."""
        return self.width * self.height - len(self.obstacles)

    # ------------------------------------------------------------- MovingAI IO

    @classmethod
    def from_movingai(cls, path: str | Path, connectivity: int = 4) -> GridGraph:
        """Load a MovingAI ``.map`` file.

        The MovingAI ``.map`` format is::

            type octile
            height H
            width W
            map
            <H rows of W chars each>

        Free characters are ``.`` and ``G``; obstacle characters are
        ``@``, ``O``, ``T``, and ``W``.

        Examples
        --------
        >>> import tempfile, pathlib
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "tiny.map"
        >>> _ = p.write_text("type octile\\nheight 2\\nwidth 2\\nmap\\n.@\\n..\\n")
        >>> g = GridGraph.from_movingai(p)
        >>> g.width, g.height, sorted(g.obstacles)
        (2, 2, [(1, 0)])
        """
        text = Path(path).read_text(encoding="utf-8").splitlines()
        if len(text) < 5 or text[0].strip() != "type octile":
            raise ValueError(f"{path}: not a MovingAI octile map (missing 'type octile' header)")
        try:
            height = int(text[1].split()[1])
            width = int(text[2].split()[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{path}: malformed height/width header") from exc
        if text[3].strip() != "map":
            raise ValueError(f"{path}: expected 'map' literal at line 4")

        rows = text[4 : 4 + height]
        if len(rows) != height:
            raise ValueError(
                f"{path}: header declares height={height} but file has {len(rows)} grid rows"
            )

        obstacles: set[Cell] = set()
        for y, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"{path}: row {y} has length {len(row)}, expected width={width}")
            for x, ch in enumerate(row):
                if ch in _OBSTACLE_CHARS:
                    obstacles.add((x, y))
                elif ch not in _FREE_CHARS:
                    raise ValueError(f"{path}: unrecognised cell {ch!r} at ({x}, {y})")
        return cls(width=width, height=height, obstacles=obstacles, connectivity=connectivity)

    # --------------------------------------------------------------- (de)serial

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the graph.

        Examples
        --------
        >>> g = GridGraph(2, 2, {(0, 0)})
        >>> g.to_dict() == {
        ...     "width": 2,
        ...     "height": 2,
        ...     "connectivity": 4,
        ...     "obstacles": [[0, 0]],
        ... }
        True
        """
        return {
            "width": self.width,
            "height": self.height,
            "connectivity": self.connectivity,
            "obstacles": sorted([list(c) for c in self.obstacles]),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GridGraph:
        """Reconstruct a :class:`GridGraph` from :meth:`to_dict` output.

        Examples
        --------
        >>> g1 = GridGraph(2, 2, {(0, 1)})
        >>> g2 = GridGraph.from_dict(g1.to_dict())
        >>> g1 == g2
        True
        """
        obstacles = {(int(c[0]), int(c[1])) for c in data.get("obstacles", [])}
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            obstacles=obstacles,
            connectivity=int(data.get("connectivity", 4)),
        )
