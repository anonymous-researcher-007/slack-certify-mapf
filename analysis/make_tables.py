"""Phase 8.2 LaTeX table-generation driver.

Reads ``results/raw/*.csv`` and writes ``results/summary/tab*.tex``.
``--smoke`` substitutes the bundled fixture under
``tests/data/analysis_smoke/`` so the pipeline can be validated
without Phase 7 having run.

Run::

    python analysis/make_tables.py            # full grid
    python analysis/make_tables.py --smoke    # CI-friendly smoke
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402

from analysis.tables.analogues_table import generate_analogues_table  # noqa: E402
from analysis.tables.optimality_gap_table import (  # noqa: E402
    generate_optimality_gap_table,
)
from analysis.tables.runtime_table import generate_runtime_table  # noqa: E402
from analysis.tables.status_breakdown_table import (  # noqa: E402
    generate_status_breakdown_table,
)

__all__ = ["main"]


# Phase 7 CSVs the breakdown table iterates over. Labels match the
# experiment names used in the paper's §V text.
_STATUS_BREAKDOWN_SOURCES: dict[str, str] = {
    "RQ1 headline": "rq1_headline.csv",
    "RQ2 kR Pareto": "rq2_kr_pareto.csv",
    "RQ3 close analogues": "rq3_close_analogues.csv",
    "RQ4 ILP gap": "rq4_ilp_gap.csv",
    "RQ5 persistent": "rq5_persistent.csv",
    "Ablation ordering": "ablation_ordering.csv",
    "Ablation solver": "ablation_solver.csv",
    "Ablation budget": "ablation_budget.csv",
}


def _resolve_paths(smoke: bool) -> tuple[Path, Path]:
    if smoke:
        return (
            _REPO_ROOT / "tests" / "data" / "analysis_smoke",
            _REPO_ROOT / "results" / "summary_smoke",
        )
    return (
        _REPO_ROOT / "results" / "raw",
        _REPO_ROOT / "results" / "summary",
    )


def _run_step(
    console: Console,
    name: str,
    fn: Callable[[], object],
) -> None:
    console.log(f"[cyan]{name}[/cyan]: starting")
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - per-step isolation
        console.log(f"[red]{name}[/red]: failed — {type(exc).__name__}: {exc}")
        raise
    console.log(f"[green]{name}[/green]: done")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the bundled tests/data/analysis_smoke/ CSV fixture.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override the raw-CSV directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the summary output directory.",
    )
    args = parser.parse_args(argv)
    default_in, default_out = _resolve_paths(args.smoke)
    raw_dir: Path = args.input_dir or default_in
    summary_dir: Path = args.output_dir or default_out
    summary_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    console.log(f"[bold]raw csvs[/bold]: {raw_dir}")
    console.log(f"[bold]summary[/bold]: {summary_dir}")

    _run_step(
        console,
        "tab1_runtime",
        lambda: generate_runtime_table(
            raw_dir / "rq1_headline.csv",
            raw_dir / "rq2_kr_pareto.csv",
            raw_dir / "rq3_close_analogues.csv",
            summary_dir / "tab1_runtime.tex",
        ),
    )
    _run_step(
        console,
        "tab2_analogues",
        lambda: generate_analogues_table(
            raw_dir / "rq3_close_analogues.csv",
            summary_dir / "tab2_analogues.tex",
        ),
    )
    _run_step(
        console,
        "tab3_optimality_gap",
        lambda: generate_optimality_gap_table(
            raw_dir / "rq4_ilp_gap.csv",
            summary_dir / "tab3_optimality_gap.tex",
        ),
    )
    _run_step(
        console,
        "tab4_status_breakdown",
        lambda: generate_status_breakdown_table(
            {label: raw_dir / fname for label, fname in _STATUS_BREAKDOWN_SOURCES.items()},
            summary_dir / "tab4_status_breakdown.tex",
        ),
    )
    console.log("[bold green]all tables written[/bold green]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
