"""Phase 8.1 figure-generation driver.

Reads ``results/raw/*.csv`` and writes a paper-ready PDF per
research-question plot to ``results/figures/``. In ``--smoke`` mode
the driver substitutes a small hand-crafted CSV fixture
(``tests/data/analysis_smoke/``) so the pipeline can be validated
without Phase 7 having run.

Run::

    python analysis/make_figures.py            # full grid
    python analysis/make_figures.py --smoke    # CI-friendly smoke
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

# Self-bootstrap: when the script is invoked as
# ``python analysis/make_figures.py``, the ``analysis`` package isn't
# on sys.path because hatch only ships ``src/slackcertify``. Inject
# the repo root so the analysis-internal imports resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402

from analysis.plots.ablation import generate_ablation_figures  # noqa: E402
from analysis.plots.close_analogues import generate_close_analogues_figure  # noqa: E402
from analysis.plots.headline import generate_headline_figure  # noqa: E402
from analysis.plots.optimality_gap import generate_optimality_gap_figure  # noqa: E402
from analysis.plots.pareto import generate_pareto_figure  # noqa: E402
from analysis.plots.persistent import generate_persistent_figure  # noqa: E402

__all__ = ["main"]


def _resolve_paths(smoke: bool) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    if smoke:
        return (
            repo_root / "tests" / "data" / "analysis_smoke",
            repo_root / "results" / "figures_smoke",
        )
    return (
        repo_root / "results" / "raw",
        repo_root / "results" / "figures",
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
        help="Override the raw-CSV directory (default depends on --smoke).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the figures output directory.",
    )
    parser.add_argument(
        "--only-figures",
        action="store_true",
        help="Skip the LaTeX table generation step.",
    )
    parser.add_argument(
        "--tables-output-dir",
        type=Path,
        default=None,
        help=(
            "Override the LaTeX summary output directory. Defaults to "
            "results/summary[_smoke] depending on --smoke."
        ),
    )
    args = parser.parse_args(argv)
    default_in, default_out = _resolve_paths(args.smoke)
    raw_dir: Path = args.input_dir or default_in
    fig_dir: Path = args.output_dir or default_out
    fig_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    console.log(f"[bold]raw csvs[/bold]: {raw_dir}")
    console.log(f"[bold]figures[/bold]: {fig_dir}")

    _run_step(
        console,
        "rq1_headline",
        lambda: generate_headline_figure(
            raw_dir / "rq1_headline.csv", fig_dir / "rq1_headline.pdf"
        ),
    )
    _run_step(
        console,
        "rq2_kr_pareto",
        lambda: generate_pareto_figure(
            raw_dir / "rq2_kr_pareto.csv", fig_dir / "rq2_kr_pareto.pdf"
        ),
    )
    _run_step(
        console,
        "rq3_close_analogues",
        lambda: generate_close_analogues_figure(
            raw_dir / "rq3_close_analogues.csv",
            fig_dir / "rq3_close_analogues.pdf",
        ),
    )
    _run_step(
        console,
        "rq4_ilp_gap",
        lambda: generate_optimality_gap_figure(
            raw_dir / "rq4_ilp_gap.csv", fig_dir / "rq4_ilp_gap.pdf"
        ),
    )
    _run_step(
        console,
        "rq5_persistent",
        lambda: generate_persistent_figure(
            raw_dir / "rq5_persistent.csv", fig_dir / "rq5_persistent.pdf"
        ),
    )
    _run_step(
        console,
        "ablations",
        lambda: generate_ablation_figures(
            raw_dir / "ablation_ordering.csv",
            raw_dir / "ablation_solver.csv",
            raw_dir / "ablation_budget.csv",
            fig_dir,
        ),
    )
    console.log("[bold green]all figures written[/bold green]")

    if args.only_figures:
        console.log("[yellow]--only-figures[/yellow]: skipping table generation")
        return 0

    # Chain into make_tables for the LaTeX deliverables. Argument
    # forwarding is explicit so this driver remains the single
    # entry point for "rebuild every paper artifact".
    from analysis.make_tables import main as _tables_main

    tables_argv: list[str] = []
    if args.smoke:
        tables_argv.append("--smoke")
    if args.input_dir is not None:
        tables_argv += ["--input-dir", str(args.input_dir)]
    if args.tables_output_dir is not None:
        tables_argv += ["--output-dir", str(args.tables_output_dir)]
    rc = _tables_main(tables_argv)
    return int(rc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
