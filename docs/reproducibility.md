# Reproducibility recipe

This page documents the exact commands that regenerate each §V
artifact, the CSV that feeds each figure / table, and the runtime
budget you should expect on the reference hardware.

## Reference hardware

| Component | Specification                              |
|-----------|--------------------------------------------|
| CPU       | 16-core x86_64 @ 3.5 GHz                   |
| RAM       | 64 GB                                      |
| OS        | Ubuntu 22.04 LTS                           |
| Python    | 3.11.x                                     |
| Disk      | NVMe SSD, ≥ 100 GB free                    |

Smaller machines work — the Phase 7 runners are append-only and
resumable — but the wall-clock estimates below are based on this
spec. Halve the agent counts in the per-RQ YAML configs for a
laptop-friendly mini-sweep.

## One-command rebuild

```bash
bash scripts/repro_paper.sh
```

Walks the entire pipeline: environment check → benchmark fetch →
baseline build → 8 Phase 7 runners → Phase 8 analysis → paper PDF.
Estimated total runtime: **24-72 hours**. Re-launchable on crash;
completed cells are skipped via the JSONL manifests under
`results/manifests/`.

## Per-experiment recipe

| RQ  | Config                                          | Runner                       | CSV                                         | Figure                            | Table              | Runtime (ref HW) |
|-----|-------------------------------------------------|------------------------------|---------------------------------------------|-----------------------------------|--------------------|------------------|
| RQ1 | `experiments/configs/rq1_headline.yaml`         | `RQ1HeadlineRunner`          | `results/raw/rq1_headline.csv`              | `rq1_headline.pdf`                | `tab1_runtime.tex` (col 1) | ~ 4 h            |
| RQ2 | `experiments/configs/rq2_kr_pareto.yaml`        | `RQ2KRParetoRunner`          | `results/raw/rq2_kr_pareto.csv`             | `rq2_kr_pareto.pdf`               | `tab1_runtime.tex` (kR cols)     | ~ 6 h            |
| RQ3 | `experiments/configs/rq3_close_analogues.yaml`  | `RQ3CloseAnaloguesRunner`    | `results/raw/rq3_close_analogues.csv`       | `rq3_close_analogues.pdf`         | `tab2_analogues.tex` | ~ 5 h            |
| RQ4 | `experiments/configs/rq4_ilp_gap.yaml`          | `RQ4ILPGapRunner`            | `results/raw/rq4_ilp_gap.csv`               | `rq4_ilp_gap.pdf`                 | `tab3_optimality_gap.tex` | ~ 8 h            |
| RQ5 | `experiments/configs/rq5_persistent.yaml`       | `RQ5PersistentRunner`        | `results/raw/rq5_persistent.csv`            | `rq5_persistent.pdf`              | — (see RQ1 wrt SoC)| ~ 5 h            |
| Abl. ordering | `experiments/configs/ablation_ordering.yaml` | `AblationOrderingRunner` | `results/raw/ablation_ordering.csv`         | `ablation_ordering.pdf`           | —                  | ~ 2 h            |
| Abl. solver   | `experiments/configs/ablation_solver.yaml`   | `AblationSolverRunner`   | `results/raw/ablation_solver.csv`           | `ablation_solver.pdf`             | —                  | ~ 3 h            |
| Abl. budget   | `experiments/configs/ablation_budget.yaml`   | `AblationBudgetRunner`   | `results/raw/ablation_budget.csv`           | `ablation_budget.pdf`             | —                  | ~ 2 h            |

To run a single RQ end-to-end:

```bash
python -m experiments \
    --config experiments/configs/rq1_headline.yaml \
    --output results/raw/rq1_headline.csv \
    --manifest results/manifests/rq1_headline.jsonl
python analysis/make_figures.py     # regenerates every figure + table from results/raw/*.csv
```

The figures-only path (`python analysis/make_figures.py --only-figures`)
is useful when iterating on plot aesthetics without rebuilding the
LaTeX tables.

## Per-figure / per-table recipe

| Output                                  | Generator                                                       | CSV input                                |
|-----------------------------------------|-----------------------------------------------------------------|------------------------------------------|
| `results/figures/rq1_headline.pdf`      | `analysis.plots.headline.generate_headline_figure`              | `rq1_headline.csv`                       |
| `results/figures/rq2_kr_pareto.pdf`     | `analysis.plots.pareto.generate_pareto_figure`                  | `rq2_kr_pareto.csv`                      |
| `results/figures/rq3_close_analogues.pdf` | `analysis.plots.close_analogues.generate_close_analogues_figure` | `rq3_close_analogues.csv`              |
| `results/figures/rq4_ilp_gap.pdf`       | `analysis.plots.optimality_gap.generate_optimality_gap_figure`  | `rq4_ilp_gap.csv`                        |
| `results/figures/rq5_persistent.pdf`    | `analysis.plots.persistent.generate_persistent_figure`          | `rq5_persistent.csv`                     |
| `results/figures/ablation_*.pdf` (×3)   | `analysis.plots.ablation.generate_ablation_figures`             | `ablation_{ordering,solver,budget}.csv`  |
| `results/summary/tab1_runtime.tex`      | `analysis.tables.runtime_table.generate_runtime_table`          | rq1 + rq2 + rq3                          |
| `results/summary/tab2_analogues.tex`    | `analysis.tables.analogues_table.generate_analogues_table`      | `rq3_close_analogues.csv`                |
| `results/summary/tab3_optimality_gap.tex` | `analysis.tables.optimality_gap_table.generate_optimality_gap_table` | `rq4_ilp_gap.csv`                  |
| `results/summary/tab4_status_breakdown.tex` | `analysis.tables.status_breakdown_table.generate_status_breakdown_table` | every CSV                          |

## What if I don't have all the binaries?

`scripts/install_baselines.sh` builds every baseline solver from
source. If a binary fails to build (e.g. a system header is missing),
the affected Phase 7 cell records `status="binary_missing"` rather
than crashing. The Phase 8 analysis pipeline drops binary-missing
rows from comparisons, so:

- **Figures** that depend on a missing baseline (e.g. the BTPG bar in
  the RQ3 grouped bar chart) display a shorter bar group rather than
  no figure at all.
- **Tables** show `--` in the cell corresponding to the missing
  baseline. The runtime table (`tab1_runtime.tex`) also degrades
  gracefully.

You'll see the dropped rows surfaced explicitly in
`tab4_status_breakdown.tex` (the honest-reporting table), so
reviewers can tell which methods were exercised against which
benchmarks.

If the upstream **Kottinger** binary is absent, the wrapper raises
`SolverNotFoundError` and the runner records
`status="binary_missing"` on the `rq3_close_analogues.csv` row.
There is no silent fallback — earlier versions of this codebase
shipped a Python reimpl that delegated to `slack_certify`, but the
audit's D3 resolution removed it because tagged `_reimpl` rows
looked like real Kottinger comparisons and were easy to forget to
filter. The `solver_used` column on a successful row is always
`<nominal>+kottinger_offline_delta=<delta>_binary`.
