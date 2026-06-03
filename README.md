# SlackCertify: Certifying Delay-Robust MAPF Execution by Proactive Wait Insertion

**SlackCertify: Certifying Delay-Robust MAPF Execution by Proactive Wait Insertion** This repository accompanies the
ASYU 2026 submission of the same name. It provides a reference implementation
of *slack certification*, a post-processing layer that takes any feasible
multi-agent path finding (MAPF) plan and inserts a minimal set of integer wait
actions so that the resulting schedule remains conflict-free under bounded or
probabilistic per-step execution delays. The library wraps open-source MAPF
solvers behind a common interface, exposes the certifier as a stand-alone
post-processor, and ships with a benchmark harness and reproducibility scripts
for the figures and tables in the paper.

## Key features

- **Solver-agnostic wrapper layer** with a unified plan / instance schema
  (`slackcertify.solvers`); the paper's experiments use EECBS and LaCAM*.
- **One-shot slack certifier** with both bounded-delay (worst-case `k`-step,
  Δ-certificate) and probabilistic-delay (per-step survival probability,
  ε-certificate) modes, implemented as an STN longest-path computation.
- **Conflict-freeness guarantees** under the certified delay model, with
  machine-checkable certificates emitted alongside every plan, and an explicit
  infeasibility certificate when no wait-only repair exists for the given order.
- **Benchmark harness** over the MovingAI `warehouse-10-20-10-2-1` and
  `random-32-32-20` maps and the delay perturbation suite used in the paper.
- **Typed Python API** (PEP 561), CLI, and YAML-based experiment configs.

## Installation

Install from source (Python 3.10–3.12):

```bash
git clone 
cd slack-certify-mapf
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz,ilp]"
```

The `viz` extra pulls in matplotlib/seaborn for figure generation; `ilp` pulls
in PuLP/HiGHS for the exact-optimum baseline used in the optimality-gap study.
A PyPI package and Zenodo DOI are planned for a future tagged release.

## Quick start

```python
from slackcertify.cli import pipeline

result = pipeline.run(
    map_path="benchmarks/maps/warehouse-10-20-10-2-1.map",
    scen_path="benchmarks/scenarios/warehouse-10-20-10-2-1-random-1.scen",
    n_agents=50,
    solver="eecbs",
    delay_model={"kind": "bounded", "k": 2},
    n_rollouts=1000,
    seed=20260514,
)
print(f"SoC={result.sum_of_costs} makespan={result.makespan} "
      f"success_rate={result.rollout_success_rate:.3f}")
```

## Running experiments

```bash
# Smoke check — validates the end-to-end pipeline on a small fixture.
bash scripts/repro_smoke.sh

# A single research question — e.g. the RQ1 bounded-delay sweep:
python -m experiments \
    --config experiments/configs/rq1.yaml \
    --output  results/rq1_bounded.csv
```

The per-experiment invocation is resumable: cells already recorded in the
output CSV are skipped on a re-run, so an interrupted run can be re-launched.

## Repository layout

- `src/slackcertify/` — Python package (solvers, certifier, simulator, CLI).
- `benchmarks/`        — MovingAI map and scenario files used in the paper.
- `experiments/`       — Experiment configs and runners for each research question.
- `analysis/`          — Figure- and table-generation code.
- `baselines/`         — Baseline solvers used for comparison.
- `tests/`             — Unit, integration, and property tests (pytest).
- `docs/`              — Documentation sources.
- `scripts/`           — Helper scripts (data download, reproduction).
- `results/`, `Results/` — Recorded experiment outputs (CSV / solver I/O).
- `third_party/`       — Vendored upstream solver sources, kept verbatim.

## Reproducibility

Experiments fix their random seeds, and MovingAI map/scenario files are bundled
or downloaded with checksums. See `docs/reproducibility.md` for the per-RQ
recipe (which CSV feeds which figure and table) and expected runtimes.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for the full text.
