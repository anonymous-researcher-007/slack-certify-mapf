# slack-certify-mapf

**Slack-Certified One-Shot MAPF: Proactive, Solver-Agnostic Wait Insertion for
Bounded and Probabilistic Delay Tolerance.** This repository accompanies the
ASYU 2026 submission of the same name. It provides a reference implementation
of *slack certification*, a post-processing layer that takes any feasible
multi-agent path finding (MAPF) plan and inserts a minimal set of wait actions
so that the resulting schedule remains conflict-free under bounded or
probabilistic per-step execution delays. The library wraps several open-source
MAPF solvers behind a common interface, exposes the certifier as a stand-alone
post-processor, and ships with a full benchmark harness and reproducibility
scripts for every figure and table in the paper.

[![CI](https://github.com/anonymized/slack-certify-mapf-test/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/anonymized/slack-certify-mapf-test/actions/workflows/ci.yml)
[![integration](https://github.com/anonymized/slack-certify-mapf-test/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/anonymized/slack-certify-mapf-test/actions/workflows/integration.yml)
[![smoke](https://github.com/anonymized/slack-certify-mapf-test/actions/workflows/smoke.yml/badge.svg?branch=main)](https://github.com/anonymized/slack-certify-mapf-test/actions/workflows/smoke.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://anonymized.github.io/slack-certify-mapf-test/)
![PyPI](https://img.shields.io/badge/pypi-pending%20release-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![DOI](https://img.shields.io/badge/DOI-pending%20release-lightgrey)

> PyPI package and Zenodo DOI will be published with the first
> tagged release; see [RELEASE.md](RELEASE.md) for the release
> checklist.

## Key features

- **Solver-agnostic wrapper layer** for CBS, EECBS, PBS, and LaCAM with a
  unified plan / instance schema (`slackcertify.solvers`).
- **One-shot slack certifier** with both bounded-delay (worst-case `k`-step)
  and probabilistic-delay (per-edge survival probability) certification modes.
- **Provable conflict-freeness guarantees** under the certified delay model,
  with machine-checkable certificates emitted alongside every plan.
- **Reproducible benchmark harness** covering the standard MovingAI MAPF map
  suite (warehouse, room, maze, game, random) and the delay perturbation suite
  introduced in the paper.
- **First-class tooling**: typed Python API (PEP 561), CLI, Hydra-based
  experiment configs, Docker images, and Jupyter notebook tutorials.

## Installation

### pip

```bash
pip install slack-certify-mapf
```

### conda

```bash
conda install -c conda-forge slack-certify-mapf
```

### docker

```bash
docker pull ghcr.io/anonymized/slack-certify-mapf:0.1.0
docker run --rm -it ghcr.io/anonymized/slack-certify-mapf:0.1.0 slackcertify --help
```

## Quick start

```python
from slackcertify.cli import pipeline

result = pipeline.run(
    map_path="benchmarks/maps/warehouse-10-20-10-2-1.map",
    scen_path="benchmarks/scen/warehouse-10-20-10-2-1-random-1.scen",
    n_agents=40,
    solver="eecbs",
    delay_model={"kind": "bounded", "k": 2},
    n_rollouts=1000,
    seed=20260514,
)
print(f"SoC={result.sum_of_costs} makespan={result.makespan} success_rate={result.rollout_success_rate:.3f}")
```

## Running experiments

Three entry points cover the spectrum from CI-friendly smoke to the
full §V sweep:

```bash
# Smoke — under 15 minutes on a stock laptop. Validates the full
# pipeline end-to-end against the bundled tiny_4x4 fixture and the
# analysis-smoke CSVs.
bash scripts/repro_smoke.sh

# Full §V grid — 24-72 hours on the reference hardware. Regenerates
# every paper figure, table, and CSV from scratch.
bash scripts/repro_paper.sh

# A single experiment — e.g. the RQ1 headline:
python -m experiments \
    --config experiments/configs/rq1_headline.yaml \
    --output  results/raw/rq1_headline.csv \
    --manifest results/manifests/rq1_headline.jsonl
```

The per-experiment invocation is resumable: every cell that's already
been recorded in the manifest is skipped on the second pass, so an
interrupted run can simply be re-launched. See
[`docs/reproducibility.md`](docs/reproducibility.md) for the full
per-RQ recipe (which CSV → which figure → which table).

## Repository layout

- `src/slackcertify/`  — Python package (solvers, certifier, simulator, CLI).
- `benchmarks/`        — MovingAI MAPF map and scenario files used in the paper.
- `experiments/`       — Hydra configs and launch scripts for every figure / table.
- `notebooks/`         — Tutorials and exploratory analysis notebooks.
- `tests/`             — Unit and integration tests (pytest).
- `docs/`              — Sphinx / mkdocs documentation sources.
- `scripts/`           — Stand-alone helper scripts (data download, plotting).
- `docker/`            — Dockerfiles and compose files for reproducible runs.
- `third_party/`       — Vendored upstream solver sources, kept verbatim.

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) for documented intentional
limitations (sandbox fallbacks, accepted coverage gaps, ruff
exceptions, one-time release prerequisites). If you encounter a
behavior matching an entry there, it is expected, not a bug.

## Reproducibility

Every figure and table in the paper is reproducible from a single command on a
single machine. Datasets are downloaded from the MovingAI repository with
SHA-256 checksums, every experiment fixes its random seeds, and the Docker
image pins the toolchain (Python, GCC, Boost, CMake) used in the camera-ready
results. See [`docs/reproducibility.md`](docs/reproducibility.md) for the full
recipe, including expected runtimes on the reference hardware and the
seed lists used for each rollout study.

## Citing this work

Once the paper and software DOIs are assigned (see RELEASE.md
for the release checklist), canonical BibTeX entries will be
provided here and in CITATION.cff. Pre-release reference: the
work is described in the manuscript submitted to ASYU 2026 and
the source code on this repository.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for the full text.
