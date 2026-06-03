# Installation

`slack-certify-mapf` ships as a pure-Python package. The CSV / figure
/ table pipeline is self-contained; the §V baselines require external
C++ binaries that are built by `scripts/install_baselines.sh`.

## System requirements

| Tool        | Minimum version | Notes                                  |
|-------------|-----------------|----------------------------------------|
| Python      | 3.10            | Tested on 3.10, 3.11, 3.12             |
| pip         | (latest)        |                                        |
| git         | (any modern)    | For submodules in `third_party/`       |
| cmake       | 3.16            | For the baseline solver builds         |
| g++         | 11              | C++17 with concepts; some lacam variants need C++20 |
| Boost       | 1.74            | Required by EECBS                      |
| Eigen3      | 3.3             |                                        |
| Disk        | ~50 GB          | Full MovingAI benchmark suite          |

`python scripts/check_environment.py` walks the same checklist and
prints a green/red status table.

## From PyPI (after first release)

```bash
pip install slack-certify-mapf
```

The PyPI wheel includes the Python package only — no baselines, no
benchmarks. Use it for analysis-only consumers (loading the CSV/JSON
schema, reading a certificate, plotting an existing result CSV).

## From source

```bash
git clone --recurse-submodules \
  https://github.com/anonymized/slack-certify-mapf-test.git
cd slack-certify-mapf-test
pip install -e ".[dev,ilp]"
```

The `dev` extra adds pytest / mypy / ruff; the `ilp` extra adds
HiGHS / PuLP so the RQ4 optimality-gap runner can build its
wait-insertion ILP.

## With baseline solver binaries

After the source install:

```bash
bash scripts/install_baselines.sh
```

The script clones each upstream solver into `third_party/` (as a
submodule), runs CMake + make, and copies the resulting binaries
into `src/slackcertify/solvers/external_bin/`. See
[`third_party/README.md`](../third_party/README.md) for the pinned
commit hashes and per-solver build dependencies. Missing binaries
land Phase 7 cells at `status=binary_missing` rather than crashing
the run.

## With benchmark data

```bash
bash scripts/fetch_benchmarks.sh
```

Downloads the standard MovingAI MAPF map and scenario suite under
`benchmarks/` and verifies SHA-256 checksums. Idempotent.

## Verifying the install

```bash
python scripts/check_environment.py
bash scripts/repro_smoke.sh   # under 15 min on a laptop
```

`repro_smoke.sh` is the canonical end-to-end smoke: it runs the
integration smoke tests, drives the analysis pipeline against the
bundled fixture, and verifies every figure and table artifact.
