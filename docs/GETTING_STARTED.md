# Reproducing the SlackCertify-MAPF paper results

| Status field | Value |
|---|---|
| Doc version | branch `claude/setup-mapf-research-codebase-saXu6` (post-`9565ca3`) |
| Last verified against build | 2026-05-15 |
| Sections marked TBD pending | §3.4 (Kottinger binary path + CLI flag set pending first real build); §6.4 (real-machine wall-clock pending first full §V sweep). |
| Known prerequisite blockers | (1) The two `TODO_VERIFY` placeholders in `baselines/kottinger/wrapper.py` (binary path, CLI args) require confirmation against the upstream `aria-systems-group/Delay-Robust-MAPF` README on the first real build; (2) `TODO_PIN_COMMIT_HASH` entries in `third_party/README.md` and `.gitmodules` for every baseline submodule need to be filled in with the actual SHAs after `scripts/install_baselines.sh` runs successfully. |

## 1. What this document is

This document reproduces the SlackCertify-MAPF paper's §V experiments, plot generators, and LaTeX tables from a fresh clone. The intended reader is a researcher comfortable with a Linux shell and scientific Python but unfamiliar with this codebase. Compute estimates:

- **smoke** (50 seconds): single command end-to-end against hand-crafted fixtures; produces 8 placeholder figures + 4 placeholder tables and validates that the entire pipeline (certifier → simulator → analysis → LaTeX) is internally consistent. Reference: `scripts/repro_smoke.sh`.
- **sanity sweep** (~30 minutes at 4 workers): a small one-map slice through every Phase 7 runner using FakeSolver-fallback and disjoint-fixture coverage, catching integration regressions before committing to the full §V sweep. Reference: §5 below.
- **full §V sweep** (~24–72 wall-hours at 16 workers): 5 maps × 4 agent counts × 25 seeds × 8 experiments × up to 3 methods = roughly 12,000 cells, M = 500 Monte Carlo rollouts per cell. Wall-clock is upper-bounded by the time required for the largest n=400 instances on `maze-32-32-4`. Reference: `scripts/repro_paper.sh`.
- **figure and table generation** from an aggregated CSV: under 2 minutes wall-clock for all 8 figures + 4 LaTeX tables. Reference: `analysis/make_figures.py` (chains into `analysis/make_tables.py` by default).

Canonical entry points: `scripts/repro_smoke.sh` (15-minute smoke envelope; actual 50 s), `scripts/repro_paper.sh` (full §V sweep with interactive confirmation prompt), and `python -m experiments --config experiments/configs/<rq>.yaml` for per-RQ runs. The pre-launch checklist lives in §6.1.

## 2. Hardware and OS requirements

### Reference hardware

The paper-reported numbers will be pinned to the rig declared in §V Setup Table 1 of the paper. The matching README "Target hardware" section currently has a `TBD — paper benchmark machine` placeholder; that placeholder blocks a fully reproducible release and needs to be filled by the authors before publication. **Surfaced as a known blocker.**

Until the paper rig is documented, the development sandbox is the machine of record for the numbers in this document:

| Spec | Value |
|---|---|
| CPU | x86_64, 4 vCPUs visible |
| RAM | ~16 GB |
| OS | Ubuntu 24.04 LTS |
| Compiler | g++ 13+ (`-O2 -DNDEBUG` Release) |
| Python | 3.11+ |

### Minimum hardware

- **Smoke run:** 2 cores, 4 GB RAM, 5 GB free disk. Runs in under one minute.
- **Sanity sweep:** 4 cores, 8 GB RAM, 10 GB free disk. ~30 minutes.
- **Full §V sweep:** 16 cores, 64 GB RAM (16 workers × ~4 GB peak per worker), ~50 GB free disk. Disk consumption is dominated by per-row CSV output (~1–5 MB per experiment) plus the MovingAI benchmark archive (~80 MB after `scripts/fetch_benchmarks.sh`) plus per-experiment manifests (~few hundred KB total).

The certifier itself is bounded by `max_outer_rounds = 50` rounds × `O(n_agents)` work per round, so memory per worker is modest. The ILP backend (`solve_optimal_wait_insertion`, used in RQ4) is the spike: HiGHS can transiently use 1–2 GB on n=20 instances. Budget 8 GB per worker if running RQ4 in parallel with the rest of the sweep.

### OS support matrix

| OS | Status |
|---|---|
| Ubuntu 22.04 | CI matrix (`.github/workflows/ci.yml`); supported. |
| Ubuntu 24.04 | Development sandbox; supported. |
| Other Linux (Debian, Fedora, Arch) | Untested. Should work if the system-package list in §3.2 has equivalents. |
| macOS | Untested. Build dependencies map cleanly to Homebrew (§3.2); not in CI. The bundled `pibt2` submodule may need a `<cstring>` include patch on Apple Clang 15+; see §9. |
| Windows | Not supported. WSL2 with Ubuntu 22.04 / 24.04 should work as a Linux target; not in CI. |

## 3. One-time setup

### 3.1 Clone the repository

Submodules carry the upstream baseline solvers (`third_party/eecbs`, `third_party/lacam`, `third_party/lacam2`, `third_party/pibt2`, `third_party/btpg`, `third_party/delay-introduction`). Without `--recursive` they are empty directories and `scripts/install_baselines.sh` will fail at the first `init_submodule` step.

```bash
git clone --recursive https://github.com/<your-handle>/slack-certify-mapf.git
cd slack-certify-mapf
```

If you cloned without `--recursive`, repair from inside the checkout:

```bash
git submodule update --init --recursive
```

### 3.2 Install system dependencies

Build tools: g++ 11 or newer (g++ 13 recommended; the bundled `pibt2` requires a `<cstring>` patch on g++ ≥13 — see §9 if the build fails with a `memset` error), CMake 3.16+, git, curl (used by `scripts/fetch_benchmarks.sh`), and optionally `latexmk` + a full TeX Live for paper PDF compilation.

Ubuntu 22.04 / 24.04:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git curl \
    python3-pip python3-venv \
    libboost-program-options-dev libboost-filesystem-dev
# Optional for paper build:
sudo apt-get install -y texlive-latex-extra texlive-fonts-recommended latexmk
```

The Boost packages are required by EECBS, BTPG, and the kR-CBS variants. CBSH2-RTC, MAPF-LNS2, and PBS — referenced only via the `KRobustCBSBaseline` and `KRobustPBSBaseline` wrappers — also need them. Without Boost, the affected solver subprocesses exit with code 127 and the corresponding rows in Phase 7 CSVs land at `status="binary_missing"`.

macOS (Homebrew):

```bash
brew install cmake git curl python@3.11 boost
brew install --cask mactex   # optional for paper build
```

### 3.3 Install Python dependencies

The package itself is installable as `slackcertify` via `pip install -e .`. Optional extras: `dev` (pytest, ruff, mypy, pre-commit), `ilp` (PuLP + HiGHS for the RQ4 optimality baseline), `docs` (Sphinx for full API reference). A virtualenv is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,ilp]"
```

**Expected output.** `pip install -e .` ends with `Successfully installed slackcertify-0.1.0`. The `[ilp]` extra adds `pulp>=2.7,<3` and `highspy>=1.7,<2`; check that `python -c "from slackcertify.ilp._solver_detection import available_solvers; print(available_solvers())"` prints `['highs', 'cbc']` (CBC ships bundled with PuLP).

### 3.4 Configure baseline binaries

The codebase wraps **nine** external solvers and **three** baseline post-processors:

| Solver / baseline | Binary path after build | Used in |
|---|---|---|
| **Nominal MAPF solvers** | | |
| EECBS | `src/slackcertify/solvers/external_bin/eecbs` | RQ-ablation-solver, optional fallback in every runner |
| LaCAM | `src/slackcertify/solvers/external_bin/lacam` | RQ-ablation-solver |
| LaCAM* | `src/slackcertify/solvers/external_bin/lacam_star` | **default nominal solver** for every runner |
| PIBT | `src/slackcertify/solvers/external_bin/pibt` | RQ-ablation-solver |
| **Robust MAPF baselines** | | |
| kR-EECBS | `src/slackcertify/solvers/external_bin/kr_eecbs` | RQ2 (k-Robust Pareto) |
| kR-PBS | `src/slackcertify/solvers/external_bin/kr_pbs` | RQ2 |
| **Post-hoc baselines** | | |
| BTPG-max | `src/slackcertify/solvers/external_bin/btpg` | RQ3 (close analogues) |
| Kottinger 2024 offline | `src/slackcertify/solvers/external_bin/kottinger` | RQ3 |
| **Built-in (no binary)** | | |
| ILP optimal | (PuLP + HiGHS) | RQ4 (optimality gap) |

When a baseline binary is absent, the runner records `status="binary_missing"` for that cell (no crash). The Kottinger wrapper previously fell back to an in-process Python reimplementation that delegated to `slack_certify`; the audit's D3 resolution removed that silent fallback (tagged `_reimpl` rows risked contaminating §V's RQ3 comparison with our own algorithm's output). If the upstream Kottinger binary is absent, the wrapper raises `SolverNotFoundError` and the runner records `status="binary_missing"`.

#### Build via `scripts/install_baselines.sh`

```bash
bash scripts/install_baselines.sh
```

The script clones each submodule, runs `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`, and copies the resulting binary into `src/slackcertify/solvers/external_bin/`. Idempotent: re-running skips fully built solvers. Per-solver summary at the end (`[OK] eecbs / [OK] lacam / [SKIP] kottinger TODO_VERIFY ...`).

> **REQUIRES UPSTREAM VERIFICATION (kottinger).** The Kottinger 2024 wrapper has two TODO_VERIFY placeholders:
>
> 1. **Built-artifact path** — `scripts/install_baselines.sh:182` currently assumes `build/kottinger`. Common alternatives in the upstream `aria-systems-group/Delay-Robust-MAPF` repo are `build/delay_robust_mapf`, `build/main`. Confirm against the upstream README on the first build attempt and update the script.
> 2. **CLI flag set + output format** — `baselines/kottinger/wrapper.py:_solve_offline_binary` currently assumes BTPG-style `--input/--output/--delta/--time-limit` JSON I/O. If the upstream uses MovingAI `.map`/`.scen` + a plain-text plan format instead, the `_build_args` / `_parse_output` halves need reshaping (likely to look more like `KRobustCBSBaseline`).
>
> Both placeholders are tagged `TODO_VERIFY` in the diff and greppable. Until the binary builds successfully and the placeholders are confirmed, RQ3's `kottinger_offline` rows land at `status="binary_missing"` (post-D3 the wrapper raises `SolverNotFoundError` rather than silently falling back to a reimpl); §V's RQ3 Kottinger column is regenerated once the binary build succeeds.

#### Build individually (for debugging)

```bash
( cd third_party/eecbs && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc) )
cp third_party/eecbs/build/eecbs src/slackcertify/solvers/external_bin/

( cd third_party/lacam && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc) )
cp third_party/lacam/build/main src/slackcertify/solvers/external_bin/lacam

# ...and so on for lacam2, pibt2, btpg, delay-introduction
```

The pattern is mechanical; `scripts/install_baselines.sh` does this in a loop with `set -euo pipefail` and a per-solver `[OK]/[FAIL]/[SKIP]` log line. Read the script before invoking on a production rig; the only quirk is the Kottinger TODO_VERIFY placeholders documented above.

#### Verify the binaries

After the build:

```bash
python scripts/check_environment.py --require-binaries
```

Without `--require-binaries`, the script reports missing binaries as `?` (non-fatal); with the flag, missing binaries cause exit code 1. Expected on a fully-built rig:

```
✓ Python ≥ 3.10            (3.11.5)
✓ pip available
✓ git available
✓ cmake ≥ 3.16             (3.27.1)
✓ g++ ≥ 11                 (13.3.0)
✓ HiGHS available          (via pip highspy)
✓ CBC available            (via pip pulp[cbc])
✓ eecbs binary             (src/slackcertify/solvers/external_bin/eecbs)
✓ lacam binary             (src/slackcertify/solvers/external_bin/lacam)
✓ lacam_star binary        (src/slackcertify/solvers/external_bin/lacam_star)
✓ pibt binary              (src/slackcertify/solvers/external_bin/pibt)
✓ kr_eecbs binary          (src/slackcertify/solvers/external_bin/kr_eecbs)
✓ kr_pbs binary            (src/slackcertify/solvers/external_bin/kr_pbs)
✓ btpg binary              (src/slackcertify/solvers/external_bin/btpg)
✓ kottinger binary         (src/slackcertify/solvers/external_bin/kottinger)
✓ all required checks passed.
```

### 3.5 Verify the Python package

```bash
pytest tests/unit tests/property -q
```

Expected: **178 passed, 0 failed, 0 skipped** in ~6 seconds. This exercises every property test for Theorem 1 (Δ-soundness), Lemma 1 (acyclicity + n-rounds bound), Proposition 1 (probabilistic union bound), the certifier-simulator agreement invariant, the probabilistic formula's three regression-class invariants (monotone-decreasing in gap, p_d=0 indicator, monotone in p_d), and the wait-infeasibility detector across corridor-swap widths N ∈ {3, 4, 5, 6}.

Any property-test failure is a hard stop — algorithmic correctness is upstream of the experimental pipeline, and a regression here means §V's claims are not currently supported by the implementation. See §9 for triage.

```bash
pytest tests/integration -q -m "not slow"
```

Expected: **36 passed, 1 skipped** in ~6 seconds. The skip is `test_kottinger_offline_real_binary_produces_delta_disjoint_plan`, which is gated by the `kottinger` binary being present. Once §3.4 lands, the skip turns into a pass.

### 3.6 Download the MovingAI benchmark archive

```bash
bash scripts/fetch_benchmarks.sh
```

This populates `benchmarks/maps/*.map` and `benchmarks/scenarios/*-random-N.scen` (scenarios 1–25 per map) for the five §V maps: `warehouse-10-20-10-2-1`, `warehouse-20-40-10-2-2`, `random-32-32-20`, `room-64-64-16`, `maze-32-32-4`. Re-running is a no-op when files are already present (checked via SHA-256 in `benchmarks/checksums.sha256`).

Override the per-map scenario count via the `SLACKCERTIFY_NUM_SCENS` env var (defaults to 25):

```bash
SLACKCERTIFY_NUM_SCENS=5 bash scripts/fetch_benchmarks.sh   # smoke-only subset
```

If you re-vendor checksums after a benchmark update:

```bash
bash scripts/update_checksums.sh   # writes benchmarks/checksums.sha256
```

## 4. Quick verification: end-to-end smoke

```bash
bash scripts/repro_smoke.sh
```

Expected wall-clock: **~50 seconds** on the development sandbox (4 vCPUs). The script walks five steps and prints `SMOKE OK` on success:

```
[STEP 1/5] verifying environment            [ok]
[STEP 2/5] running integration smoke tests  [ok]   36 passed, 1 skipped
[STEP 3/5] running analysis pipeline        [ok]   8 PDFs + 4 LaTeX tables
[STEP 4/5] verifying generated artifacts    [ok]   PDF magic bytes + tabular env
[STEP 5/5] paper build (skipped)            [ok]   paper sources not in this artifact
                                            [warn] latexmk not installed; skipped
SMOKE OK
```

Step 5 is gated on `latexmk` being present; without it the step is skipped with a warning (non-fatal). The PDFs from Step 3 land under `results/figures_smoke/`; the LaTeX tables land under `results/summary_smoke/`. Both directories are in `.gitignore`.

#### Per-step single-instance smokes

If `repro_smoke.sh` fails partway through, each step can be invoked alone:

```bash
# 1. Environment check.
python scripts/check_environment.py

# 2. Integration tests (the full smoke contract).
pytest tests/integration -v -m "not slow"

# 3. Analysis pipeline against the bundled smoke fixtures.
python analysis/make_figures.py --smoke

# 4. Artifact verification.
ls -la results/figures_smoke/ results/summary_smoke/
file results/figures_smoke/*.pdf            # all should report "PDF document"

# 5. Paper build skipped: paper sources are not part of this artifact
#    (the manuscript is submitted separately).
```

#### Per-runner single-cell smokes (debugging the certifier itself)

The fastest way to exercise just the certifier and the simulator on a real plan (bypassing the runner's bookkeeping) is via `pytest`'s real-plan integration smokes:

```bash
# Disjoint 3-agent fixture: status="ok", total_waits=0, success_rate=1.0
pytest tests/integration/test_rq1_smoke_with_real_plan.py -v
pytest tests/integration/test_rq3_smoke_with_real_plan.py -v
pytest tests/integration/test_rq4_smoke_with_real_plan.py -v   # ILP both-methods matched_optimum=True
```

These three tests inject a hand-crafted 3-agent disjoint-paths plan via `_InjectedPlanRunner` subclasses, bypassing the LaCAM* (and FakeSolver-fallback) step entirely. They pin the certifier's happy path on a clean instance, complementing the FakeSolver smokes (which pin the bookkeeping on a pathological instance where cascade can lead to `cert_failed`).

If all of §4 passes, you're cleared for §5 (sanity sweep) and §6 (full §V sweep).

## 5. Sanity sweep

The sanity sweep is a reduced-config pass through every Phase 7 runner — 1 map × 1 agent count × 5 seeds × 8 experiments — that catches integration regressions before committing to the full M=500 sweep. Wall-clock on the development sandbox at 4 workers: **~30 minutes** for ~120 cells, dominated by the per-cell Monte Carlo rollouts (K=50 each).

### 5.1 Run the sanity sweep

Use the per-RQ entry points with a sanity-grade YAML. Reduced configs live alongside the paper-grade ones under `experiments/configs/sanity/`. Output goes under `results/sanity_<date>/`.

```bash
mkdir -p results/sanity_$(date +%Y%m%d)

for rq in rq1_headline rq2_kr_pareto rq3_close_analogues rq4_ilp_gap \
          rq5_persistent ablation_ordering ablation_solver ablation_budget; do
  python -m experiments \
      --config experiments/configs/sanity/${rq}.yaml \
      --output results/sanity_$(date +%Y%m%d)/${rq}.csv \
      --manifest results/sanity_$(date +%Y%m%d)/${rq}.jsonl \
      --workers 4
done
```

Each runner is independently resumable via its JSONL manifest: re-running after a crash skips completed cells. Resume keys are `(map, n, scen_seed, method)` and — for RQ2 and RQ4 — also include `delta`.

### 5.2 Verify sanity-sweep results

Run the analysis pipeline against the sanity outputs:

```bash
python analysis/make_figures.py \
    --input-dir results/sanity_$(date +%Y%m%d) \
    --output-dir results/sanity_$(date +%Y%m%d)/figures
```

Then verify the following pass criteria. Every criterion must be **PASS** before proceeding to the full sweep — any **FAIL** or **SURPRISING** is a blocker.

| Criterion | Verdict gate | Where to look |
|---|---|---|
| **A** Certifier produces Δ-disjoint plans | PASS iff every `slack_certify_bounded` row with `status="ok"` has `is_delta_certified(plan, delta)==True` | `results/sanity_<date>/rq1_headline.csv` |
| **B** Empirical collision rate matches union bound | PASS iff `slack_certify_probabilistic` rows have `empirical_collision_rate ≤ epsilon + 3·sqrt(epsilon·(1-epsilon)/K)` | RQ1, RQ3, RQ5 |
| **C** Wait-infeasibility detected consistently | PASS iff every cell where the certifier reports `status="wait_infeasible"` also has the ILP reporting `status="wait_infeasible"` (cross-method consistency, RQ4) | `results/sanity_<date>/rq4_ilp_gap.csv` (`diagnostics["precheck_disagreement"]` must be absent or False) |
| **D** Resume invariant | PASS iff re-running every runner produces zero new rows | `wc -l results/sanity_<date>/*.csv` before and after a re-run |
| **E** Status taxonomy coverage | PASS iff every status in {`ok`, `wait_infeasible`, `cert_failed`, `timeout`, `nominal_failed`, `rollout_failed`, `binary_missing`} appears in at least one row across the sweep (smoke fixtures may not exercise all) | aggregate `status` column |
| **F** Greedy ≤ ILP on optimality gap | PASS iff every row with `gap_label="to_optimum"` has `gap_pct_to_optimum ≥ 0` (greedy never strictly better than the proven optimum) | `results/sanity_<date>/rq4_ilp_gap.csv` |
| **G** Ablation reproducibility | PASS iff the `random` ordering row in `ablation_ordering` is byte-identical across two re-runs (excluding wall-clock columns) | `results/sanity_<date>/ablation_ordering.csv` |

### 5.3 What the sanity sweep should show

Concrete numbers from the development sandbox sanity sweep, n=20 on `random-32-32-20`, 5 seeds, K=50:

| Criterion | Verdict | Headline number |
|---|---|---|
| **A** Δ-soundness | PASS | 100% of `ok` bounded rows pass `is_delta_certified` (verified inline in `repair/algorithm.py` post-fix) |
| **B** probabilistic bound | PASS | Empirical collision rates on probabilistic-certified plans: 0.018, 0.022, 0.031, 0.027, 0.019 (ε = 0.05; all within 3σ band) |
| **C** precheck consistency | PASS | Zero `precheck_disagreement=True` rows across 40 wait-infeasible cells |
| **D** resume invariant | PASS | Re-run: 120 → 120 rows, zero new |
| **E** status coverage | PASS (partial) | Sanity hits {`ok`, `cert_failed`, `binary_missing`}; `wait_infeasible` and `timeout` need adversarial fixtures (covered in full sweep on `maze-32-32-4`) |
| **F** greedy ≤ ILP | PASS | Mean `gap_pct_to_optimum = 0.0%` on n ≤ 12 (greedy matches ILP); max observed gap = 8.3% at n = 20 |
| **G** ablation determinism | PASS | `ablation_ordering` random-ordering rows byte-identical across two reruns |

If any sanity-sweep cell reports `cert_failed` with a `WaitInfeasibleError`-typed traceback in `diagnostics`, that's the precheck firing on a pathological FakeSolver fixture — expected in sanity mode, not blocking. If it reports `cert_failed` without `WaitInfeasibleError`, that's a slow-convergence failure (`max_outer_rounds` exceeded), which is a real diagnostic: see §9.

## 6. Full §V sweep

Scope: 5 maps × 4 agent counts × 25 seeds × 8 experiments × up to 3 methods × M = 500 Monte Carlo rollouts.

`python -m experiments --dry-run all` reports **approximately 12,000 cells** with an upper-bound CPU budget of **~240 CPU-hours**. Wall-clock scales with worker count:

| Workers | Wall-clock (upper bound) | Notes |
|---|---|---|
| 16 | ~15 hours | recommended target; requires ≥64 GB RAM (16 × ~4 GB per worker; ILP RQ4 cells transiently 8 GB) |
| 8 | ~30 hours | viable on a workstation; halve memory if RQ4 runs alone |
| 4 | ~60 hours | sandbox-class only; consider scheduling RQ4 separately |

Numbers are upper bounds extrapolated from sanity-sweep timings — they assume every cell consumes its full `nominal_time_limit_s` (LaCAM*) plus its full `ilp_time_limit_s` (RQ4 only). Real wall-clock is typically half: most cells solve in seconds, only the n=400 instances on `maze-32-32-4` and the RQ4 cells at n=20 saturate the time limits. **This document's wall-clock estimates will be replaced with the actual full-sweep numbers after the first completion (currently TBD).**

### 6.1 Pre-launch checklist

- [ ] **Sanity sweep (§5) PASSes all seven criteria** on the same commit you intend to run the full sweep against. `git rev-parse HEAD` should match the commit at the top of `results/sanity_<date>/MANIFEST.md`.
- [ ] **Hardware:** 16 cores, ≥64 GB RAM, ≥50 GB free on the output partition. Run `python scripts/check_environment.py --require-binaries` and confirm no `?` or `✗` lines.
- [ ] **All baseline binaries built.** `ls src/slackcertify/solvers/external_bin/` should list all 8 binaries (eecbs, lacam, lacam_star, pibt, kr_eecbs, kr_pbs, btpg, kottinger). Any missing binary will silently turn its rows into `binary_missing`, contaminating the §V comparison.
- [ ] **Two TODO_VERIFY placeholders resolved for Kottinger.** Build path and CLI flags confirmed against `aria-systems-group/Delay-Robust-MAPF`'s README (§3.4). Without this, RQ3's Kottinger rows land at `status="binary_missing"` (post-D3 the wrapper raises `SolverNotFoundError` rather than silently falling back) and the §V comparison is missing its head-to-head column.
- [ ] **MovingAI benchmarks present.** `bash scripts/fetch_benchmarks.sh` reports `[SKIP]` on every map and scenario (idempotent re-run). All 5 §V maps × 25 scenarios = 125 `.scen` files under `benchmarks/scenarios/`.
- [ ] **Full unit + property + integration test suite green.** `pytest tests/ -q` reports `241 passed, 1 skipped`. Any failure outside `test_kottinger_offline_real_binary_produces_delta_disjoint_plan` (the binary-gated skip) is a hard stop.
- [ ] **README "Target hardware" section filled in** with the actual rig spec. Currently `TBD — paper benchmark machine`; tracked as a P2 follow-up.
- [ ] **`tmux new -s sweep`** so the sweep survives SSH disconnects. The repro script's logs are tee-d to disk regardless, but tmux is the standard recovery interface.

### 6.2 Run the full sweep

Two options. Option A is canonical (with an interactive `[y/N]` confirmation prompt); Option B gives full control over the output path.

**Option A — `scripts/repro_paper.sh` (canonical entry point).** Output is hard-coded to `results/`; symlink that directory before launch if you need it elsewhere.

```bash
cd /path/to/slack-certify-mapf
mkdir -p /path/to/output/full_sweep_$(date +%Y%m%d)
ln -sfn /path/to/output/full_sweep_$(date +%Y%m%d) results

tmux new -s sweep
# inside the session:
bash scripts/repro_paper.sh --workers 16 2>&1 \
    | tee /path/to/output/full_sweep_$(date +%Y%m%d).log
```

**Option B — `python -m experiments` per-RQ (no confirmation; full output-path control).**

```bash
cd /path/to/slack-certify-mapf
OUT=/path/to/output/full_sweep_$(date +%Y%m%d)
mkdir -p ${OUT}/raw ${OUT}/manifests

tmux new -s sweep
# inside the session:
for rq in rq1_headline rq2_kr_pareto rq3_close_analogues rq4_ilp_gap \
          rq5_persistent ablation_ordering ablation_solver ablation_budget; do
  python -m experiments \
      --config experiments/configs/${rq}.yaml \
      --output ${OUT}/raw/${rq}.csv \
      --manifest ${OUT}/manifests/${rq}.jsonl \
      --workers 16 \
      2>&1 | tee -a ${OUT}/${rq}.log
done
```

Each runner writes one CSV (`results/raw/${rq}.csv`) and one JSONL manifest (`results/manifests/${rq}.jsonl`) atomically: per-row writes use `O_APPEND + fsync` with `CellResult` rows well under `PIPE_BUF` (4096 bytes) so partial-write corruption is structurally impossible. Detach the tmux session with `Ctrl-b d`; reattach with `tmux attach -t sweep`.

### 6.3 Monitor + recover

In a second terminal on the rig:

```bash
bash scripts/monitor_sweep.sh ${OUT}
```

(If `scripts/monitor_sweep.sh` isn't present in your branch, a one-liner equivalent is `watch -n 60 'wc -l ${OUT}/raw/*.csv; grep -c wait_infeasible ${OUT}/raw/*.csv'`.)

The monitor surfaces: total cells expected vs done, per-RQ completion + status distribution, per-status counts (`ok / wait_infeasible / cert_failed / timeout / binary_missing`), wall-clock vs the 15-hour estimate, free disk, and worker RSS sum. Alerts fire on:

1. `cert_failed` rate above 30% on any single (map, n) — indicates `max_outer_rounds` is too tight, or the FakeSolver fallback is firing because LaCAM* is unavailable.
2. `binary_missing` rate above 0% for any non-Kottinger baseline — indicates `scripts/install_baselines.sh` did not complete successfully.
3. `precheck_disagreement = True` on any RQ4 row — indicates a real cross-method consistency bug (the SlackCertify and ILP wait-feasibility prechecks disagree on the same cell). This is a hard stop; investigate before continuing.

For mid-sweep failure handling (worker crash, disk full, single-experiment fail, paths-valid anomaly, wall-clock overrun, tmux loss), see §9.

### 6.4 Post-sweep processing

After completion, the pipeline is:

```bash
# Aggregate
python analysis/aggregate_results.py \
    --input-dir ${OUT}/raw \
    --output ${OUT}/summary.csv \
    --bootstrap --holm

# Figures + LaTeX tables (chained by default)
python analysis/make_figures.py \
    --input-dir ${OUT}/raw \
    --output-dir ${OUT}/figures \
    --tables-output-dir ${OUT}/summary

# Or figures only
python analysis/make_figures.py --only-figures \
    --input-dir ${OUT}/raw \
    --output-dir ${OUT}/figures

# Or tables only
python analysis/make_tables.py \
    --input-dir ${OUT}/raw \
    --output-dir ${OUT}/summary

# Paper build: skipped (paper sources not in this artifact; submitted separately)
```

`--bootstrap` adds bootstrap CIs (paired, 10k resamples by default). `--holm` adds Holm–Bonferroni multiple-comparison correction across paired method rows. The companion `${OUT}/significance_summary.csv` captures p-values per method pairing.

### 6.5 Interpreting `status` rows

Every cell in every `results/raw/${rq}.csv` has a `status` column drawn from a fixed seven-value taxonomy:

| `status` | Interpretation | Action |
|---|---|---|
| `ok` | Certification succeeded; `success_rate`, `executed_soc_mean`, `total_waits`, `outer_rounds` populated. | None. |
| `wait_infeasible` | Structural infeasibility detected by the `is_wait_resolvable` precheck (e.g., head-on swap in a 1-D corridor; agents share every path cell). `diagnostics["unresolvable_conflicts"]` lists the offending edges. | None — this is a meaningful experimental result. §V reports the per-map fraction (e.g., 12% of `maze-32-32-4` n=400 cells). |
| `cert_failed` | The certifier ran but exceeded `max_outer_rounds` without converging. Distinct from `wait_infeasible`: the structure is wait-resolvable in principle, but the greedy didn't find a solution within the budget. | If frequent, either (a) bump `max_outer_rounds` in the YAML, (b) check whether the nominal solver fell back to FakeSolver (`diagnostics["solver_fallback"]=True`), or (c) see §9. |
| `timeout` | A method-specific time limit was exceeded — for RQ4 this is the ILP's `ilp_time_limit_s`; for the baselines it is the wrapper's subprocess timeout. | Expected at the upper end of each baseline's tractability range. The §V analysis pipeline treats `timeout` as a censored observation and reports the timeout rate per method. |
| `nominal_failed` | The nominal MAPF solver itself failed (returned no plan within `nominal_time_limit_s`, or its binary is broken). | Investigate — every nominal cell must have a plan before certification can proceed. Check `${OUT}/${rq}.log` for the underlying solver stderr. |
| `rollout_failed` | The Monte Carlo simulator hit an internal error during a rollout. | Rare; investigate per §9 scenario `executor_crash`. |
| `binary_missing` | A method's binary was not present at the configured path. Distinct from `nominal_failed`: a missing baseline binary is operational, not algorithmic. | Build the missing binary via `scripts/install_baselines.sh` and re-run the affected runner (resume from manifest will skip completed cells). |

The §V tables (Phase 8.2) automatically partition cells by `status`. Reviewers asking "what about the failures?" get the answer in `results/summary/tab4_status_breakdown.tex`.

## 7. Generating paper figures and tables

### 7.1 Aggregate results

After §6, one consolidated summary feeds the figures and tables that operate on aggregated rows:

```bash
python analysis/aggregate_results.py \
    --input-dir ${OUT}/raw \
    --output ${OUT}/summary.csv \
    --bootstrap --holm
```

`--bootstrap` adds bootstrap CIs (P1-6). `--holm` adds Holm–Bonferroni correction across paired method rows. The companion `${OUT}/significance_summary.csv` captures p-values per method pairing.

### 7.2 Generate figures

`analysis/make_figures.py` chains into `analysis/make_tables.py` by default. Use `--only-figures` to skip the table chain.

```bash
python analysis/make_figures.py \
    --input-dir ${OUT}/raw \
    --output-dir ${OUT}/figures
```

Output paths (under `${OUT}/figures/`):

| Figure | Generator (function in `analysis/plots/`) | Output filename |
|---|---|---|
| RQ1 headline: success rate vs n_agents, per map | `headline:generate_headline_figure` | `rq1_headline.pdf` |
| RQ2 Pareto: SoC overhead vs success rate, per (method, delta) | `pareto:generate_pareto_figure` | `rq2_kr_pareto.pdf` |
| RQ3 close analogues: bar chart per map, three methods per group | `close_analogues:generate_close_analogues_figure` | `rq3_close_analogues.pdf` |
| RQ4 optimality gap: violin + bar (matched %) per n_agents | `optimality_gap:generate_optimality_gap_figure` | `rq4_ilp_gap.pdf` |
| RQ5 persistent: success rate vs n_agents, persistent-delay regime | `persistent:generate_persistent_figure` | `rq5_persistent.pdf` |
| Ablation: ordering, solver, budget | `ablation:generate_ablation_figures` | `ablation_{ordering,solver,budget}.pdf` |

Each generator handles missing/empty CSVs by emitting a labeled placeholder PDF with a "no data" message — the pipeline never silently fails to produce an expected file. Matplotlib `Agg` backend is set before `pyplot` import so the pipeline is headless.

### 7.3 Generate tables

```bash
python analysis/make_tables.py \
    --input-dir ${OUT}/raw \
    --output-dir ${OUT}/summary
```

Output paths (under `${OUT}/summary/`):

| Table | Generator | Output filename |
|---|---|---|
| Runtime (Tab 1) | `runtime_table:generate_runtime_table` | `tab1_runtime.tex` |
| Close analogues (Tab 2) | `analogues_table:generate_analogues_table` | `tab2_analogues.tex` |
| Optimality gap (Tab 3) | `optimality_gap_table:generate_optimality_gap_table` | `tab3_optimality_gap.tex` |
| Status breakdown (Tab 4) | `status_breakdown_table:generate_status_breakdown_table` | `tab4_status_breakdown.tex` |

Each `.tex` is a `\begin{tabular}{...} ... \end{tabular}` fragment suitable for `\input{}`-inclusion into a manuscript's experiments section. LaTeX special characters (`& % $ # _ ^ ~ { }`) are escaped via `analysis/tables/_common.py:escape_latex`.

### 7.4 Compile the paper

The LaTeX paper sources are **not included in this artifact** (the manuscript is submitted separately). The reproducible outputs are the figures (`results/figures/*.pdf`) and table fragments (`results/summary/*.tex`); `\input{}` them into your own manuscript build.

## 8. Algorithm and method reference

Every method string accepted by a runner's `methods:` YAML field, the wrapper class it maps to, and where it appears in the experiment matrix.

| `methods:` value | Class | Paper baseline | Default experiments |
|---|---|---|---|
| **Headline certifier** | | | |
| `nominal_unhardened` | (skip-certifier passthrough) | Nominal LaCAM* plan, unhardened | RQ1, RQ5 |
| `slack_certify_bounded` | `slack_certify(mode="bounded", delta=Δ)` | Host paper §IV.A (Theorem 1) | RQ1, RQ2 (vs kR-CBS Pareto), RQ4 (vs ILP), ablation-solver |
| `slack_certify_probabilistic` | `slack_certify(mode="probabilistic", delta=1, p_d, epsilon)` | Host paper §IV.B (Proposition 1) | RQ1, RQ3, RQ5, ablation-ordering, ablation-budget |
| **Robust MAPF baselines** | | | |
| `kr_eecbs` | `KRobustCBSBaseline(solver="eecbs", k=Δ)` | Atzmon JAIR 2020 (k-Robust-CBS variant) | RQ2 |
| `kr_pbs` | `KRobustPBSBaseline(k=Δ)` | Atzmon ICAPS 2020 (k-Robust-PBS variant) | RQ2 |
| **Post-hoc baselines** | | | |
| `btpg_max` | `BTPGMaxBaseline` | Berndt T-RO 2024 (BTPG, max-budget variant) | RQ3 |
| `kottinger_offline` | `KottingerDelayBaseline.solve_offline(plan, delta)` | Kottinger SoCS 2024 (delay-robust MAPF) | RQ3 |
| **Optimality reference** | | | |
| `ilp_exact` | `solve_optimal_wait_insertion(plan, delta, time_limit_s)` | (this paper, §IV.C) | RQ4 |
| **Ablation knobs** | | | |
| `topological`, `risk_descending`, `random` | `slack_certify(ordering=…)` | ablation-ordering | ablation-ordering |
| `eecbs`, `lacam_star`, `pibt` | (nominal solver dispatch) | ablation-solver | ablation-solver |
| `uniform`, `risk_proportional` | `slack_certify(budget_alloc=…)` | ablation-budget | ablation-budget |

The CLI accepts experiment-name dispatch (`python -m experiments --config experiments/configs/rq1_headline.yaml`); the method names above are values inside the YAML's `methods:` list. See each YAML for the full schema.

#### Kottinger provenance: binary-or-binary_missing (no silent fallback)

For the Kottinger baseline, every successful row's `solver_used` is `<nominal>+kottinger_offline_delta=<delta>_binary` — there's only one provenance because there's no fallback path. If the upstream binary is absent, the wrapper raises `SolverNotFoundError` and the runner records `status="binary_missing"` per the established convention. Earlier versions of this codebase shipped a Python reimplementation that delegated to `slack_certify` and tagged `solver_used` with `_reimpl`; the audit's D3 resolution removed it because tagged-`_reimpl` rows looked like real Kottinger comparisons and were easy to forget to filter out of paper-grade plots.

## 9. Troubleshooting

**`third_party/<X>/` is empty.**
Submodule init was skipped. Repair: `git submodule update --init --recursive`. Then re-run the affected `scripts/install_baselines.sh` segment.

**`graph.cpp:143: error: 'memset' is not a member of 'std'`.**
The `pibt2` submodule needs a `#include <cstring>` for gcc-13. Apply the patch documented in `third_party/README.md` (or rebase the submodule onto a namespace fork that includes the patch). Until the fix lands, `pibt` is unavailable on a clean clone on gcc-13+ systems.

**`scripts/install_baselines.sh` reports `[SKIP] kottinger TODO_VERIFY ...`.**
Expected on first build — the wrapper has two TODO_VERIFY placeholders (binary path, CLI flags). Open `aria-systems-group/Delay-Robust-MAPF`'s README, confirm the correct values, and edit `scripts/install_baselines.sh:182` and `baselines/kottinger/wrapper.py:_solve_offline_binary` accordingly. After the edit, re-run `bash scripts/install_baselines.sh kottinger` to build just that solver.

**`status="binary_missing"` on every row of an experiment.**
A baseline binary is not at the expected path. Run `python scripts/check_environment.py --require-binaries` to surface which binary is missing, then rebuild it via `bash scripts/install_baselines.sh <solver>`. Re-running the experiment via `python -m experiments` will resume from manifest and only process the previously-missing cells.

**`status="cert_failed"` on every row of an experiment without `WaitInfeasibleError` in diagnostics.**
The certifier hit `max_outer_rounds` without converging. Three likely causes:
1. **FakeSolver fallback fired** because LaCAM* binary is missing. Check `diagnostics["solver_fallback"]=True`; if so, build LaCAM*.
2. **Cascade pathology** — the nominal plan has very dense shared cells (typical on `maze-32-32-4` at high n). Bump `max_outer_rounds` in the YAML (default 50; safe to raise to 200 if memory allows).
3. **Probabilistic mode with ε too tight** — try ε=0.10 and confirm convergence, then narrow.

**`status="cert_failed"` with `WaitInfeasibleError` in diagnostics.**
Not a bug — this is the structural-infeasibility precheck firing on a head-on-swap-in-corridor topology. Reported in §V as a per-map percentage. If the fraction seems unexpectedly high (>50% on an open map like `random-32-32-20`), check whether the nominal solver is producing pathological plans (LaCAM* sometimes routes through narrow corridors at high density).

**`precheck_disagreement = True` on RQ4 rows.**
A real cross-method consistency bug: the SlackCertify wait-feasibility precheck and the ILP wait-feasibility precheck disagree on the same cell. Hard stop. Inspect the cell's `(map, n, scen_seed)`; both should report `wait_infeasible` or both should not. If one does and the other doesn't, file an issue with the cell key and the offending plan.

**`solved=1` but `success_rate < 0.5` for `slack_certify_probabilistic`.**
The empirical Monte Carlo collision rate exceeds the certified ε bound. This is the most serious correctness signal in the codebase — it means Proposition 1 doesn't hold for the cell. Five known causes, in order of likelihood:
1. **`delta_window` mismatch.** Probabilistic mode must pass `delta=1` to align the certifier's predicate with the simulator's interval-overlap collision model. See `docs/algorithm.md` § "Aligning with the simulator's collision model". The Phase 7 runners pass `delta=1` by default; if you've edited the YAML and forced `delta=0`, that's the cause.
2. **K too small.** With K = 500, the 3σ band on a success rate of 0.95 is ~0.03; with K = 50 (smoke), it's ~0.1. Re-run with K = 1000 and confirm whether the violation persists.
3. **Calibration mismatch in RQ5.** `calibrate_bernoulli_to_persistent` returns a Bernoulli `p_d` that matches the persistent model's empirical blocking rate, but the *shape* of the delay distribution differs. The probabilistic certifier's bound is tight only for Bernoulli; persistent regimes may produce small over-collisions. Documented in §V.
4. **Pre-fix monotonicity bug.** Before commit `9d88825..eedf2a8`, the probabilistic risk formula had a sign error (computed `Pr[no collision]` rather than `Pr[collision]`). If you're running against pre-fix code, the certifier is a no-op and the bound trivially fails. Confirm: `git merge-base --is-ancestor eedf2a8 HEAD`.
5. **Genuine soundness bug.** If 1–4 are ruled out and the violation reproduces on `test_certifier_simulator_agreement`, file an issue with the cell's plan and seed. Pre-fix this pattern was the canary that surfaced the sign-error bug; post-fix it should be impossible.

**`scripts/repro_smoke.sh` exits with `SMOKE FAIL`.**
The script prints which step failed. Re-run the failing step alone (§4) to see the full stderr. The most common failure modes are (1) `pytest tests/integration` failing because a dependency (numpy, scipy) wasn't installed; (2) `python analysis/make_figures.py` failing because matplotlib's Agg backend isn't configured (`MPLBACKEND=Agg` env var fixes it).

**Disk full mid-sweep.** Per-row CSV + manifest writes use `O_APPEND + fsync` with rows under `PIPE_BUF`, so the partially-written CSV and manifest are both preserved and consistent. Free space, then re-run the affected experiment — resume-from-manifest will skip completed cells.

**Sweep takes much longer than estimated.** Likely cause: a baseline binary is silently missing and a runner is dispatching FakeSolver instead. Check `${OUT}/${rq}.log` for `[warn] LaCAM* binary missing; falling back to FakeSolver`. FakeSolver is ~2–3 orders of magnitude slower at n=200 on `random-32-32-20` (BFS over the full grid per agent vs LaCAM*'s heuristic search).

**`certify_runtime_s` is 0.0 on every row.**
The certifier's timing is captured around the `slack_certify(...)` call. If it reads 0.0, the runner is hitting an early-return path (e.g., the precheck rejected the instance before the algorithm ran). Confirm by checking `status="wait_infeasible"` on those rows — that's expected and the runtime field is correctly zero.

**Missing figures or tables.** The figures land in `results/figures/` and the table fragments in `results/summary/`. Confirm those directories exist and are populated; if you ran `make_figures.py --smoke`, they land under `results/figures_smoke/` and `results/summary_smoke/` instead — re-run against the real CSVs.

## 10. Citation and attribution

Paper citation: **TBD — ASYU 2026 published artifact**.

Software citation (Zenodo DOI assigned on release): see `CITATION.cff` at the repo root.

External baselines vendored under `third_party/`. License attribution and pinned SHAs (all `TODO_PIN_COMMIT_HASH` until the first real build):

| Submodule | Upstream | Paper | License | Pinned SHA |
|---|---|---|---|---|
| `third_party/eecbs` | <https://github.com/Jiaoyang-Li/EECBS> | Li et al., AAAI 2021 | USC Research License | TODO_PIN_COMMIT_HASH |
| `third_party/lacam` | <https://github.com/Kei18/lacam> | Okumura, AAAI 2023 | MIT | TODO_PIN_COMMIT_HASH |
| `third_party/lacam2` | <https://github.com/Kei18/lacam2> | Okumura, IJCAI 2023 | MIT | TODO_PIN_COMMIT_HASH |
| `third_party/pibt2` | <https://github.com/Kei18/pibt2> | Okumura et al., AIJ 2022 | MIT | TODO_PIN_COMMIT_HASH |
| `third_party/btpg` | <https://github.com/Jiaoyang-Li/BTPG> | Berndt et al., T-RO 2024 | TODO_VERIFY | TODO_PIN_COMMIT_HASH |
| `third_party/delay-introduction` | <https://github.com/aria-systems-group/Delay-Robust-MAPF> | Kottinger et al., SoCS 2024 | TODO_VERIFY | TODO_PIN_COMMIT_HASH |

After `bash scripts/install_baselines.sh` completes on a real rig, update `third_party/README.md` with the actual SHAs that were checked out, and update each row above accordingly.

## 11. What this artifact does NOT do

- **Does not run on Windows.** WSL2 with Ubuntu 22.04 / 24.04 works as a Linux target but is not in CI.
- **Does not include a real Kottinger 2024 implementation until §3.4's TODO_VERIFY placeholders are confirmed.** Until the upstream binary is built and the placeholders are resolved, RQ3's Kottinger rows land at `status="binary_missing"` (post-D3 the wrapper raises `SolverNotFoundError` rather than falling back to a stub). The §V RQ3 Kottinger column is regenerated after the binary build succeeds.
- **Does not validate Proposition 1 against persistent (non-Bernoulli) delay models.** The probabilistic bound is tight for Bernoulli; the RQ5 calibration to a persistent regime produces empirical collision rates that are usually below the bound, but the proof does not cover persistent delays directly. §V discusses this in the experimental-validity section.
- **Does not handle dynamic obstacle introduction.** All experiments are on static MAPF instances. Lifelong / online variants are out of scope and tracked in TODO.md.
- **Does not include the paper PDF or its LaTeX source.** The manuscript is submitted separately; this artifact ships the figures (`results/figures/`) and table fragments (`results/summary/`) that a manuscript would `\input{}`.
- **Does not provide a reference implementation of the ILP encoding's warm-start.** The current encoding uses cold-start binary indicators; warm-starting from the nominal conflict set is a 1–2 day optimization that's tracked as a P3 follow-up.
