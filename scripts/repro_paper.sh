#!/usr/bin/env bash
# scripts/repro_paper.sh — full §V experimental sweep.
#
# Production-quality, used to regenerate every paper artifact from
# scratch. Estimated runtime: 24-72 hours on the reference hardware
# described in docs/reproducibility.md.
#
# Each runner appends to its own CSV under results/raw/ and tracks
# completion in results/manifests/, so a crashed run can be resumed by
# re-invoking this script — completed cells are skipped on the second
# pass.
#
# Usage:
#     bash scripts/repro_paper.sh
#
# Environment variables (passed through):
#     SLACKCERTIFY_JOBS        per-rollout parallelism
#     SLACKCERTIFY_FORCE_REBUILD  rebuild solver binaries from scratch

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENTS=(
    rq1_headline
    rq2_kr_pareto
    rq3_close_analogues
    rq4_ilp_gap
    rq5_persistent
    ablation_ordering
    ablation_solver
    ablation_budget
)
TOTAL_STEPS=$((3 + ${#EXPERIMENTS[@]} + 2))

step() {
    local n="$1"; shift
    printf '\033[1;34m[STEP %d/%d]\033[0m %s\n' "${n}" "${TOTAL_STEPS}" "$*"
}
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

mkdir -p results/raw results/manifests results/figures results/summary

i=1
# ---------------------------------------------------------------- environment
step "${i}" "verifying environment with --require-binaries"
python scripts/check_environment.py --require-binaries
ok "environment check passed"
i=$((i + 1))

# ---------------------------------------------------------------- benchmarks
step "${i}" "fetching benchmarks (idempotent)"
bash scripts/fetch_benchmarks.sh
ok "benchmarks fetched"
i=$((i + 1))

# ---------------------------------------------------------------- baseline binaries
step "${i}" "building / verifying baseline binaries"
bash scripts/install_baselines.sh
ok "baseline binaries built"
i=$((i + 1))

# ---------------------------------------------------------------- experiments
for experiment in "${EXPERIMENTS[@]}"; do
    step "${i}" "running ${experiment}"
    python -m experiments \
        --config "experiments/configs/${experiment}.yaml" \
        --output "results/raw/${experiment}.csv" \
        --manifest "results/manifests/${experiment}.jsonl"
    ok "${experiment} complete"
    i=$((i + 1))
done

# ---------------------------------------------------------------- analysis
step "${i}" "rendering figures + tables from results/raw/*.csv"
python analysis/make_figures.py
ok "figures + tables written"
i=$((i + 1))

# ---------------------------------------------------------------- paper (removed)
step "${i}" "paper PDF (skipped)"
# The paper sources (docs/paper/) are not part of this artifact; the manuscript
# is submitted separately through the conference system. Figures and tables are
# still produced above and listed below.
ok "paper sources not included in this artifact; skipping paper compilation"

printf '\n\033[1;32mARTIFACTS OK\033[0m\n'
printf '  figures:     results/figures/*.pdf\n'
printf '  tables:      results/summary/*.tex\n'
printf '  raw CSVs:    results/raw/*.csv\n'
