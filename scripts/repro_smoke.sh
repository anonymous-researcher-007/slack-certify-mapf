#!/usr/bin/env bash
# scripts/repro_smoke.sh — end-to-end smoke run for slack-certify-mapf.
#
# Validates the entire pipeline against the bundled fixture CSVs and
# tiny_4x4 smoke tests. Target runtime: under 15 minutes on a stock
# laptop. Used by the weekly .github/workflows/smoke.yml schedule.
#
# Usage:
#     bash scripts/repro_smoke.sh
#
# Exit codes:
#     0   SMOKE OK
#     1   any step failed (the step prefix in the log identifies which)

set -euo pipefail

# ---------------------------------------------------------------- paths
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------- logging
TOTAL_STEPS=5
step() {
    local n="$1"; shift
    printf '\033[1;34m[STEP %d/%d]\033[0m %s\n' "${n}" "${TOTAL_STEPS}" "$*"
}
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. environment
step 1 "verifying environment"
python scripts/check_environment.py
ok "environment check passed"

# ---------------------------------------------------------------- 2. integration smokes
step 2 "running integration smoke tests"
python -m pytest tests/integration -v -m "not slow" --no-cov
ok "integration smokes passed"

# ---------------------------------------------------------------- 3. analysis pipeline
step 3 "running analysis pipeline against smoke fixture"
python analysis/make_figures.py --smoke
ok "make_figures.py --smoke completed"

# ---------------------------------------------------------------- 4. verify artifacts
step 4 "verifying generated artifacts"
FIG_DIR="${REPO_ROOT}/results/figures_smoke"
TAB_DIR="${REPO_ROOT}/results/summary_smoke"

EXPECTED_PDFS=(
    rq1_headline.pdf
    rq2_kr_pareto.pdf
    rq3_close_analogues.pdf
    rq4_ilp_gap.pdf
    rq5_persistent.pdf
    ablation_ordering.pdf
    ablation_solver.pdf
    ablation_budget.pdf
)
EXPECTED_TEX=(
    tab1_runtime.tex
    tab2_analogues.tex
    tab3_optimality_gap.tex
    tab4_status_breakdown.tex
)

for f in "${EXPECTED_PDFS[@]}"; do
    path="${FIG_DIR}/${f}"
    if [[ ! -s "${path}" ]]; then
        fail "missing or empty figure: ${path}"
    fi
    # Magic-byte check via head -c.
    magic="$(head -c 4 "${path}")"
    if [[ "${magic}" != "%PDF" ]]; then
        fail "not a PDF (magic=${magic}): ${path}"
    fi
done
ok "all 8 figure PDFs verified"

for f in "${EXPECTED_TEX[@]}"; do
    path="${TAB_DIR}/${f}"
    if [[ ! -s "${path}" ]]; then
        fail "missing or empty tex: ${path}"
    fi
    if ! head -1 "${path}" | grep -q '\\begin{tabular}'; then
        fail "no \\begin{tabular} on first line of ${path}"
    fi
done
ok "all 4 table .tex fragments verified"

# ---------------------------------------------------------------- 5. paper build (removed)
step 5 "paper build (skipped)"
# The paper sources (docs/paper/) are not part of this artifact; the manuscript
# is submitted separately. Figures and tables are still generated above.
warn "paper sources not included in this artifact; skipping paper build"

printf '\n\033[1;32mSMOKE OK\033[0m\n'
