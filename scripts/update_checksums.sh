#!/usr/bin/env bash
# update_checksums.sh — refresh benchmarks/checksums.sha256 from the
# current contents of benchmarks/{maps,scenarios}/.
#
# Format: "<sha256>  <path>" (one line per file), preceded by a header
# block. Run after benchmarks/download_movingai.sh produces a known-good
# benchmark set.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MAPS_DIR="${REPO_ROOT}/benchmarks/maps"
SCEN_DIR="${REPO_ROOT}/benchmarks/scenarios"
OUT="${REPO_ROOT}/benchmarks/checksums.sha256"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf 'NO_SHA_TOOL'
    fi
}

emit_dir() {
    local dir="$1"
    if [[ ! -d "${dir}" ]]; then
        return
    fi
    while IFS= read -r -d '' file; do
        rel="${file#${REPO_ROOT}/}"
        printf '%s  %s\n' "$(sha256_of "${file}")" "${rel}"
    done < <(find "${dir}" -type f -print0 | sort -z)
}

{
    printf '# MovingAI benchmark checksums.\n'
    printf '# Auto-populated by scripts/update_checksums.sh after the\n'
    printf '# first successful run of benchmarks/download_movingai.sh.\n'
    printf '# Format: <sha256>  <path>\n'
    printf '# Generated: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    emit_dir "${MAPS_DIR}"
    emit_dir "${SCEN_DIR}"
} >"${OUT}"

printf 'Wrote %s\n' "${OUT}"
