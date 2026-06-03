#!/usr/bin/env bash
# download_movingai.sh — fetch §V MovingAI benchmarks via per-map ZIP archives.
# Real layout: https://www.movingai.com/benchmarks/mapf/<name>.map.zip
#          and https://www.movingai.com/benchmarks/mapf/<name>.map-scen-random.zip
# (scen archive nests files under scen-random/; -j flattens that.)
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MAPS_DIR="${REPO_ROOT}/benchmarks/maps"
SCEN_DIR="${REPO_ROOT}/benchmarks/scenarios"
TMP_DIR="${REPO_ROOT}/benchmarks/_dl"
BASE_URL="https://www.movingai.com/benchmarks/mapf"
mkdir -p "${MAPS_DIR}" "${SCEN_DIR}" "${TMP_DIR}"
MAPS=(
    "warehouse-10-20-10-2-1"
    "warehouse-20-40-10-2-2"
    "random-32-32-20"
    "room-64-64-16"
    "maze-32-32-4"
)
ok=0; skip=0; fail=0
log_ok()   { printf '\033[1;32m[OK]\033[0m   %s\n' "$*"; ok=$((ok+1)); }
log_skip() { printf '\033[1;34m[SKIP]\033[0m %s\n' "$*"; skip=$((skip+1)); }
log_fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; fail=$((fail+1)); }
fetch_zip() {
    local url="$1" dest="$2"
    if curl --max-time 120 -sSLf -o "${dest}" "${url}"; then return 0
    else log_fail "download failed: ${url}"; rm -f "${dest}"; return 1; fi
}
for name in "${MAPS[@]}"; do
    if [[ -f "${MAPS_DIR}/${name}.map" ]]; then
        log_skip "maps/${name}.map (present)"
    else
        mz="${TMP_DIR}/${name}.map.zip"
        if fetch_zip "${BASE_URL}/${name}.map.zip" "${mz}"; then
            unzip -o -j "${mz}" -d "${TMP_DIR}/m_${name}" >/dev/null
            f="$(find "${TMP_DIR}/m_${name}" -name "${name}.map" | head -n1)"
            if [[ -n "${f}" ]]; then cp "${f}" "${MAPS_DIR}/${name}.map"; log_ok "maps/${name}.map"
            else log_fail "maps/${name}.map (not in archive)"; fi
        fi
    fi
    existing="$(find "${SCEN_DIR}" -name "${name}-random-*.scen" 2>/dev/null | wc -l)"
    if [[ "${existing}" -ge 25 ]]; then
        log_skip "scenarios/${name}-random-*.scen (${existing} present)"
    else
        sz="${TMP_DIR}/${name}.scen.zip"
        if fetch_zip "${BASE_URL}/${name}.map-scen-random.zip" "${sz}"; then
            unzip -o -j "${sz}" -d "${TMP_DIR}/s_${name}" >/dev/null
            n=0
            while IFS= read -r ff; do cp "${ff}" "${SCEN_DIR}/$(basename "${ff}")"; n=$((n+1)); done \
                < <(find "${TMP_DIR}/s_${name}" -name "${name}-random-*.scen")
            if [[ "${n}" -gt 0 ]]; then log_ok "scenarios/${name}-random-*.scen (${n} files)"
            else log_fail "scenarios/${name}-random (none in archive)"; fi
        fi
    fi
done
rm -rf "${TMP_DIR}"
printf '\n=== summary ===\n  OK:   %d\n  SKIP: %d\n  FAIL: %d\n' "${ok}" "${skip}" "${fail}"
[[ "${fail}" -gt 0 ]] && { printf '\nSome downloads failed.\n'; exit 1; } || exit 0
