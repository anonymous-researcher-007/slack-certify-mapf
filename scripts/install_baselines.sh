#!/usr/bin/env bash
# install_baselines.sh — fetch and build every vendored MAPF solver,
# then copy the resulting binaries into slackcertify/solvers/external_bin/.
#
# Designed to be idempotent: a binary that already exists with the
# expected SHA-256 is *not* rebuilt. After a clean run, every wrapper in
# slackcertify/solvers/ has a working binary at the expected path.
#
# Usage:
#     bash scripts/install_baselines.sh
#
# Environment variables:
#     SLACKCERTIFY_FORCE_REBUILD=1   rebuild every solver from scratch
#     SLACKCERTIFY_JOBS=N            override -j parallelism (default: nproc)

set -euo pipefail

# ---------------------------------------------------------------- paths
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"
BIN_DIR="${REPO_ROOT}/src/slackcertify/solvers/external_bin"
mkdir -p "${BIN_DIR}"

# ---------------------------------------------------------------- helpers
log()   { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[install]\033[0m %s\n' "$*" >&2; }
fail()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        fail "required command '${cmd}' not found on PATH; see third_party/README.md"
    fi
}

cores() {
    if [[ -n "${SLACKCERTIFY_JOBS:-}" ]]; then
        printf '%s' "${SLACKCERTIFY_JOBS}"
    elif command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    else
        printf '2'
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        fail "neither sha256sum nor shasum is available"
    fi
}

human_size() {
    local file="$1"
    if command -v du >/dev/null 2>&1; then
        du -h "${file}" | awk '{print $1}'
    else
        printf '?'
    fi
}

# ---------------------------------------------------------------- prereqs
log "checking prerequisites"
require_cmd git
require_cmd cmake
require_cmd make
# Either g++ or clang++ will do; prefer g++.
if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    fail "no C++ compiler found (need g++ or clang++)"
fi

JOBS="$(cores)"
log "building with -j${JOBS}"

# ---------------------------------------------------------------- expected sha256s
# Replace these with the actual digests once a known-good build exists,
# then commit the updated values. `scripts/update_baseline_pins.sh`
# refreshes them automatically after a successful build.
declare -A EXPECTED_SHA256=(
    [eecbs]="TODO_EXPECTED_SHA256"
    [lacam]="TODO_EXPECTED_SHA256"
    [lacam_star]="TODO_EXPECTED_SHA256"
    [pibt]="TODO_EXPECTED_SHA256"
    [btpg]="TODO_EXPECTED_SHA256"
    [kottinger]="TODO_EXPECTED_SHA256"
)

# ---------------------------------------------------------------- build helpers
init_submodule() {
    local sub_path="$1"
    if [[ ! -d "${THIRD_PARTY_DIR}/${sub_path}/.git" ]] && \
       [[ ! -f "${THIRD_PARTY_DIR}/${sub_path}/.git" ]]; then
        log "initialising submodule third_party/${sub_path}"
        (cd "${REPO_ROOT}" && git submodule update --init --recursive --depth 1 \
            "third_party/${sub_path}")
    fi
}

cmake_build() {
    # cmake_build <submodule_dir> <extra_cmake_args...>
    local sub_dir="$1"; shift
    local build_dir="${THIRD_PARTY_DIR}/${sub_dir}/build"
    mkdir -p "${build_dir}"
    (cd "${build_dir}" && cmake -DCMAKE_BUILD_TYPE=Release "$@" .. && \
                          make -j"${JOBS}")
}

install_binary() {
    # install_binary <built_path> <output_name>
    local built="$1"
    local out_name="$2"
    local dest="${BIN_DIR}/${out_name}"

    if [[ ! -f "${built}" ]]; then
        fail "expected built binary not found at ${built}"
    fi

    if [[ -f "${dest}" && "${SLACKCERTIFY_FORCE_REBUILD:-0}" != "1" ]]; then
        local actual_sha
        actual_sha="$(sha256_of "${dest}")"
        local expected="${EXPECTED_SHA256[${out_name}]:-TODO_EXPECTED_SHA256}"
        if [[ "${expected}" != "TODO_EXPECTED_SHA256" && "${actual_sha}" == "${expected}" ]]; then
            log "skipping ${out_name}: already installed with matching SHA-256"
            return 0
        fi
    fi

    cp -f "${built}" "${dest}"
    chmod +x "${dest}"
    local sha; sha="$(sha256_of "${dest}")"
    log "installed ${out_name} -> ${dest} (sha256=${sha:0:12}...)"

    local expected="${EXPECTED_SHA256[${out_name}]:-TODO_EXPECTED_SHA256}"
    if [[ "${expected}" != "TODO_EXPECTED_SHA256" && "${sha}" != "${expected}" ]]; then
        warn "WARNING: ${out_name} sha256 mismatch (expected ${expected}, got ${sha})"
    fi
}

# ---------------------------------------------------------------- per-solver
build_eecbs() {
    log "==> EECBS"
    init_submodule "EECBS"
    cmake_build "EECBS"
    install_binary "${THIRD_PARTY_DIR}/EECBS/build/eecbs" "eecbs"
}

build_lacam() {
    log "==> lacam"
    init_submodule "lacam"
    cmake_build "lacam"
    # Upstream produces ./build/main; rename on copy.
    install_binary "${THIRD_PARTY_DIR}/lacam/build/main" "lacam"
}

build_lacam_star() {
    log "==> lacam_star (lacam2)"
    init_submodule "lacam_star"
    cmake_build "lacam_star"
    install_binary "${THIRD_PARTY_DIR}/lacam_star/build/main" "lacam_star"
}

build_pibt2() {
    log "==> pibt2"
    init_submodule "pibt2"
    cmake_build "pibt2"
    # pibt2 build directory layout: build/pibt
    install_binary "${THIRD_PARTY_DIR}/pibt2/build/pibt" "pibt"
}

build_btpg() {
    log "==> BTPG"
    init_submodule "btpg"
    cmake_build "btpg"
    install_binary "${THIRD_PARTY_DIR}/btpg/build/btpg" "btpg"
}

build_delay_introduction() {
    log "==> delay-introduction (Kottinger 2024)"
    init_submodule "delay-introduction"
    # TODO_VERIFY: confirm against upstream README which CMake flags
    # (if any) are required and whether the binary name is "kottinger",
    # "delay_robust", "min_delay" or similar.
    cmake_build "delay-introduction"
    # TODO_VERIFY: the built artifact's path inside build/. Common
    # candidates: build/kottinger, build/delay_robust_mapf,
    # build/main. Update once the maintainer verifies against the upstream build.
    install_binary "${THIRD_PARTY_DIR}/delay-introduction/build/kottinger" "kottinger"
}

# ---------------------------------------------------------------- run
build_eecbs
build_lacam
build_lacam_star
build_pibt2
build_btpg
build_delay_introduction

# ---------------------------------------------------------------- summary
log "build summary:"
printf '  %-16s %-12s %-64s\n' "binary" "size" "sha256"
for name in eecbs lacam lacam_star pibt btpg kottinger; do
    file="${BIN_DIR}/${name}"
    if [[ -f "${file}" ]]; then
        printf '  %-16s %-12s %s\n' "${name}" "$(human_size "${file}")" "$(sha256_of "${file}")"
    else
        printf '  %-16s %-12s %s\n' "${name}" "MISSING" "-"
    fi
done

log "done. To pin the produced binaries, run scripts/update_baseline_pins.sh."
