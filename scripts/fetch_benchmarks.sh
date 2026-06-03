#!/usr/bin/env bash
# fetch_benchmarks.sh — convenience alias for benchmarks/download_movingai.sh.
set -euo pipefail
exec bash "$(dirname "$0")/../benchmarks/download_movingai.sh" "$@"
