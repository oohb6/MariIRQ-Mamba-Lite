#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Examples:
# bash scripts/run_e0_v2_cached.sh --variant e0 --phase all
# bash scripts/run_e0_v2_cached.sh --variant both --phase cache
exec python -u tools/run_e0_v2_cached.py "$@"
