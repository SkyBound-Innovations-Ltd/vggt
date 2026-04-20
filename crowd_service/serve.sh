#!/usr/bin/env bash
# Serve the crowd viewer folder on http://localhost:8000/
# (static files only — frames.json must exist next to index.html)
set -eu
cd "$(dirname "$0")"
if [ ! -f frames.json ]; then
  echo "frames.json is missing. Run:"
  echo "  python preprocess.py -i ../outputs/cc_run_20260407_200138_crowd/state_estimation.json -o frames.json"
  exit 1
fi
exec python3 -m http.server "${PORT:-8000}"
