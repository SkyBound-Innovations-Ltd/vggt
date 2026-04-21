#!/usr/bin/env bash
# Serve the crowd viewer on http://localhost:8000/
#   • If uvicorn + fastapi are available → full service with /api/eta
#   • Otherwise falls back to static-only (no ETA endpoint).
set -eu
cd "$(dirname "$0")"
if [ ! -f frames.json ]; then
  echo "frames.json is missing. Run:"
  echo "  python preprocess.py -i ../outputs/cc_run_20260407_200138_crowd/state_estimation.json -o frames.json"
  exit 1
fi
if python3 -c "import uvicorn, fastapi" >/dev/null 2>&1; then
  echo "Launching FastAPI (uvicorn) with /api/eta on port ${PORT:-8000} …"
  exec python3 -m uvicorn eta_service:app --host 0.0.0.0 --port "${PORT:-8000}"
else
  echo "uvicorn/fastapi not installed — static-only fallback."
  exec python3 -m http.server "${PORT:-8000}"
fi
