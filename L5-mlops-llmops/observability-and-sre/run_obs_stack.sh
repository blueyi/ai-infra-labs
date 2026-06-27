#!/usr/bin/env bash
# L5 observability: mock_infer + prometheus/grafana stack smoke test
set -euo pipefail
LAB="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB"
LOG="$LAB/obs_stack.log"
: > "$LOG"

source "$HOME/workspace/repos/ai-infra-labs/.venv/bin/activate"
pkill -f "mock_infer.py" 2>/dev/null || true
python mock_infer.py & MOCK_PID=$!
sleep 3
curl -sf http://127.0.0.1:8000/metrics | head -5 2>&1 | tee -a "$LOG"

docker compose down -v 2>/dev/null || true
docker compose up -d 2>&1 | tee -a "$LOG"
sleep 20
curl -sf http://127.0.0.1:9090/-/ready 2>&1 | tee -a "$LOG"
curl -sf http://127.0.0.1:3000/api/health 2>&1 | tee -a "$LOG"
sleep 70
ALERTS=$(curl -sf http://127.0.0.1:9090/api/v1/alerts 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('data',{}).get('alerts',[])))")
echo "[alerts] count=$ALERTS" | tee -a "$LOG"

docker compose down -v 2>&1 | tee -a "$LOG"
kill $MOCK_PID 2>/dev/null || true
echo "[ok] observability stack smoke passed" | tee -a "$LOG"
