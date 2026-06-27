#!/usr/bin/env bash
# L1 resource-scheduling: kind + fake GPU + Volcano gang job
set -euo pipefail
export PATH="$HOME/.local/bin:/tmp:$PATH"
LAB="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB"
LOG="$LAB/kind_gang.log"
: > "$LOG"

kind delete cluster --name gpu-lab 2>/dev/null || true
kind create cluster --name gpu-lab --config kind-gpu.yaml 2>&1 | tee -a "$LOG"
kubectl get nodes 2>&1 | tee -a "$LOG"

for N in $(kubectl get nodes -l '!node-role.kubernetes.io/control-plane' -o name); do
  kubectl proxy --port=8001 & PROXY_PID=$!
  sleep 2
  NODE=${N#node/}
  curl -s --header "Content-Type: application/json-patch+json" \
    --request PATCH \
    --data '[{"op":"add","path":"/status/capacity/nvidia.com~1gpu","value":"4"}]' \
    "http://localhost:8001/api/v1/nodes/${NODE}/status" >/dev/null
  kill $PROXY_PID
done
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.capacity.nvidia\\.com/gpu 2>&1 | tee -a "$LOG"

helm repo add volcano-sh https://volcano-sh.github.io/helm-charts 2>/dev/null || true
helm repo update 2>&1 | tee -a "$LOG"
helm upgrade --install volcano volcano-sh/volcano -n volcano-system --create-namespace 2>&1 | tee -a "$LOG"
kubectl -n volcano-system wait --for=condition=Available deploy --all --timeout=180s 2>&1 | tee -a "$LOG"

kubectl apply -f gang-job.yaml 2>&1 | tee -a "$LOG"
sleep 15
kubectl get podgroup 2>&1 | tee -a "$LOG"
kubectl get pods -l volcano.sh/job-name=dist-train-demo -o wide 2>&1 | tee -a "$LOG"
RUNNING=$(kubectl get pods -l volcano.sh/job-name=dist-train-demo --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
echo "[gang] running_pods=$RUNNING (expect 6)" | tee -a "$LOG"
test "$RUNNING" -eq 6

kubectl delete -f gang-job.yaml --ignore-not-found 2>&1 | tee -a "$LOG"
kind delete cluster --name gpu-lab 2>&1 | tee -a "$LOG"
echo "[ok] kind gang lab passed" | tee -a "$LOG"
