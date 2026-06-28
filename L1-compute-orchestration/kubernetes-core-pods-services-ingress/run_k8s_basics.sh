#!/usr/bin/env bash
# L1.7 — kind + mock Deployment/Service smoke (no GPU)
set -euo pipefail
CLUSTER="${KIND_CLUSTER_NAME:-k8s-basics-lab}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

if ! command -v kind >/dev/null 2>&1; then
  echo "[skip] kind not installed"
  exit 0
fi
if ! command -v kubectl >/dev/null 2>&1; then
  echo "[skip] kubectl not installed"
  exit 0
fi

kind get clusters 2>/dev/null | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER"

cat >"${ROOT}/mock-infer.yaml" <<'YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mock-infer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mock-infer
  template:
    metadata:
      labels:
        app: mock-infer
    spec:
      containers:
        - name: server
          image: python:3.12-slim
          command: ["python", "-m", "http.server", "8080"]
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: mock-infer-svc
spec:
  selector:
    app: mock-infer
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
YAML

kubectl apply -f "${ROOT}/mock-infer.yaml"
kubectl rollout status deployment/mock-infer --timeout=120s
kubectl run curl-test --rm -i --restart=Never --image=curlimages/curl:8.5.0 -- \
  curl -sf http://mock-infer-svc/ | head -c 80
echo
echo "[ok] k8s basics lab passed"
