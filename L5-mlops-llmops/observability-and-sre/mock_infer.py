# mock_infer.py —— 用 prometheus_client 暴露推理指标，纯 CPU
# pip install prometheus_client
import random
import time
import threading
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# 请求总数（用于算 QPS：rate(infer_requests_total[1m])）
REQUESTS = Counter("infer_requests_total", "推理请求总数", ["status"])
# 请求延迟直方图（用于算 P99：histogram_quantile）
LATENCY = Histogram(
    "infer_request_latency_seconds", "推理请求延迟(秒)",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)
# KV Cache 利用率（模拟显存压力）
KV_UTIL = Gauge("infer_kv_cache_utilization", "KV Cache 利用率 0-1")

def serve_one():
    """模拟一次推理：随机延迟，偶发慢请求触发长尾。"""
    base = random.uniform(0.03, 0.12)
    # 10% 概率制造长尾，用来触发 SLO 告警
    tail = random.uniform(0.4, 1.5) if random.random() < 0.10 else 0.0
    latency = base + tail
    time.sleep(latency)
    LATENCY.observe(latency)
    REQUESTS.labels(status="ok").inc()
    KV_UTIL.set(min(0.95, random.gauss(0.6, 0.15)))

def workload():
    while True:
        serve_one()

if __name__ == "__main__":
    start_http_server(8000)           # /metrics 暴露在 :8000
    print("[mock-infer] metrics on http://0.0.0.0:8000/metrics")
    # 起多个并发「请求」制造流量
    for _ in range(4):
        threading.Thread(target=workload, daemon=True).start()
    while True:
        time.sleep(3600)
