import asyncio
import time
import random
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("gw")

app = FastAPI()

# ---- 路由阈值与成本表（单位：相对成本） ----
LEN_THRESHOLD = 40          # prompt 长度阈值：超过走 slow
COST = {"fast": 1.0, "slow": 8.0}

# ---- 用信号量模拟两个模型的有限并发（容量） ----
SEM = {"fast": asyncio.Semaphore(8), "slow": asyncio.Semaphore(2)}


class Req(BaseModel):
    prompt: str
    slow_inject: bool = False   # 实验开关：是否注入慢请求


async def mock_model(name: str, prompt: str, slow_inject: bool) -> dict:
    """模拟一次推理：fast 快、slow 慢；prefill 随 prompt 长度增长。"""
    t0 = time.perf_counter()
    async with SEM[name]:                      # 排队点：拿不到信号量就在这等
        t_acquired = time.perf_counter()
        queue_ms = (t_acquired - t0) * 1000

        # prefill：随 prompt 长度线性增长（模拟 prefill-bound）
        prefill = len(prompt) * (0.002 if name == "fast" else 0.006)
        # decode：固定基线
        decode = 0.05 if name == "fast" else 0.4
        if slow_inject:                        # 注入一个长尾慢请求
            decode += 2.0
        await asyncio.sleep(prefill + decode)

        svc_ms = (time.perf_counter() - t_acquired) * 1000
        return {"model": name, "queue_ms": round(queue_ms, 1),
                "service_ms": round(svc_ms, 1), "cost": COST[name]}


@app.post("/v1/fast")
async def fast(r: Req):
    return await mock_model("fast", r.prompt, r.slow_inject)


@app.post("/v1/slow")
async def slow(r: Req):
    return await mock_model("slow", r.prompt, r.slow_inject)


# ---------------- 路由网关 ----------------
@app.post("/route")
async def route(r: Req):
    t_in = time.perf_counter()

    # 1) 决策：按 prompt 长度路由（成本/延迟策略的最简形态）
    target = "slow" if len(r.prompt) > LEN_THRESHOLD else "fast"
    t_decide = time.perf_counter()

    # 2) 转发到对应 mock 模型（此处直接进程内调用以保持纯 CPU 单文件）
    resp = await mock_model(target, r.prompt, r.slow_inject)
    t_out = time.perf_counter()

    # 3) 分段计时日志：路由 / 排队 / 服务 三段拆开
    route_ms = (t_decide - t_in) * 1000
    total_ms = (t_out - t_in) * 1000
    log.info(
        f"[ROUTE] target={target} len={len(r.prompt)} "
        f"route_ms={route_ms:.2f} queue_ms={resp['queue_ms']} "
        f"service_ms={resp['service_ms']} total_ms={total_ms:.1f} "
        f"cost={resp['cost']} inject={r.slow_inject}"
    )
    return {"routed_to": target, "total_ms": round(total_ms, 1), **resp}
