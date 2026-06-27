import asyncio, time
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N, CONCURRENCY = 200, 50          # 总请求数 / 并发数
PROMPT = "用三句话解释什么是分页式 KV Cache。"

async def one_req(sem, latencies):
    async with sem:
        t0 = time.perf_counter()
        await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=128,
        )
        latencies.append(time.perf_counter() - t0)

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    latencies = []
    t_start = time.perf_counter()
    await asyncio.gather(*[one_req(sem, latencies) for _ in range(N)])
    wall = time.perf_counter() - t_start

    latencies.sort()
    p50 = latencies[int(0.50 * N)]
    p99 = latencies[int(0.99 * N) - 1]
    print(f"[throughput] {N / wall:.1f} req/s  (wall={wall:.1f}s)")
    print(f"[latency] P50={p50*1000:.0f}ms  P99={p99*1000:.0f}ms")

asyncio.run(main())
