import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, HnswConfigDiff, PointStruct, SearchParams,
)

DIM, N, NQ, TOPK = 128, 2000, 50, 10
rng = np.random.default_rng(42)

# 构造数据集与查询集（随机向量，仅用于对比相对趋势）
data = rng.normal(size=(N, DIM)).astype("float32")
queries = rng.normal(size=(NQ, DIM)).astype("float32")

client = QdrantClient(path="/tmp/ai_infra_qdrant_lab")

def l2_topk(q, base, k):
    d = ((base - q) ** 2).sum(axis=1)
    return set(np.argsort(d)[:k].tolist())

# 暴力计算 ground truth（CPU 即可，仅 2 万条）
gt = [l2_topk(queries[i], data, TOPK) for i in range(NQ)]

# 对比不同 (m, ef_construct, ef_search) 组合
configs = [
    dict(m=8,  ef_construct=64,  ef_search=32),
    dict(m=16, ef_construct=100, ef_search=64),
    dict(m=32, ef_construct=200, ef_search=128),
]

print(f"{'m':>3} {'ef_con':>7} {'ef_srch':>8} {'recall@10':>10} {'p50(ms)':>9}")
for cfg in configs:
    coll = f"lab_m{cfg['m']}_efc{cfg['ef_construct']}"
    if client.collection_exists(coll):
        client.delete_collection(coll)
    # 建集合：HNSW 的 m / ef_construct 在 hnsw_config 指定
    client.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID),
        hnsw_config=HnswConfigDiff(m=cfg["m"], ef_construct=cfg["ef_construct"]),
    )
    client.upsert(
        collection_name=coll,
        points=[PointStruct(id=i, vector=data[i].tolist()) for i in range(N)],
    )

    hits, lat = 0, []
    for i in range(NQ):
        t0 = time.perf_counter()
        res = client.query_points(
            collection_name=coll,
            query=queries[i].tolist(),
            limit=TOPK,
            # ef_search 在查询时通过 hnsw_ef 控制
            search_params=SearchParams(hnsw_ef=cfg["ef_search"]),
        ).points
        lat.append((time.perf_counter() - t0) * 1000)
        got = {p.id for p in res}
        hits += len(got & gt[i])

    recall = hits / (NQ * TOPK)
    p50 = float(np.percentile(lat, 50))
    print(f"{cfg['m']:>3} {cfg['ef_construct']:>7} {cfg['ef_search']:>8} "
          f"{recall:>10.3f} {p50:>9.2f}")
