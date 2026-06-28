#!/usr/bin/env python3
"""L2.9 — mini streaming embed pipeline (queue + batch worker, no Kafka)."""
import hashlib
import queue
import threading
import time

BATCH_SIZE = 4


def embed_batch(batch: list[dict]) -> list[dict]:
    return [
        {"doc_id": d["doc_id"], "vec": hashlib.sha256(d["text"].encode()).hexdigest()[:8]}
        for d in batch
    ]


def main() -> int:
    docs_q: queue.Queue = queue.Queue()
    results: list[dict] = []

    def producer() -> None:
        for i in range(20):
            docs_q.put({"doc_id": f"d{i}", "text": f"content {i}"})
            time.sleep(0.02)
        docs_q.put(None)

    def consumer() -> None:
        batch: list[dict] = []
        while True:
            item = docs_q.get()
            if item is None:
                if batch:
                    results.extend(embed_batch(batch))
                break
            batch.append(item)
            if len(batch) >= BATCH_SIZE:
                results.extend(embed_batch(batch))
                batch = []

    t0 = threading.Thread(target=producer, daemon=True)
    t1 = threading.Thread(target=consumer, daemon=True)
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    assert len(results) == 20, len(results)
    print(f"embedded={len(results)} sample={results[:2]}")
    print("[ok] mini stream embed lab passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
