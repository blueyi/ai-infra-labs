#!/usr/bin/env python3
"""L1.10 — minimal Ray task (local head, no GPU)."""
import ray


@ray.remote(num_cpus=1)
def hello(name: str) -> str:
    return f"ray ok: {name}"


def main() -> int:
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False)

    refs = [hello.remote(f"worker-{i}") for i in range(3)]
    results = ray.get(refs)
    for r in results:
        print(r)
    assert all(r.startswith("ray ok:") for r in results)

    ray.shutdown()
    print("[ok] ray hello lab passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
