#!/usr/bin/env python3
"""L5.6 — OpenTelemetry RAG span tree (ConsoleSpanExporter)."""
import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
tracer = trace.get_tracer("rag.pipeline", "1.0.0")


def fake_vector_search(q: str, k: int = 5) -> list[str]:
    time.sleep(0.01)
    return [f"doc{i}" for i in range(k)]


def fake_llm(q: str, docs: list[str]) -> dict:
    time.sleep(0.02)
    return {"text": "answer", "prompt_tokens": 120, "completion_tokens": 40}


def rag_query(q: str) -> str:
    with tracer.start_as_current_span("rag.request") as root:
        root.set_attribute("user.query.length", len(q))
        with tracer.start_as_current_span("retrieve") as s:
            docs = fake_vector_search(q, k=5)
            s.set_attribute("retrieve.k", 5)
            s.set_attribute("retrieve.hits", len(docs))
        with tracer.start_as_current_span("llm.call") as s:
            s.set_attribute("gen_ai.system", "vllm")
            s.set_attribute("gen_ai.request.model", "TinyLlama-1.1B")
            resp = fake_llm(q, docs)
            s.set_attribute("gen_ai.usage.input_tokens", resp["prompt_tokens"])
            s.set_attribute("gen_ai.usage.output_tokens", resp["completion_tokens"])
        return resp["text"]


def main() -> int:
    for i in range(2):
        out = rag_query(f"question {i}")
        assert out == "answer"
    print("[ok] otel rag demo passed (see stderr for span JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
