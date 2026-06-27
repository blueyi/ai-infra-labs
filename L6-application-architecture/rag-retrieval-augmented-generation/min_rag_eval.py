import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- 0. 知识库（实际中来自切块后的文档） ----------
corpus = [
    "RAG 通过检索外部文档来约束 LLM 生成，缓解幻觉问题。",
    "向量检索用 embedding 的余弦相似度找语义相近的片段。",
    "Reranker 是 cross-encoder，对召回结果做二阶精排。",
    "BM25 是基于词频的稀疏检索，擅长精确关键词匹配。",
    "Chunking 切块策略直接影响检索召回的上限。",
    "HBM 显存带宽是 GPU 推理的主要瓶颈之一。",  # 干扰项：与 RAG 无关
]

model = SentenceTransformer("all-MiniLM-L6-v2")  # 小模型，CPU 友好
doc_emb = model.encode(corpus, normalize_embeddings=True)  # 已归一化

# ---------- 1. 检索：内存余弦 top-k ----------
def retrieve(query: str, k: int = 3):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = doc_emb @ q          # 归一化后点积 = 余弦相似度
    idx = np.argsort(-scores)[:k]
    return [(corpus[i], float(scores[i])) for i in idx]

# ---------- 2. 组装 prompt + mock LLM ----------
def mock_llm(query: str, contexts: list[str]) -> str:
    # 真实场景换成 ollama / OpenAI；这里抽取式生成，保证答案"忠于上下文"
    joined = " ".join(contexts)
    if "幻觉" in query or "约束" in query:
        return "RAG 通过检索外部文档约束 LLM 生成，从而缓解幻觉。"
    return joined[:80]  # 退化：直接返回上下文摘要

def rag_pipeline(query: str, k: int = 3):
    hits = retrieve(query, k)
    contexts = [c for c, _ in hits]
    answer = mock_llm(query, contexts)
    return answer, contexts, hits

# ---------- 3. RAGAS 风格离线指标（手写简化版） ----------
def cos(a: str, b: str) -> float:
    e = model.encode([a, b], normalize_embeddings=True)
    return float(e[0] @ e[1])

def context_precision(contexts, ground_truth):
    """召回上下文中与标准答案相关的比例（相关性 > 阈值算命中）"""
    rel = [1 if cos(c, ground_truth) > 0.4 else 0 for c in contexts]
    return sum(rel) / len(rel) if rel else 0.0

def answer_relevancy(query, answer):
    """生成答案与问题的语义相关度"""
    return cos(query, answer)

def faithfulness(answer, contexts):
    """答案是否"忠于"检索到的上下文（防幻觉的核心指标）"""
    return max(cos(answer, c) for c in contexts) if contexts else 0.0

# ---------- 4. 跑离线评估 ----------
eval_set = [
    {"q": "RAG 如何缓解幻觉？", "gt": "RAG 检索外部文档约束生成，缓解幻觉。"},
    {"q": "什么是 reranker？", "gt": "Reranker 是 cross-encoder，做二阶精排。"},
]

print(f"{'问题':<20}{'ctx_prec':>10}{'ans_rel':>10}{'faith':>10}")
for s in eval_set:
    ans, ctxs, _ = rag_pipeline(s["q"])
    cp = context_precision(ctxs, s["gt"])
    ar = answer_relevancy(s["q"], ans)
    fa = faithfulness(ans, ctxs)
    print(f"{s['q']:<20}{cp:>10.3f}{ar:>10.3f}{fa:>10.3f}")
