# ppl.py —— 在 wikitext 上测 perplexity，量化前后对比精度损失
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def perplexity(path, label):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if "awq" in path.lower():
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_pretrained(path, device_map="cuda", trust_remote_code=True).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype="auto", device_map="cuda", trust_remote_code=True,
        ).eval()
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(data["text"])[:8000]
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")[:, :2048]
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    print(f"[{label}] perplexity = {torch.exp(loss).item():.3f}")

if __name__ == "__main__":
    perplexity("Qwen/Qwen2.5-0.5B-Instruct", "FP16")
    perplexity("qwen0.5b-awq",  "AWQ-INT4")
    import os
    if os.path.isdir("qwen0.5b-gptq"):
        perplexity("qwen0.5b-gptq", "GPTQ-INT4")
    else:
        print("[GPTQ-INT4] skip — run gptq_quant.py first to create qwen0.5b-gptq/")
