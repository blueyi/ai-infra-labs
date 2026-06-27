# bench.py —— 通用基准：显存峰值 + 平均 token 时延
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark(path, label, n_tokens=128):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype="auto", device_map="cuda", trust_remote_code=True)
    torch.cuda.reset_peak_memory_stats()
    prompt = "用三句话解释什么是投机采样。"
    ids = tok(prompt, return_tensors="pt").to("cuda")

    # 预热
    _ = model.generate(**ids, max_new_tokens=8)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = model.generate(**ids, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    gen = out.shape[-1] - ids["input_ids"].shape[-1]
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{label}] 显存峰值={mem:.2f}GB  生成{gen}token  "
          f"TPOT={dt/gen*1000:.1f}ms/token  吞吐={gen/dt:.1f}tok/s")

if __name__ == "__main__":
    benchmark("Qwen/Qwen2.5-0.5B-Instruct", "FP16 baseline")
    benchmark("qwen0.5b-awq",               "AWQ-INT4")
    # benchmark("qwen0.5b-gptq",            "GPTQ-INT4")  # GPTQ 量化产物
