"""NF4 fallback when auto-gptq is unavailable — still demonstrates INT4 memory savings."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "qwen0.5b-bnb4"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=cfg, device_map="cuda", trust_remote_code=True,
    )
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"[BNB-NF4] saved → {OUT}")

    torch.cuda.reset_peak_memory_stats()
    ids = tok("hello", return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    model.generate(**ids, max_new_tokens=32)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[BNB-NF4] peak_mem={mem:.2f}GB generate_32tok={1000*(time.perf_counter()-t0):.0f}ms")


if __name__ == "__main__":
    main()
