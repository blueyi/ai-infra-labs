# gptq_quant.py —— 对同一模型做 GPTQ INT4 量化
# 依赖：optimum + gptqmodel（Transformers 5.x 已弃用 auto-gptq）
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.gptq import GPTQQuantizer
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
out_dir = "qwen0.5b-gptq"

tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
)

# GPTQ：用 wikitext2 校准集算 Hessian、逐列量化补偿误差（backend=gptqmodel）
quantizer = GPTQQuantizer(bits=4, dataset="wikitext2", group_size=128, desc_act=False)
qmodel = quantizer.quantize_model(model, tok)
qmodel.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print(f"[GPTQ] 量化完成 → {out_dir}")
