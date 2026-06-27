# gptq_quant.py —— 对同一模型做 GPTQ INT4 量化
from transformers import AutoTokenizer
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                             device_map="auto", trust_remote_code=True)

# GPTQ：用 wikitext 校准集算 Hessian、逐列量化补偿误差
quantizer = GPTQQuantizer(bits=4, dataset="wikitext2", group_size=128, desc_act=False)
qmodel = quantizer.quantize_model(model, tok)
qmodel.save_pretrained("qwen0.5b-gptq")
tok.save_pretrained("qwen0.5b-gptq")
print("[GPTQ] 量化完成 → qwen0.5b-gptq")
