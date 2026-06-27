# awq_quant.py —— 对小模型做 AWQ INT4 量化并测显存/时延
import time, torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
quant_path = "qwen0.5b-awq"

tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# AWQ 量化配置：4bit、group=128、GEMM kernel
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tok, quant_config=quant_config)   # 用内置校准集做激活感知量化
model.save_quantized(quant_path)
tok.save_pretrained(quant_path)
print(f"[AWQ] 量化完成 → {quant_path}")
