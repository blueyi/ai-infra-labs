import torch._dynamo as dynamo

with torch.no_grad():
    explanation = dynamo.explain(model)(x)
print(explanation)  # 打印 graph count / graph break count / 原因
