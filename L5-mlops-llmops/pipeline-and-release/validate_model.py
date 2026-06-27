# scripts/validate_model.py
import json
import sys
import pathlib

THRESHOLD = 0.80


def load(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def predict(text, weights):
    for kw, label in weights.items():
        if kw in text:
            return label
    return "positive"


def main():
    model = load("model.json")
    weights = model["weights"]
    eval_set = load("eval_set.json")

    assert weights, "模型权重为空，加载失败"
    out = predict("好评", weights)
    assert out in ("positive", "negative"), f"非法输出: {out}"
    print(f"[smoke] OK  model.version={model['version']}")

    correct = sum(predict(s["text"], weights) == s["label"] for s in eval_set)
    acc = correct / len(eval_set)
    print(f"[eval ] accuracy={acc:.2%}  threshold={THRESHOLD:.0%}")

    if acc < THRESHOLD:
        print(f"::error::精度 {acc:.2%} 低于门禁 {THRESHOLD:.0%}，阻断发布 ❌")
        sys.exit(1)
    print("[gate ] 通过质量门禁 ✅")


if __name__ == "__main__":
    main()
