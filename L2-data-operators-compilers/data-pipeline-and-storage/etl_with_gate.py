# etl_with_gate.py —— 抽取→脱敏→质量断言；不达标即非零退出（CI 门禁）
import sys
import hashlib
import pandas as pd

SRC = "data/users.csv"
DST = "data/users_clean.csv"

# ---------- Extract ----------
df = pd.read_csv(SRC)

# ---------- Transform：PII 脱敏（email 不可逆哈希）----------
def mask_email(x: str) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]

df["email"] = df["email"].map(mask_email)

# ---------- Quality Gate：质量断言（行数 / 空值 / 范围 / Schema）----------
def quality_gate(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    expected_cols = {"id", "age", "country", "email"}

    # 1) Schema 检查：列集合必须匹配
    if set(frame.columns) != expected_cols:
        errors.append(f"schema 不匹配: {set(frame.columns)} != {expected_cols}")

    # 2) 行数下限：少于 3 行视为抽取异常
    if len(frame) < 3:
        errors.append(f"行数不足: {len(frame)} < 3")

    # 3) 关键列非空：id / age 不允许空值
    for col in ("id", "age"):
        n_null = frame[col].isna().sum()
        if n_null > 0:
            errors.append(f"列 {col} 存在 {n_null} 个空值")

    # 4) 数值范围：age 必须落在 [0, 120]
    if "age" in frame.columns:
        bad = frame[(frame["age"] < 0) | (frame["age"] > 120)]
        if len(bad) > 0:
            errors.append(f"age 越界行数: {len(bad)}")

    return errors


errors = quality_gate(df)
if errors:
    print("❌ 质量门禁未通过，阻断管道：", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)  # 非零退出码 → CI/调度器感知失败

# ---------- Load ----------
df.to_csv(DST, index=False)
print(f"✅ 质量门禁通过，已写出 {DST}（{len(df)} 行，email 已脱敏）")
