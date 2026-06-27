# chip_compare.py —— 纯 CPU 运行，零三方依赖
# 数据为公开 datasheet 量级口径，以官方为准，可能随版本变化。

# 价格为公开市场参考量级（美元），仅用于演示性价比计算口径，非报价。
# 型号与数字为 2026 视角的量级口径，以官方为准。云厂自研 ASIC（如 TPU v7）无
# 公开零售价，不纳入性价比脚本，仅在上方对比矩阵中作为参考。
CHIPS = {
    #              fp16_tflops, hbm_gb, bw_tbs, link_gbs, tdp_w, price_usd
    "B200":        dict(fp16=2250, hbm=192, bw=8.0,  link=1800, tdp=1000, price=35000),
    "B300-Ultra":  dict(fp16=2500, hbm=288, bw=8.0,  link=1800, tdp=1400, price=45000),
    "Ascend-910C": dict(fp16=800,  hbm=128, bw=3.2,  link=600,  tdp=450,  price=20000),
    "MI355X":      dict(fp16=2500, hbm=288, bw=8.0,  link=1075, tdp=1400, price=30000),
    "MI300X":      dict(fp16=1300, hbm=192, bw=5.3,  link=896,  tdp=750,  price=18000),
}

def enrich(name, c):
    # 算术强度阈值 = 峰值算力(FLOP/s) / 带宽(Byte/s)，单位 FLOP/Byte
    # fp16 TFLOPS = 1e12 FLOP/s；bw TB/s = 1e12 Byte/s，故比值可直接用 fp16/bw
    ai_threshold = c["fp16"] / (c["bw"] * 1000)   # *1000: TFLOPS / (TB/s)
    return dict(
        name=name,
        fp16=c["fp16"],
        hbm=c["hbm"],
        bw=c["bw"],
        # 每 TFLOPS 成本（美元/TFLOPS）：越低越划算
        cost_per_tflops=round(c["price"] / c["fp16"], 2),
        # 每 GB HBM 成本（美元/GB）：推理大模型时更关键
        cost_per_gb=round(c["price"] / c["hbm"], 1),
        # 算术强度阈值（FLOP/Byte）：越低越偏「带宽型」，越擅长 memory-bound 的 decode
        ai_threshold=round(ai_threshold, 1),
        # 每瓦算力（TFLOPS/W）：能效，数据中心 TCO 关键
        tflops_per_watt=round(c["fp16"] / c["tdp"], 2),
    )

rows = [enrich(n, c) for n, c in CHIPS.items()]

def show(title, key, reverse=True, unit=""):
    print(f"\n=== 按 {title} 排序 ===")
    for r in sorted(rows, key=lambda x: x[key], reverse=reverse):
        print(f"  {r['name']:<14} {r[key]:>8}{unit}  "
              f"(FP16={r['fp16']}TFLOPS, HBM={r['hbm']}GB, BW={r['bw']}TB/s)")

# 1) 性价比：每 TFLOPS 成本，越低越好（升序）
show("每 TFLOPS 成本（越低越划算）", "cost_per_tflops", reverse=False, unit=" $/TFLOPS")
# 2) 每 GB HBM 成本，越低越好（升序）—— 大模型推理选型关键
show("每 GB HBM 成本（推理大模型关键）", "cost_per_gb", reverse=False, unit=" $/GB")
# 3) 算术强度阈值，越低越偏带宽型（升序）
show("算术强度阈值（越低越擅长 decode/memory-bound）", "ai_threshold",
     reverse=False, unit=" FLOP/Byte")
# 4) 能效：每瓦算力，越高越好（降序）
show("每瓦算力（能效/TCO）", "tflops_per_watt", reverse=True, unit=" TFLOPS/W")
