#!/usr/bin/env python3
"""
multi_region_dr_calc.py
双地域容灾 RPO/RTO 估算器。
输入：训练 checkpoint 间隔、跨域复制延迟、各阶段切换耗时。
输出：RPO/RTO 估算 + 主备/双活方案对比。
本地模拟用：把下方常量替换为云监控实测值即可上生产评估。
"""
from dataclasses import dataclass


@dataclass
class DRParams:
    # —— RPO 相关（训练进度新鲜度）——
    checkpoint_interval_min: float   # checkpoint 间隔(分钟)
    replication_lag_min: float       # 跨域异步复制延迟(分钟)
    # —— RTO 相关（切换四段耗时, 单位: 秒）——
    detect_interval_s: float         # 单次健康探测周期
    detect_fail_count: int           # 连续失败几次才判定故障
    decision_s: float                # 决策耗时(自动≈0, 人工较大)
    reroute_s: float                 # GSLB 改路由 + DNS/Anycast 收敛
    warmup_s: float                  # 备区拉起+模型加载+缓存预热
    standby_mode: str                # "active-standby" | "active-active"


def estimate_rpo(p: DRParams) -> float:
    """最坏情况丢失的训练进度(分钟) = 上个 checkpoint 之后 + 尚未复制到备区的部分。"""
    return p.checkpoint_interval_min + p.replication_lag_min


def estimate_rto(p: DRParams) -> float:
    """恢复服务耗时(秒) = 检测 + 决策 + 重路由 + 预热。双活省去备区拉起预热。"""
    detect = p.detect_interval_s * p.detect_fail_count
    warmup = 0.0 if p.standby_mode == "active-active" else p.warmup_s
    return detect + p.decision_s + p.reroute_s + warmup


def report(name: str, p: DRParams) -> None:
    rpo = estimate_rpo(p)
    rto = estimate_rto(p)
    print(f"[{name:<16}] mode={p.standby_mode:<14} "
          f"RPO≈{rpo:6.1f} min   RTO≈{rto:6.1f} s ({rto/60:.1f} min)")


if __name__ == "__main__":
    # 方案 A：主备 + 人工决策 + 大间隔 checkpoint（省钱但慢）
    standby = DRParams(
        checkpoint_interval_min=30, replication_lag_min=5,
        detect_interval_s=10, detect_fail_count=3, decision_s=120,
        reroute_s=60, warmup_s=300, standby_mode="active-standby",
    )
    # 方案 B：双活 + 自动决策 + 高频 checkpoint（贵但快）
    active = DRParams(
        checkpoint_interval_min=5, replication_lag_min=1,
        detect_interval_s=5, detect_fail_count=2, decision_s=0,
        reroute_s=20, warmup_s=0, standby_mode="active-active",
    )
    report("主备/省成本", standby)
    report("双活/低 RTO", active)
