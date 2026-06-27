"""
uniir_demo.py —— 最小统一 IR 数据结构 + lowering 演示 + 算子覆盖率统计
纯标准库，CPU 可跑：python3 uniir_demo.py
"""
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class MemLevel(Enum):
    GLOBAL = "GLOBAL"
    SHARED = "SHARED"
    REG = "REG"


@dataclass
class UniTensor:
    name: str
    shape: tuple
    dtype: str = "f16"
    level: MemLevel = MemLevel.GLOBAL


@dataclass
class UniOp:
    kind: str
    inputs: List[UniTensor]
    output: UniTensor
    attrs: Dict = field(default_factory=dict)


class HAL:
    target = "abstract"

    def mem_alloc(self, t: UniTensor) -> str:
        raise NotImplementedError

    def compute(self, op: UniOp) -> str:
        raise NotImplementedError


class NvHAL(HAL):
    target = "nvidia"
    _LV = {MemLevel.GLOBAL: "HBM", MemLevel.SHARED: "SMEM", MemLevel.REG: "reg"}

    def mem_alloc(self, t):
        return f"// nv: alloc {t.name}{t.shape} on {self._LV[t.level]}"

    def compute(self, op):
        m = {"matmul": "mma.sync.tile", "relu": "vec.max(x,0)"}
        return f"// nv: {m.get(op.kind, op.kind)} -> {op.output.name}"


class AscendHAL(HAL):
    target = "ascend"
    _LV = {MemLevel.GLOBAL: "GM", MemLevel.SHARED: "UB", MemLevel.REG: "L0"}

    def mem_alloc(self, t):
        return f"// ascend: alloc {t.name}{t.shape} on {self._LV[t.level]}"

    def compute(self, op):
        m = {"matmul": "Cube.mmad", "relu": "Vector.relu"}
        return f"// ascend: {m.get(op.kind, op.kind)} -> {op.output.name}"


def lower(ops: List[UniOp], hal: HAL) -> List[str]:
    code = [f"=== lowering to [{hal.target}] ==="]
    seen = set()
    for op in ops:
        for t in op.inputs + [op.output]:
            if t.name not in seen:
                code.append(hal.mem_alloc(t))
                seen.add(t.name)
        code.append(hal.compute(op))
    return code


CORE_OP_SET = {
    "matmul", "conv", "relu", "reduce", "reshape", "softmax",
    "layernorm", "collective_allreduce",
}
ASCEND_NPU_IR = {"matmul", "conv", "relu", "reduce", "reshape", "softmax"}
CUTILE_IR = {"matmul", "relu", "softmax", "layernorm", "reshape", "reduce"}


def coverage(ir_ops: set, base: set = CORE_OP_SET) -> float:
    return len(ir_ops & base) / len(base) * 100


def report_coverage():
    print("=== 算子覆盖率评估（基准 = 核心算子集 %d 个）===" % len(CORE_OP_SET))
    for name, ops in [("AscendNPU IR", ASCEND_NPU_IR), ("cuTile IR", CUTILE_IR)]:
        miss = sorted(CORE_OP_SET - ops)
        print(f"  {name:14s}: {coverage(ops):5.1f}%  缺失={miss}")
    both = ASCEND_NPU_IR & CUTILE_IR
    print(f"  两者交集（统一 IR 首批稳态算子）: {sorted(both)}")


if __name__ == "__main__":
    x = UniTensor("X", (1024, 1024))
    w = UniTensor("W", (1024, 1024))
    h = UniTensor("H", (1024, 1024), level=MemLevel.SHARED)
    y = UniTensor("Y", (1024, 1024))
    graph = [
        UniOp("matmul", [x, w], h, attrs={"tile": (128, 128)}),
        UniOp("relu", [h], y),
    ]
    for hal in (NvHAL(), AscendHAL()):
        print("\n".join(lower(graph, hal)), "\n")
    report_coverage()
