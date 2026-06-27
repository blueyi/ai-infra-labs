import re
import operator

# ---------- 工具层（确定性，可被 Agent 调用）----------
def calculator(expr: str) -> str:
    """安全的四则运算计算器工具。"""
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expr or ""):
        return "ERROR: 非法表达式"
    try:
        # 受限 eval：仅允许数字与四则运算字符
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"ERROR: {e}"

TOOLS = {"calc": calculator}

# ---------- Mock LLM：用规则模拟「模型决策」----------
# 真实场景这里换成 ollama / OpenAI 调用，返回同样格式的决策。
def mock_llm(goal: str, history: list) -> dict:
    """
    根据任务与历史轨迹，返回下一步决策：
      {"type": "action", "tool": "calc", "input": "..."}  或
      {"type": "final", "answer": "..."}
    模拟「先拆解、逐步计算、最后汇总」的多步推理。
    """
    nums = re.findall(r"-?\d+", goal)
    steps_done = [h for h in history if h["role"] == "observation"]

    # 任务1：求和 a..b 区间内整数和 -> 用公式分步算
    if "区间和" in goal and len(nums) >= 2:
        a, b = int(nums[0]), int(nums[1])
        if len(steps_done) == 0:
            return {"type": "action", "tool": "calc", "input": f"({a}+{b})*({b}-{a}+1)/2"}
        last = steps_done[-1]["content"]
        if last.startswith("ERROR"):
            # 反思：换一种不带除法精度问题的表达式
            return {"type": "action", "tool": "calc", "input": f"({a}+{b})*({b}-{a}+1)//2"}
        return {"type": "final", "answer": str(int(float(last)))}

    # 任务2：复利 本金*(1+r)^n -> 需要分两步（DAG 只算一步会失败）
    if "复利" in goal and len(nums) >= 3:
        p, r_pct, n = int(nums[0]), int(nums[1]), int(nums[2])
        factor = f"(1+{r_pct}/100)"
        if len(steps_done) == 0:
            return {"type": "action", "tool": "calc", "input": f"{factor}**{n}"}
        if len(steps_done) == 1:
            growth = steps_done[-1]["content"]
            return {"type": "action", "tool": "calc", "input": f"{p}*{growth}"}
        return {"type": "final", "answer": f"{float(steps_done[-1]['content']):.2f}"}

    return {"type": "final", "answer": "无法处理的任务"}

# ---------- Agentic 流程：ReAct 循环 + 反思 ----------
def run_agentic(goal: str, max_steps: int = 6) -> str:
    history = []
    for _ in range(max_steps):
        decision = mock_llm(goal, history)
        if decision["type"] == "final":
            return decision["answer"]
        history.append({"role": "thought", "content": f"调用 {decision['tool']}"})
        obs = TOOLS[decision["tool"]](decision["input"])
        history.append({"role": "observation", "content": obs})
        # 反思边：观测出错则记录，下一轮 mock_llm 会读到并改策略
    return "FAIL: 超出最大步数"

# ---------- DAG 流程：固定单步「检索->计算->输出」，无反思无多轮 ----------
def run_dag(goal: str) -> str:
    # 静态流程假设「一步计算即得答案」，无法应对需要多步的任务
    decision = mock_llm(goal, history=[])
    if decision["type"] == "final":
        return decision["answer"]
    obs = TOOLS[decision["tool"]](decision["input"])
    # DAG 到此结束：直接把第一次工具结果当答案（不再回流、不再多步）
    try:
        return str(int(float(obs)))
    except Exception:
        return f"FAIL: {obs}"

# ---------- 评测：一组任务上对比成功率 ----------
TASKS = [
    ("求 1 到 100 的区间和", "5050"),       # 单步可解，DAG/Agentic 都行
    ("求 5 到 50 的区间和", "1265"),         # 单步可解
    ("本金 1000 利率 5 复利 3 年的复利终值", "1157.63"),  # 需两步，DAG 会失败
    ("本金 2000 利率 10 复利 2 年的复利终值", "2420.00"), # 需两步，DAG 会失败
]

def grade(name, fn):
    ok = 0
    for goal, gold in TASKS:
        got = fn(goal)
        passed = got == gold
        ok += passed
        print(f"  [{name}] {goal[:20]:<22} 期望={gold:<9} 得到={got:<9} {'✅' if passed else '❌'}")
    print(f"  >>> {name} 成功率: {ok}/{len(TASKS)} = {ok/len(TASKS):.0%}\n")

if __name__ == "__main__":
    print("== DAG 静态流程 ==")
    grade("DAG", run_dag)
    print("== Agentic 动态流程（ReAct+反思）==")
    grade("Agentic", run_agentic)
