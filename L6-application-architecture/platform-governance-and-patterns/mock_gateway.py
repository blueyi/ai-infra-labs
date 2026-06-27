"""mock_gateway.py —— 多租户推理网关：RBAC + 配额 + 审计（纯 CPU）"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------- 1. 策略定义（对应 tenant-policy.yaml）----------
TENANTS = {
    "tenant-a": {"max_requests": 3, "roles": {"reader", "invoker"},
                 "allowed_models": {"llama-7b"}},
    "tenant-b": {"max_requests": 5, "roles": {"reader"},
                 "allowed_models": {"llama-13b"}},
}
RBAC = {                                  # 角色 -> 允许动作（最小权限）
    "reader":  {"list_models"},
    "invoker": {"list_models", "invoke"},
}

class AccessDenied(Exception):
    """所有越权都抛它，便于网关统一拒绝并审计"""

@dataclass
class Gateway:
    usage: dict = field(default_factory=dict)   # per-tenant 配额计数器
    audit_log: list = field(default_factory=list)

    def _audit(self, tenant, action, resource, decision, reason=""):
        self.audit_log.append({
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "tenant": tenant, "action": action, "resource": resource,
            "decision": decision, "reason": reason,
        })

    def handle(self, tenant_id: str, action: str, model: str):
        # 关卡 0：租户是否存在
        cfg = TENANTS.get(tenant_id)
        if cfg is None:
            self._audit(tenant_id, action, model, "DENY", "unknown tenant")
            raise AccessDenied(f"未知租户 {tenant_id}")

        # 关卡 1：RBAC —— 该租户的角色集合里有没有人允许这个 action
        permitted = set().union(*(RBAC.get(r, set()) for r in cfg["roles"]))
        if action not in permitted:
            self._audit(tenant_id, action, model, "DENY", "RBAC: 角色无此权限")
            raise AccessDenied(f"{tenant_id} 的角色无权执行 {action}")

        # 关卡 2：资源隔离 —— 只能访问本租户被授权的模型（跨租户访问拦截）
        if action == "invoke" and model not in cfg["allowed_models"]:
            self._audit(tenant_id, action, model, "DENY", "跨租户/越权模型访问")
            raise AccessDenied(f"{tenant_id} 无权访问模型 {model}")

        # 关卡 3：配额 —— 硬配额计数（吵闹邻居防护）
        if action == "invoke":
            used = self.usage.get(tenant_id, 0)
            if used >= cfg["max_requests"]:
                self._audit(tenant_id, action, model, "DENY", "超出配额(429)")
                raise AccessDenied(f"{tenant_id} 配额已耗尽 ({used}/{cfg['max_requests']})")
            self.usage[tenant_id] = used + 1

        # 全部通过：执行（mock）并审计 ALLOW
        self._audit(tenant_id, action, model, "ALLOW")
        return f"[OK] {tenant_id} 执行 {action} on {model}"


# ---------- 2. 越权测试：构造非法请求验证全部被拒 ----------
def expect_deny(fn, label):
    try:
        fn()
        print(f"❌ 安全漏洞！本应拒绝却放行：{label}")
    except AccessDenied as e:
        print(f"✅ 正确拒绝 [{label}]：{e}")

if __name__ == "__main__":
    gw = Gateway()

    # 正常请求：tenant-a 是 invoker，调自己的 llama-7b
    print(gw.handle("tenant-a", "invoke", "llama-7b"))
    print(gw.handle("tenant-a", "invoke", "llama-7b"))
    print(gw.handle("tenant-a", "invoke", "llama-7b"))

    print("\n--- 越权测试 ---")
    # 越权 1：超配额（tenant-a 第 4 次 invoke，max=3）
    expect_deny(lambda: gw.handle("tenant-a", "invoke", "llama-7b"), "超配额")
    # 越权 2：跨租户访问（tenant-a 调 tenant-b 的 llama-13b）
    expect_deny(lambda: gw.handle("tenant-a", "invoke", "llama-13b"), "跨租户模型访问")
    # 越权 3：RBAC 不足（tenant-b 只有 reader，无权 invoke）
    expect_deny(lambda: gw.handle("tenant-b", "invoke", "llama-13b"), "RBAC 权限不足")
    # 越权 4：未知租户
    expect_deny(lambda: gw.handle("ghost", "list_models", "llama-7b"), "未知租户")

    print("\n--- 审计日志（DENY 是一等公民）---")
    for r in gw.audit_log:
        print(f"{r['ts']} | {r['tenant']:9} | {r['action']:11} | "
              f"{r['resource']:9} | {r['decision']:5} | {r['reason']}")
