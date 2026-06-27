import numpy as np
from sklearn.datasets import make_moons

np.random.seed(0)

# ---------- 1) 玩具数据集：两个交错的半月，线性不可分 ----------
X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
y = y.reshape(-1, 1).astype(np.float64)          # 形状 [N, 1]，二分类标签 0/1

# ---------- 2) 初始化两层网络参数 ----------
d_in, d_hid, d_out = 2, 16, 1
W1 = np.random.randn(d_in, d_hid) * np.sqrt(2.0 / d_in)   # He 初始化（配 ReLU）
b1 = np.zeros((1, d_hid))
W2 = np.random.randn(d_hid, d_out) * np.sqrt(1.0 / d_hid)
b2 = np.zeros((1, d_out))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

lr = 0.5
losses = []

for epoch in range(2000):
    # ===== ① Forward 前向（两次 GEMM + element-wise 激活）=====
    z1 = X @ W1 + b1            # GEMM: [N,2]·[2,16] -> [N,16]
    a1 = np.maximum(0, z1)      # ReLU（element-wise）
    z2 = a1 @ W2 + b2           # GEMM: [N,16]·[16,1] -> [N,1]
    a2 = sigmoid(z2)            # 输出概率

    # ===== ② Loss：二分类交叉熵（reduction，求和成标量）=====
    eps = 1e-8
    loss = -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))
    losses.append(loss)

    # ===== ③ Backward 反向：链式法则，逐层手推梯度 =====
    N = X.shape[0]
    # 交叉熵 + sigmoid 的组合梯度直接化简为 (a2 - y)
    dz2 = (a2 - y) / N          # [N,1]
    dW2 = a1.T @ dz2            # [16,1]  注意：梯度形状 == 参数形状
    db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ W2.T            # 把误差传回隐层 [N,16]
    dz1 = da1 * (z1 > 0)        # ReLU 的导数：z>0 处为 1，否则 0
    dW1 = X.T @ dz1             # [2,16]
    db1 = dz1.sum(axis=0, keepdims=True)

    # ===== ④ optimizer.step：朴素 SGD 参数更新 θ ← θ - lr·g =====
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if epoch % 200 == 0:
        acc = ((a2 > 0.5) == y).mean()
        print(f"epoch {epoch:4d} | loss {loss:.4f} | acc {acc:.3f}")

# ---------- 最终评估 ----------
pred = (sigmoid(np.maximum(0, X @ W1 + b1) @ W2 + b2) > 0.5)
print(f"final accuracy: {(pred == y).mean():.3f}")

# ---------- 画 loss 曲线 ----------
import matplotlib
matplotlib.use("Agg")          # 无显示环境也能保存图片
import matplotlib.pyplot as plt
plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("two-layer MLP on make_moons (pure NumPy)")
plt.savefig("loss_curve.png", dpi=120)
print("saved loss_curve.png")
