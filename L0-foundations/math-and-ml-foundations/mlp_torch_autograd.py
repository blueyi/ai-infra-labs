import torch, numpy as np
from sklearn.datasets import make_moons

dev = "cuda" if torch.cuda.is_available() else "cpu"
X_np, y_np = make_moons(400, noise=0.2, random_state=0)
X = torch.tensor(X_np, dtype=torch.float32, device=dev)
y = torch.tensor(y_np, dtype=torch.float32, device=dev).reshape(-1, 1)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.ReLU(),
    torch.nn.Linear(16, 1), torch.nn.Sigmoid(),
).to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.5)
loss_fn = torch.nn.BCELoss()

for epoch in range(2000):
    opt.zero_grad()
    loss = loss_fn(net(X), y)   # forward + loss
    loss.backward()             # backward：autograd 自动算我们手推的那些梯度
    opt.step()                  # optimizer.step
print(f"[{dev}] final loss {loss.item():.4f}")
