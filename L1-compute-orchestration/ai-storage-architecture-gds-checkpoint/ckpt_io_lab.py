import os
import time
import threading
import torch

CKPT = "/tmp/ckpt_demo.pt"
# 构造一个 ~2GB 的大 tensor 模拟 checkpoint（按内存调整规模）
big = torch.randn(512, 1024, 1024)  # 512 * 1M * 4B ≈ 2GB

def timeit(label, fn):
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    print(f"[{label}] {dt:.3f} s")
    return dt

# ---------- ① 同步写：阻塞直到落盘 ----------
def sync_save():
    torch.save(big, CKPT)
    # 强制刷到磁盘，排除 page cache 的「假完成」
    fd = os.open(CKPT, os.O_RDWR)
    os.fsync(fd)
    os.close(fd)

# ---------- ② 异步写：后台线程写盘，主线程立即返回 ----------
def async_save():
    # 真实场景应先拷到 pinned host buffer，这里用 clone 模拟「快照已暂存」
    snapshot = big.clone()
    def _bg():
        torch.save(snapshot, CKPT + ".async")
        fd = os.open(CKPT + ".async", os.O_RDWR); os.fsync(fd); os.close(fd)
    th = threading.Thread(target=_bg, daemon=True)
    th.start()
    return th  # 主线程「训练」可继续，稍后 join

timeit("sync_save  (阻塞)", sync_save)

t_main0 = time.perf_counter()
th = async_save()
print(f"[async_save 主线程返回] {time.perf_counter()-t_main0:.3f} s  ← 训练可立即继续")
th.join()  # 演示用：等后台写完

# ---------- ③ 冷缓存 vs 热缓存读取 ----------
def read_all(advise_dontneed=False):
    fd = os.open(CKPT, os.O_RDONLY)
    if advise_dontneed:
        # 主动告诉内核「这块数据用不上」，逼其放弃缓存 → 模拟冷读
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    data = b""
    while chunk := os.read(fd, 64 * 1024 * 1024):
        data += chunk
    os.close(fd)

# 冷读（建议丢弃缓存）：真实磁盘 I/O
timeit("read 冷缓存 (FADV_DONTNEED)", lambda: read_all(advise_dontneed=True))
# 热读（数据已在 page cache）：几乎全内存命中
timeit("read 热缓存 (page cache 命中)", lambda: read_all(advise_dontneed=False))
