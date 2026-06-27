# 伪代码示意 cuFile 直达路径（需 kvikio / cufile 绑定）
import cupy as cp
import kvikio                      # RAPIDS 的 cuFile Python 封装

f = kvikio.CuFile(CKPT, "r")       # 打开支持 GDS 的文件句柄
buf = cp.empty(big.numel(), dtype=cp.float32)  # 直接分配 GPU 显存
f.read(buf)                        # NVMe → GPU 显存 DMA，绕过 CPU bounce buffer
f.close()
