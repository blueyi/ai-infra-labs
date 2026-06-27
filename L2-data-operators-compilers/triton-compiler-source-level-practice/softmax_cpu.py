# 紧接 3.2 的 __main__，在 NVIDIA GPU 路径下执行
if dev == "cuda":
    print("=" * 30, "TTIR", "=" * 30)
    print(kernel.asm["ttir"])      # ① Triton IR（硬件无关）

    print("=" * 30, "TTGIR", "=" * 30)
    print(kernel.asm["ttgir"])     # ② Triton GPU IR（带 layout 编码）

    print("=" * 30, "PTX", "=" * 30)
    print(kernel.asm["ptx"][:2000])  # ③ PTX（截前 2000 字符即可）
