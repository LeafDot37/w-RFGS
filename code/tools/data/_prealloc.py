#!/usr/bin/env python
"""_prealloc.py —— 同进程显存包裹（非额外占位）。

共享卡上「程序在 ~RF_VRAM_GB 的包裹里跑、这包裹第一时间归我们、后到者抢不走」的实现：
在目标脚本（train.py）所在进程、任何大分配之前，先一次性 cudaMalloc 预占 RF_VRAM_GB GiB，
然后 del 释放给 PyTorch 缓存分配器 —— 该 segment 仍被本进程 cudaMalloc 物理持有（未 empty_cache、
不返还 CUDA），后续训练的所有分配都优先复用这 12GB，于是：
  - 本进程峰值 ≈ 预占量（不超包裹），不会在租户旁边再向 CUDA 争新内存；
  - 12GB 从第 0 秒就物理占有 → 晚到的租户只能去用剩下的，挤不掉我们。

用法：python _prealloc.py train.py -s ...   （argv[1:] 原样传给目标脚本）
环境：RF_VRAM_GB>0 走预占（同进程 runpy 执行目标）；
      未设/0 时直接 subprocess 透传 `python train.py ...`，行为与直接调用完全一致。
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os
import sys


def main() -> None:
    args = sys.argv[1:]
    gb = float(os.environ.get("RF_VRAM_GB", "0") or 0)
    if gb <= 0 or not args:
        # 透传：等效 `$PY train.py <args>`（无预占语义）
        import subprocess
        raise SystemExit(subprocess.call([sys.executable] + args))

    import torch

    # 预占落在 torch 默认设备(= "cuda" 解析到的当前设备): 与 train.py 自身选卡一致。
    # 不要解析 CUDA_VISIBLE_DEVICES 里的物理号再 set_device —— launch_full 设 CUDA_VISIBLE_DEVICES=<物理号>
    # 后 torch 把可见卡重编号为逻辑 0..n-1, 物理号已非法; 让 torch 默认选第一张可见卡即可。
    n = int(gb * (1024 ** 3) // 4)          # float32 元素数 = GB*1024^3/4
    buf = torch.empty(n, dtype=torch.float32, device="cuda")  # reserve segment
    del buf                                   # 回缓存池；segment 仍物理持有(不 empty_cache)
    reserved = torch.cuda.memory_reserved()   # 本进程真正 cudaMalloc 持有的量
    free, _total = torch.cuda.mem_get_info()  # 全卡空闲(含租户, 仅参考)
    dev = torch.cuda.current_device()         # 逻辑设备号
    print(f"[prealloc] RF_VRAM_GB={gb:g}GiB → 本进程持显存≈{reserved / 2**30:.2f}GiB "
          f"(cuda:{dev}), 全卡空闲 {free / 2**20:.0f}MiB —— 训练将在该包裹内复用",
          file=sys.stderr, flush=True)
    del reserved

    target = os.path.abspath(args[0])
    sys.argv = args                          # [0]=train.py 路径, [1:]=真实参数
    sys.path.insert(0, os.path.dirname(target))
    import runpy
    runpy.run_path(target, run_name="__main__")   # 同进程执行 → 预占随训练全程持有


if __name__ == "__main__":
    main()
