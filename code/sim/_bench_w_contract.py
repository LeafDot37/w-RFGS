# -*- coding: utf-8 -*-
"""只测「把 (M²,N_tx,P) 按 w 收缩成 (M²,P)」这一步，不碰 Sionna。

为什么单独测：第一版 `np.einsum("mnp,n->mp", re, w)` 实测 1003 ms/张，占总成本的
60%。它沿**中间轴** n 求和，而 re 是 C 连续的 ⇒ 每读一个 m 都要跨 N_tx×P 个元素，
缓存全废。真正该做的是「一次转置成 (M²,P,N_tx)，再对 K 个 w 做一次 gemm」——
数据只过一遍，K 个波束几乎免费。

尺寸取实测值：M²=100、N_tx=16、P=158,835（inst 0 的实测路径数），float32。

用法::
    python code/sim/_bench_w_contract.py --K 4 --reps 2
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def timeit(fn, reps):
    fn()                                    # 预热（含首次页分配）
    best = 1e9
    for _ in range(reps):
        t = time.time()
        fn()
        best = min(best, time.time() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M2", type=int, default=100)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--P", type=int, default=158835)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    M2, N, P, K = args.M2, args.N, args.P, args.K
    rng = np.random.default_rng(0)
    re = rng.standard_normal((M2, N, P), dtype=np.float32)
    im = rng.standard_normal((M2, N, P), dtype=np.float32)
    W = (rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))).astype(
        np.complex64) * 0.25
    print(f"  {M2}×{N}×{P} float32 ×2 = "
          f"{2*re.nbytes/2**30:.2f} GiB/份  K={K}", flush=True)

    # 参考（float64，确认别的路子对得上）
    ref = [ (np.einsum("mnp,n->mp", re.astype(np.float64), W[k].real.astype(np.float64))
             - np.einsum("mnp,n->mp", im.astype(np.float64), W[k].imag.astype(np.float64))
             + 1j * (np.einsum("mnp,n->mp", re.astype(np.float64), W[k].imag.astype(np.float64))
                     + np.einsum("mnp,n->mp", im.astype(np.float64), W[k].real.astype(np.float64))))
            for k in range(K)]

    def check(name, outs, t, extra=""):
        err = max(float(np.max(np.abs(o - r))) / float(np.max(np.abs(r)))
                  for o, r in zip(outs, ref))
        print(f"  {name:34s} {t*1000:8.1f} ms / {K} 张 = {t/K*1000:6.1f} ms/张"
              f"   相对偏差 {err:.2e} {extra}", flush=True)

    # A：逐张 einsum（现基线）
    def A():
        out = []
        for k in range(K):
            wr, wi = W[k].real.astype(np.float32), W[k].imag.astype(np.float32)
            out.append(np.einsum("mnp,n->mp", re, wr) - np.einsum("mnp,n->mp", im, wi)
                       + 1j * (np.einsum("mnp,n->mp", re, wi)
                               + np.einsum("mnp,n->mp", im, wr)))
        return out
    check("A 逐张 einsum（现基线）", A(), timeit(A, args.reps))

    # B：optimize=True
    def B():
        out = []
        for k in range(K):
            wr, wi = W[k].real.astype(np.float32), W[k].imag.astype(np.float32)
            out.append(np.einsum("mnp,n->mp", re, wr, optimize=True)
                       - np.einsum("mnp,n->mp", im, wi, optimize=True)
                       + 1j * (np.einsum("mnp,n->mp", re, wi, optimize=True)
                               + np.einsum("mnp,n->mp", im, wr, optimize=True)))
        return out
    check("B einsum optimize=True", B(), timeit(B, args.reps))

    # C：转置一次 + 单次 gemm（K 个波束一起）
    def C():
        A_ = np.empty((M2, P, 2 * N), np.float32)
        A_[..., :N] = re.transpose(0, 2, 1)
        A_[..., N:] = im.transpose(0, 2, 1)
        Wc = np.zeros((2 * N, 2 * K), np.float32)
        for k in range(K):
            wr, wi = W[k].real.astype(np.float32), W[k].imag.astype(np.float32)
            Wc[:N, 2 * k] = wr
            Wc[N:, 2 * k] = -wi
            Wc[:N, 2 * k + 1] = wi
            Wc[N:, 2 * k + 1] = wr
        o = A_.reshape(M2 * P, 2 * N) @ Wc
        o = o.reshape(M2, P, 2, K)
        return [o[:, :, 0, k] + 1j * o[:, :, 1, k] for k in range(K)]
    check("C 转置+单次 gemm", C(), timeit(C, args.reps))

    # D：分块 einsum（块内数据能进缓存）
    def D(chunk=1 << 14):
        out = [np.zeros((M2, P), np.complex64) for _ in range(K)]
        for s in range(0, P, chunk):
            e = min(s + chunk, P)
            rc = re[:, :, s:e]
            ic = im[:, :, s:e]
            for k in range(K):
                wr, wi = W[k].real.astype(np.float32), W[k].imag.astype(np.float32)
                out[k][:, s:e] = (np.einsum("mnp,n->mp", rc, wr)
                                  - np.einsum("mnp,n->mp", ic, wi)
                                  + 1j * (np.einsum("mnp,n->mp", rc, wi)
                                          + np.einsum("mnp,n->mp", ic, wr)))
        return out
    check("D 分块 einsum (16k)", D(), timeit(D, args.reps))

    # E：转置一次 + 逐张 gemv（比 C 多读 K 遍矩阵，看看差多少）
    def E():
        Rt = np.ascontiguousarray(re.transpose(0, 2, 1)).reshape(M2 * P, N)
        It = np.ascontiguousarray(im.transpose(0, 2, 1)).reshape(M2 * P, N)
        out = []
        for k in range(K):
            wr, wi = W[k].real.astype(np.float32), W[k].imag.astype(np.float32)
            out.append((Rt @ wr - It @ wi + 1j * (Rt @ wi + It @ wr)).reshape(M2, P))
        return out
    check("E 转置+逐张 gemv", E(), timeit(E, args.reps))


if __name__ == "__main__":
    main()
