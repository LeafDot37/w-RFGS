"""make_count_matched_ckpt.py —— 把一份 3DGS 的 chkpnt 均匀抽样到指定点数。

为什么需要它（**「协方差初始化」实验的头号解释风险**）
====================================================
物理阶段实测：种协方差那一臂终态 **1,648,201** 个高斯，不种那一臂 **1,265,903** 个
—— 差 **+30.3%**，而同一份输入重跑的抖动只有 0.41%。机制是各向异性扁盘把 3DGS 的
致密化（clone / split）推得更狠：尺度中位没变（16 mm），各向异性比 p50 从 1.000
变成 4.140，终态点数就多三成。

问题在下一段：RRF 阶段在**冻结**点集上训，A× 标量头的逐点 embedding 是 P×32
⇒ **点数差就是参数量差**。于是「协方差的形状先验有用」与「点更多所以容量更大」
分不开，指标变好也读不出是哪一半。

本工具做**计数配平**：把处理臂的 chkpnt 均匀抽到对照臂的点数，得到一根
「同点数、不同协方差历史」的臂。它不能把两件事完全正交（抽样同时改了密度），
但这是唯一便宜可行的拆法，不跑它就只能把结论停在「分不开」。

为什么配平的是 **chkpnt** 而不是 ply
-----------------------------------
`train_rrf_scalar.py:load_visual_geometry` 是从 `--start_checkpoint` 解包
`model_params` 拿几何的（xyz / scaling / rotation / opacity / SH），**不读 ply**。
所以 RRF 的起点是 chkpnt，配平必须落在 chkpnt 上。（`render_rrf_scalar.py` 走的是
训练产出的 ply，那是下游自己的产物，与本工具无关。）

`_opt_dict` 怎么处理（**这里有个刻意选择，必须留着**）
----------------------------------------------------
`capture()` 第 11 项是 `self.optimizer.state_dict()`，其 `state` 以**优化器参数枚举
顺序**为键 —— 逐高斯子集化它需要重建整个 Adam 状态，既脆又没意义：
`load_visual_geometry` 就把这一项丢进 `_`，**根本不读**。
所以本工具把它写成 `None`，并：
  * 文件名刻意用非标准的 `chkpnt30000_计数配平.pth`（不带 `chkpnt<6位数字>.pth`
    的模式），不会被 `--checkpoint_iterations` 的产物或任何「按文件名找末档」的
    脚本误认；
  * 写回后重新 load 一遍，把**每一个**张量的形状打出来自证 —— 不靠「我写对了」。

用法
----
    <PY> tools/make_count_matched_ckpt.py \
        --src <处理臂>/chkpnt30000.pth --target 1265903 --seed 0 \
        --out <TOP>/点云/chkpnt30000_处理臂计数配平.pth
退出码 0 = 成功；非 0 = 拒绝（参数不合理 / 形状不符），**绝不静默写半成品**。
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# model_params 的 12 项（与 GaussianModel.capture 逐字对应）
PER_GAUSSIAN = ["xyz", "f_dc", "f_rest", "scaling", "rotation", "opacity",
                "max_radii2D", "xyz_grad_acc", "denom"]
SCALAR = ["active_sh_degree", "spatial_lr_scale"]
LAYOUT = (["active_sh_degree"] + PER_GAUSSIAN[:7] + ["xyz_grad_acc", "denom"]
          + ["_opt_dict", "spatial_lr_scale"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 chkpnt（3DGS 的 (model_params, iter) 元组）")
    ap.add_argument("--target", type=int, required=True, help="目标点数")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="|N_src − target| / target 超过此值就拒绝（默认 2%%）")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        raise SystemExit(f"[配平] 源 chkpnt 不存在：{src}")

    ckpt = torch.load(src, map_location="cpu")
    if not (isinstance(ckpt, (tuple, list)) and len(ckpt) == 2):
        raise SystemExit(f"[配平] {src} 不是 (model_params, first_iter) 元组，拒绝继续")
    mp, first_iter = ckpt
    if len(mp) != len(LAYOUT):
        raise SystemExit(f"[配平] model_params 有 {len(mp)} 项，期望 {len(LAYOUT)} "
                         f"（capture 的布局变了 ⇒ 本工具的先验过期，停下来重看）")
    d = dict(zip(LAYOUT, mp))

    n_src = int(d["xyz"].shape[0])
    rel = abs(n_src - args.target) / float(args.target)
    print(f"  源 {src.name}  N={n_src}  目标 N={args.target}  相对差 {rel:.2%}  "
          f"first_iter={first_iter}")
    if rel <= args.tol:
        raise SystemExit(f"[配平] 源点数与目标相差 {rel:.2%} ≤ {args.tol:.0%}，"
                         f"没什么可配平的 —— 别在这里凭空造一根臂")
    if args.target <= 0 or args.target >= n_src:
        raise SystemExit(f"[配平] 目标点数 {args.target} 不合理（必须 0 < target < {n_src}）")

    # 逐高斯张量的第一维必须都等于 N —— 有一个不等就说明布局推错了，立即停
    for k in PER_GAUSSIAN:
        if int(d[k].shape[0]) != n_src:
            raise SystemExit(f"[配平] {k} 的第一维 {tuple(d[k].shape)} ≠ N={n_src} "
                             f"⇒ 它不是逐高斯张量，先验错了，拒绝继续")
    for k in SCALAR:
        if torch.is_tensor(d[k]) and d[k].numel() != 1:
            raise SystemExit(f"[配平] 标量项 {k} 有 {d[k].numel()} 个数，先验错了")

    rng = torch.Generator().manual_seed(args.seed)
    sel = torch.randperm(n_src, generator=rng)[: args.target].sort().values
    print(f"  抽样 seed={args.seed}  取 {len(sel)} 点（占源 {len(sel)/n_src:.2%}）")

    new_mp = list(mp)
    for k in PER_GAUSSIAN:
        i = LAYOUT.index(k)
        new_mp[i] = d[k][sel].contiguous()
    new_mp[LAYOUT.index("_opt_dict")] = None      # 理由见文件头：这个加载器不读它
    print("  _opt_dict -> None（load_visual_geometry 丢进 `_`，不读；"
          "保留原样会是一个形状对不上的雷）")

    out = Path(args.out)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)
    torch.save((new_mp, first_iter), out)

    # 自证一旦**抛异常**（不是返回 False），产物也必须删掉。
    # 本次就栽在这里：`.numpy()` 对 requires_grad 的张量抛 RuntimeError，
    # 而那时文件已经写完了 ⇒ 盘上留了一份 700 MB、从未被验证的 ckpt。
    # 所以护栏要在「调用」这一层，不能只在函数内部。
    try:
        ok = _selfcheck(out, args.target, first_iter, d)
    except Exception:
        out.unlink(missing_ok=True)
        print("[配平] 自证过程本身抛异常 ⇒ 已删除产物，不留未验证的半成品")
        raise
    if not ok:
        out.unlink(missing_ok=True)
        raise SystemExit("[配平] 自证失败 ⇒ 已删除产物，拒绝留下半成品")
    return 0


def _selfcheck(out, target, first_iter, d):
    """写回自证：重新 load 一遍，逐项打形状，不靠「我写对了」。返回 True/False。"""
    mp2, fi2 = torch.load(out, map_location="cpu")
    d2 = dict(zip(LAYOUT, mp2))
    print(f"  ✓ 写出 {out}  first_iter={fi2}")
    ok = True
    for k in LAYOUT:
        v = d2[k]
        s = "None" if v is None else (tuple(v.shape) if torch.is_tensor(v) else repr(v))
        print(f"      {k:18s} {s}")
        if k in PER_GAUSSIAN and int(d2[k].shape[0]) != target:
            print(f"      ✗ {k} 第一维 ≠ 目标 {target}")
            ok = False
    if fi2 != first_iter:
        print(f"      ✗ first_iter 变了：{fi2} ≠ {first_iter}")
        ok = False
    # 抽样必须是**子集**：所有目标行的 xyz 都要能在源里找到。
    # **不要用广播 isclose** —— (N_target, N_src) 的布尔张量在百万级会开几百 MB。
    # 用「源行排序 + searchsorted」做成员检查：O(N log N)、内存 O(N)，且覆盖**全部**行。
    # `.detach()` 不能省：capture() 存下来的 xyz 是 nn.Parameter（requires_grad=True），
    # 直接 .numpy() 会抛 RuntimeError。这里只是读数据做成员检查，不参与求导。
    a = np.ascontiguousarray(d2["xyz"].detach().numpy())
    b = np.ascontiguousarray(d["xyz"].detach().numpy())
    if a.shape[1] != 3 or b.shape[1] != 3:
        # 注意用 RuntimeError 而不是 SystemExit：SystemExit 继承 BaseException，
        # 调用处的 `except Exception` 接不住 ⇒ 产物不会被删。
        raise RuntimeError(f"[配平] xyz 不是 (N,3)：{a.shape} / {b.shape}")
    va = a.view([("x", "f4"), ("y", "f4"), ("z", "f4")]).ravel()
    vb = b.view([("x", "f4"), ("y", "f4"), ("z", "f4")]).ravel()
    src_sorted = np.sort(vb)
    pos = np.searchsorted(src_sorted, va)
    pos_c = np.minimum(pos, len(src_sorted) - 1)
    hit = src_sorted[pos_c] == va
    if not hit.all():
        print(f"      ✗ 有 {int((~hit).sum())} / {len(va)} 行不在源点集内（抽样写错了）")
        ok = False
    else:
        print(f"      抽样自证：全部 {len(va)} 行的 xyz 都在源点集内 ✓")
    return ok


if __name__ == "__main__":
    sys.exit(main())
