# -*- coding: utf-8 -*-
"""阶段 1：w 条件精修器（2D 代理闸）。

背景（见 `docs/扩散模型迁移_调研与实施方案.md`）
--------------------------------------------------------------------------------
阶段 0 的判决是：**只吃 [u_hat, u_prior] 的混合器对 w 结构性死亡**。所以阶段 1
必须让 w 作为条件向量**逐层调制**网络，由网络「生成」w 响应，而不是在两张图之间
做选择。本文件就是这条路线的最小可跑实现。

它是不是最终形态？不是。它是在**不改引擎核心**的前提下，用最低成本回答一个
问题：给定冻结模型已经渲染出 u_hat（其中含 13% 的 w 响应），一个条件生成器
能不能把 w 响应补到真值的显著比例，同时不把基础保真度弄坏。能，才值得把这套
条件注入搬进 3D 高斯表示里；不能，就说明瓶颈不在解码器。

模型
--------------------------------------------------------------------------------
FiLM-U-Net：输入 u_hat（1 通道），条件 z = MLP([sinθ, cosθ, sinφ, cosφ, uniform])，
每个残差块用 (γ, β) 调制。输出是加在 u_hat 上的残差：

    u_out = clamp(u_hat + f(u_hat, w), 0, 1)

残差是为了让「什么都不做」是网络的零解，训练稳；但**不构成混合器**——残差由
网络生成，不从任何第二张图里取。

两个评估口径
--------------------------------------------------------------------------------
1. 三指标（jet 交付域）：PSNR / SSIM / LPIPS。
2. **w 响应探针**（主闸）：同一个 u_hat、只换 w 条件，输出的 RMS 差异 vs
   真值的 w 效应。这是唯一能证明「w 通道真的被用上了」的读数。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from rfgs.core.angular_data import SpectrumStore                                   # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut                   # noqa: E402
from splat.utils.image_utils import psnr                                 # noqa: E402
from splat.utils.loss_utils import ssim                                  # noqa: E402


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class FiLMBlock(nn.Module):
    """两个 3×3 卷积 + 每个卷积后的 FiLM 调制 + 残差。"""

    def __init__(self, cin: int, cout: int, zc: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.f1 = nn.Linear(zc, 2 * cout)
        self.f2 = nn.Linear(zc, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        for lin in (self.f1, self.f2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _film(self, h, lin, z):
        g, b = lin(z).chunk(2, dim=1)
        return h * (1.0 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, x, z):
        h = self.act(self._film(self.c1(x), self.f1, z))
        h = self._film(self.c2(h), self.f2, z)
        return self.act(h + self.skip(x))


class FiLMUNet(nn.Module):
    def __init__(self, base: int = 48, zc: int = 128, depths: int = 3):
        super().__init__()
        self.zc = zc
        self.cond = nn.Sequential(nn.Linear(5, zc), nn.SiLU(), nn.Linear(zc, zc))
        cs = [base * (2 ** i) for i in range(depths + 1)]      # 48,96,192,384
        self.stem = nn.Conv2d(1, cs[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c_prev = cs[0]
        for i in range(depths):
            self.enc.append(FiLMBlock(c_prev, cs[i], zc))
            self.down.append(nn.Conv2d(cs[i], cs[i + 1], 3, stride=2, padding=1))
            c_prev = cs[i + 1]
        self.mid = FiLMBlock(c_prev, cs[-1], zc)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c_prev = cs[-1]
        for i in range(depths - 1, -1, -1):
            self.up.append(nn.Conv2d(c_prev, cs[i], 3, padding=1))
            self.dec.append(FiLMBlock(2 * cs[i], cs[i], zc))
            c_prev = cs[i]
        self.head = nn.Conv2d(cs[0], 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, wc):
        z = self.cond(wc)
        h = self.stem(x)
        skips = []
        for blk, dn in zip(self.enc, self.down):
            h = blk(h, z)
            skips.append(h)
            h = dn(h)
        h = self.mid(h, z)
        for up, blk, sk in zip(self.up, self.dec, reversed(skips)):
            h = F.interpolate(h, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([up(h), sk], dim=1), z)
        return self.head(h)


def w_cond(theta_deg, phi_deg, uniform) -> torch.Tensor:
    """w 的条件向量：角度用三角函数编码（避免 0/360 断点），uniform 用 0/1。"""
    t = torch.as_tensor(theta_deg, dtype=torch.float32) * math.pi / 180.0
    p = torch.as_tensor(phi_deg, dtype=torch.float32) * math.pi / 180.0
    u = torch.as_tensor(uniform, dtype=torch.float32)
    t, p = torch.broadcast_tensors(t, p)
    u = torch.broadcast_to(u, t.shape)
    return torch.stack([torch.sin(t), torch.cos(t), torch.sin(p), torch.cos(p), u], -1)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
class PairSet:
    """u_hat 缓存 + GT 谱图。返回 (u_hat(1,H,W), wc(5), u_gt(1,H,W))。"""

    def __init__(self, cache_dir: str, spectra_root: str, device="cpu"):
        self.dir = cache_dir
        m = json.load(open(os.path.join(cache_dir, "manifest.json"), encoding="utf-8"))
        self.manifest = m
        self.views = m["views"]
        self.inst = m["instances"]
        self.spectra = SpectrumStore(spectra_root, device=device, key_fmt="inst{idx:03d}")
        self.H, self.W = self._probe_size()

    def _probe_size(self):
        for it in self.inst:
            d = os.path.join(self.dir, "u", f"inst{it['id']:03d}", f"{self.views[0]}.png")
            if os.path.exists(d):
                a = np.asarray(Image.open(d))
                return int(a.shape[0]), int(a.shape[1])
        raise SystemExit("缓存里一张图都没有")

    def rows(self, roles):
        want = {roles} if isinstance(roles, str) else set(roles)
        return [i for i, it in enumerate(self.inst) if it["role"] in want]

    def item(self, i: int, v: int):
        it = self.inst[i]
        nm = self.views[v]
        p = os.path.join(self.dir, "u", f"inst{it['id']:03d}", f"{nm}.png")
        a = np.asarray(Image.open(p)).astype(np.float32) / 255.0
        uh = torch.from_numpy(a)[None]
        gt = self.spectra.unit(int(it["id"]), nm)[0].cpu()
        # w_cond 对标量输入返回 (5,)，**不要再 [0]**——那会退化成单个标量。
        wc = w_cond(it["w_theta"], it["w_phi"], it["w_uniform"])
        return uh, wc, gt[None]


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train(args):
    dev = args.device
    ds = PairSet(args.cache, args.spectra_root, device="cpu")
    tr_rows = ds.rows(args.train_roles.split(","))
    va_rows = ds.rows(args.val_roles.split(","))
    print(f"[s1] 训练实例 {len(tr_rows)}，验证 {len(va_rows)}，"
          f"视角 {len(ds.views)}，图 {ds.H}×{ds.W}", flush=True)

    model = FiLMUNet(base=args.base).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[s1] 参数 {npar/1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters, eta_min=args.lr * 0.05)
    rng = np.random.default_rng(args.seed)

    def batch(rows, bs):
        xs, cs, ys = [], [], []
        for _ in range(bs):
            i = rows[int(rng.integers(len(rows)))]
            v = int(rng.integers(len(ds.views)))
            x, c, y = ds.item(i, v)
            if args.shuffle_w:
                # 阴性对照臂：条件向量与本图无关。若这一臂读数与正臂相同，
                # 说明网络根本没在用 w，正臂的任何「提升」都是别的因素。
                j = rows[int(rng.integers(len(rows)))]
                it = ds.inst[j]
                c = w_cond(it["w_theta"], it["w_phi"], it["w_uniform"])
            xs.append(x); cs.append(c); ys.append(y)
        return (torch.stack(xs).to(dev), torch.stack(cs).to(dev),
                torch.stack(ys).to(dev))

    t0 = time.time()
    best = -1e9
    os.makedirs(args.out, exist_ok=True)
    hist = []
    for it in range(1, args.iters + 1):
        model.train()
        x, c, y = batch(tr_rows, args.bs)
        pred = (x + model(x, c)).clamp(0, 1)
        loss = F.l1_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % args.log_every == 0:
            el = time.time() - t0
            print(f"[s1] it {it}/{args.iters} loss {float(loss):.5f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {it/el:.2f} it/s "
                  f"余 {(args.iters-it)/(it/el)/60:.0f} min", flush=True)
        if it % args.val_every == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                vl, vu = [], []
                for _ in range(args.val_batches):
                    x, c, y = batch(va_rows, args.bs)
                    pred = (x + model(x, c)).clamp(0, 1)
                    vl.append(float(F.l1_loss(pred, y)))
                    mse = ((pred - y) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
                    vu.append(float((-10 * torch.log10(mse)).mean()))
                v_l1, v_psnr = float(np.mean(vl)), float(np.mean(vu))
            print(f"[s1]   val it {it}: L1 {v_l1:.5f}  uPSNR {v_psnr:.3f}", flush=True)
            hist.append(dict(it=it, val_l1=v_l1, val_upsnr=v_psnr,
                             train_l1=float(loss)))
            if v_psnr > best:
                best = v_psnr
                torch.save(dict(model=model.state_dict(), it=it, args=vars(args)),
                           os.path.join(args.out, "best.pth"))
        if args.ckpt_every and it % args.ckpt_every == 0:
            torch.save(dict(model=model.state_dict(), it=it, args=vars(args)),
                       os.path.join(args.out, f"it{it:06d}.pth"))
    torch.save(dict(model=model.state_dict(), it=args.iters, args=vars(args)),
               os.path.join(args.out, "last.pth"))
    json.dump(hist, open(os.path.join(args.out, "hist.json"), "w"), indent=1)
    print(f"[s1] 完成，用时 {(time.time()-t0)/60:.1f} min", flush=True)


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def evaluate(args):
    dev = args.device
    ds = PairSet(args.cache, args.spectra_root, device=dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model = FiLMUNet(base=ck["args"].get("base", 48)).to(dev).eval()
    model.load_state_dict(ck["model"])
    lut = make_jet_lut(256).to(dev)
    crit = None
    if not args.no_lpips:
        from splat.lpipsPyTorch.modules.lpips import LPIPS
        crit = LPIPS("vgg", "0.1").to(dev).eval()
        for q in crit.parameters():
            q.requires_grad_(False)

    roles = args.eval_roles.split(",")
    rows = ds.rows(roles)
    by_id = {it["id"]: it for it in ds.inst}
    acc = {k: [] for k in ("psnr", "ssim", "lpips", "u_psnr")}
    base_acc = {k: [] for k in ("psnr", "ssim", "lpips", "u_psnr")}
    rms_pred_w, rms_gt_w, corr_w = [], [], []
    rms_err, rms_hat_err = [], []

    with torch.no_grad():
        for i in rows:
            it = ds.inst[i]
            bid = it["base_instance"]
            base = by_id.get(bid) if bid >= 0 else None
            for v, nm in enumerate(ds.views):
                x, c, y = ds.item(i, v)
                x, c, y = x[None].to(dev), c[None].to(dev), y[None].to(dev)
                out = (x + model(x, c)).clamp(0, 1)
                for tag, u in (("pred", out), ("hat", x)):
                    m = _metrics(u, y, lut, crit, dev)
                    tgt = acc if tag == "pred" else base_acc
                    for k in m:
                        tgt[k].append(m[k])
                rms_err.append(_rms(out, y))
                rms_hat_err.append(_rms(x, y))
                if base is not None:
                    ugt_base = ds.spectra.unit(int(bid), nm)[0][None].to(dev)
                    cb = w_cond(base["w_theta"], base["w_phi"],
                                base["w_uniform"])[None].to(dev)
                    out_b = (x + model(x, cb)).clamp(0, 1)
                    rms_pred_w.append(_rms(out, out_b))
                    rms_gt_w.append(_rms(y, ugt_base))
                    d1, d2 = (out - out_b).flatten(), (y - ugt_base).flatten()
                    if float(d1.std()) > 1e-6 and float(d2.std()) > 1e-6:
                        corr_w.append(float(torch.corrcoef(
                            torch.stack([d1, d2]))[0, 1]))

    mean = lambda d: {k: float(np.mean(v)) if v else float("nan")
                      for k, v in d.items()}
    out = dict(ckpt=args.ckpt, it=int(ck.get("it", -1)), roles=roles,
               n_views=int(len(ds.views)), n_instances=len(rows),
               pred=mean(acc), u_hat=mean(base_acc),
               rms_err_gray=float(np.mean(rms_err)),
               rms_hat_err_gray=float(np.mean(rms_hat_err)),
               rms_w_pred=float(np.mean(rms_pred_w)) if rms_pred_w else None,
               rms_w_gt=float(np.mean(rms_gt_w)) if rms_gt_w else None,
               rms_w_ratio=(float(np.mean(rms_pred_w) / max(np.mean(rms_gt_w), 1e-9))
                            if rms_pred_w else None),
               w_corr=float(np.mean(corr_w)) if corr_w else None)
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, f"eval_{'-'.join(roles)}.json"),
                        "w", encoding="utf-8"), indent=2)
    print(f"\n=== 评估（{','.join(roles)}，{len(rows)} 实例 × {len(ds.views)} 视角）===")
    print(f"  精修后  PSNR {out['pred']['psnr']:.4f}  SSIM {out['pred']['ssim']:.4f}  "
          f"LPIPS {out['pred']['lpips']:.4f}  uPSNR {out['pred']['u_psnr']:.4f}")
    print(f"  u_hat   PSNR {out['u_hat']['psnr']:.4f}  SSIM {out['u_hat']['ssim']:.4f}  "
          f"LPIPS {out['u_hat']['lpips']:.4f}  uPSNR {out['u_hat']['u_psnr']:.4f}")
    print(f"  误差 RMS  精修 {out['rms_err_gray']:.2f}  原样 {out['rms_hat_err_gray']:.2f}")
    if out["rms_w_pred"] is not None:
        print(f"  w 响应   精修 {out['rms_w_pred']:.3f}  真值 {out['rms_w_gt']:.3f}  "
              f"比例 {out['rms_w_ratio']:.4f}  corr {out['w_corr']:+.4f}")
    print(f"[s1] -> {args.out}")


def _rms(a, b):
    if b is None:
        return 0.0
    return float(((a - b) ** 2).mean().sqrt()) * 255.0


def _metrics(pred_u, gt_u, lut, crit, dev):
    a = pred_u.clamp(0, 1)
    b = gt_u.clamp(0, 1)
    ja = apply_jet(a[0], lut).unsqueeze(0)
    jb = apply_jet(b[0], lut).unsqueeze(0)
    p = float(psnr(ja, jb).mean())
    mse = float(((a - b) ** 2).mean())
    return dict(psnr=p if math.isfinite(p) else 99.0,
                ssim=float(ssim(ja, jb).mean()),
                lpips=float(crit(ja, jb).mean()) if crit is not None else 0.0,
                u_psnr=99.0 if mse <= 1e-12 else -10.0 * math.log10(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--cache", required=True, help="cache_u_hat.py 的输出目录")
    ap.add_argument("--spectra_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--train_roles", default="train")
    ap.add_argument("--val_roles", default="pos_holdout")
    ap.add_argument("--eval_roles", default="w_holdout")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_batches", type=int, default=8)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--shuffle_w", action="store_true",
                    help="阴性对照：训练时把 w 条件随机换成别的实例的")
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        if not args.ckpt:
            raise SystemExit("eval 需要 --ckpt")
        evaluate(args)


if __name__ == "__main__":
    main()
