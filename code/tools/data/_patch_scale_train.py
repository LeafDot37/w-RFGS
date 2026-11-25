import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
# -*- coding: utf-8 -*-
from pathlib import Path

p = Path(__file__).parents[1] / "train_csi_cbf.py"
s = p.read_text(encoding="utf-8")

# 1) add raw-spectrum helper
old = '''def spec_from_cir(x, grid):
    x_sum = x.conj().sum(1)                             # [100]（对时延格求和）
    sp = (x_sum @ grid.a_mat.to(x.device)).abs()
    sp = sp.reshape(grid.hw)
    mn, mx = sp.min(), sp.max()
    return (sp - mn) / torch.clamp(mx - mn, min=1e-12)'''
new = '''def spec_raw(x, grid):
    """归一化前的 CBF 幅度图；pmax 与生成器的 scales_*.jsonl 同一量纲。"""
    x_sum = x.conj().sum(1)
    return (x_sum @ grid.a_mat.to(x.device)).abs().reshape(grid.hw)


def spec_from_cir(x, grid):
    sp = spec_raw(x, grid)
    mn, mx = sp.min(), sp.max()
    return (sp - mn) / torch.clamp(mx - mn, min=1e-12)'''
assert old in s
s = s.replace(old, new)

# 2) args
old2 = '''    ap.add_argument("--w_input", choices=["none", "dir", "complex"], default="none",'''
new2 = '''    ap.add_argument("--scale_root", default="",
                    help="逐实例 scales_instXXX.jsonl 目录；给了就监督绝对尺度")
    ap.add_argument("--w_scale", type=float, default=0.0,
                    help="log pmax 尺度损失权重（配合 --scale_root）")
    ap.add_argument("--w_input", choices=["none", "dir", "complex"], default="none",'''
assert old2 in s
s = s.replace(old2, new2)

# 3) load scales after meta
old3 = '''    rows = [r for r in rows_all if r.get("role", "train") in args.train_roles]'''
new3 = '''    scales = {}
    if args.scale_root and args.w_scale > 0:
        for f in Path(args.scale_root).glob("scales_inst*.jsonl"):
            iid = int(f.stem.split("inst")[1])
            d = {}
            for ln in open(f, encoding="utf-8"):
                r = json.loads(ln)
                d[int(r["view"])] = (float(r["pmax"]), float(r["pmean"]))
            scales[iid] = d
        print(f"[scale] {len(scales)} instances, {sum(len(v) for v in scales.values())} views")
    rows = [r for r in rows_all if r.get("role", "train") in args.train_roles]'''
assert old3 in s
s = s.replace(old3, new3)

# 4) global scale parameter after embed
old4 = '''    dtau = (nn.Parameter(torch.zeros(P, device=dev))
            if args.dtau_lr > 0 else None)'''
new4 = '''    dtau = (nn.Parameter(torch.zeros(P, device=dev))
            if args.dtau_lr > 0 else None)
    log_scale = (nn.Parameter(torch.zeros((), device=dev))
                 if args.scale_root and args.w_scale > 0 else None)'''
assert old4 in s
s = s.replace(old4, new4)

# 5) optimizer
old5 = '''    ] + ([{"params": [dtau], "lr": args.dtau_lr}] if dtau is not None else []))'''
new5 = '''    ] + ([{"params": [dtau], "lr": args.dtau_lr}] if dtau is not None else [])
      + ([{"params": [log_scale], "lr": 1e-2}] if log_scale is not None else []))'''
assert old5 in s
s = s.replace(old5, new5)

# 6) loss: raw spectrum + scale term
old6 = '''            if args.train_op == "cbf":
                u = spec_from_cir(x, grid)'''
new6 = '''            if args.train_op == "cbf":
                xs = x * torch.exp(log_scale) if log_scale is not None else x
                sp = spec_raw(xs, grid)
                mn, mx = sp.min(), sp.max()
                u = (sp - mn) / torch.clamp(mx - mn, min=1e-12)'''
assert old6 in s
s = s.replace(old6, new6)

old7 = '''            li = (u - gt).abs().mean()'''
new7 = '''            li = (u - gt).abs().mean()
            if log_scale is not None:
                _iid = int(inst["id"])
                _tr = scales.get(_iid, {}).get(int(vn))
                if _tr is not None:
                    _pred = sp.max().clamp_min(1e-30)
                    li = li + args.w_scale * torch.abs(
                        torch.log(_pred) - math.log(max(_tr[0], 1e-30)))'''
assert old7 in s
s = s.replace(old7, new7)

# 7) save log_scale in extras
old8 = '''                          head, embed, it, extra={"csi_T": args.T,'''
new8 = '''                          head, embed, it, extra={"csi_T": args.T,
                                                  "log_scale": (float(log_scale.detach())
                                                                if log_scale is not None else 0.0),'''
assert old8 in s
s = s.replace(old8, new8)

p.write_text(s, encoding="utf-8")
print("ok")
