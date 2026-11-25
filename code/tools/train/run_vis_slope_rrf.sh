#!/bin/bash
# =============================================================================
# run_vis_slope_rrf.sh —— 方案C 轴1：**视觉质量 → RF 质量** 的因果斜率
#                          （GPU 段 PREP/STEP0/RRF/RRFEVAL + 零 GPU 段 GATE）
# =============================================================================
# 命题（来自 docs/视觉引导体现不明显_诊断与操作方案.md §2）
# -----------------------------------------------------------------------------
# 现在的困境是「看不出引导」，而「看不出」有两种可能：引导**弱**，或者引导
# 非线性/被掩盖。**测斜率可以直接回答「引导有多强」**，而不管它能不能被用来刷指标。
#
# 做法：只操纵**一个**变量 —— 视觉短训步数 —— 然后跑**同一个** RRF。
# 视觉侧的其它一切（init 点云、数据集、种子、配方）逐条冻结。
#
# 预登记的判据（写在看到数据之前）
# -----------------------------------------------------------------------------
#   斜率显著（RF 指标随视觉步数单调，且首末档跨度 > 3× 抖动）
#     ⇒ 视觉引导**存在且可量化**，斜率本身就是「引导强度」的量化指标。
#   斜率平坦（首末档跨度落在抖动内）
#     ⇒ 视觉对 RF 重建**没有可测的因果贡献**。硬结论，必须如实写。
#   参考抖动：MVDR 0.0418 dB / CBF 0.2461 dB（同 RRF 阶段同输入重跑实测，
#   来源 results/run_20260914_协方差初始化/说明.md）。
#
# 档位（**为什么不是文档里的 0/500/1000/2000/4000/8000**）
# -----------------------------------------------------------------------------
# 文档那六档在服务器上凑不齐：`run_20260906_212940_早期短训扫描` 只存了 500–3000 的
# **点云**、没存 ckpt，4000 也没有；而 RRF 的物理起点必须是 ckpt
# （`train_rrf_scalar.py:load_visual_geometry` 从 `--start_checkpoint` 解包，**不读 ply**）。
# 服务器上的上传带宽实测只有 0.75 MB/s（909 MB 传了 20 分 20 秒）⇒ 也不许把本地的
# 五份 chkpnt 推上去（4.8 GB ≈ 1 小时 45 分）。
# 所以改成**服务器原地重建**：从服务器自己的 `mvsplat_clouds/init.ply` 起一次 8000 步
# 短训，顺路在 500/1000/2000/3000/4000/8000 存档。视觉短训是**方法无关**的
# （MVDR/CBF 共用同一份几何，差别只在频谱 GT）⇒ 一次训练喂两根方法。
#
#   0 步   —— 纯 MVSplat 前馈、**零视觉优化**（tools/make_step0_ckpt.py 现建）。
#             这是本轴最有价值的一端：它把「前馈先验」与「视觉训练」分开。
#   500 … 8000 —— 同一次短训的六个存档点。
#   30000  —— 直接复用 parity 的 chkpnt30000.pth（就是组1 A3 臂用的那份物理基座）。
#
# ⚠ 30000 那一档与 A3 的**物理基座逐字节相同**（同一文件），但 RRF 预算不同
#   （这里是本地 1 万，A3 是 3 万）⇒ 它的读数**不许**直接和 A3 比。
#
# RRF 阶段的唯一变量
# -----------------------------------------------------------------------------
# A× 标量头，超参逐条相同（--embed_dim 32 --mlp_hidden 128 --mlp_depth 2
# --pe_xyz_L 5 --pe_dir_L 3 --use_dist --n_bands 2 --band_mix mult），
# LR 协议相同（exp, 8e-3→8e-5, max_steps 40000），本地步数相同（1 万）。
# ⇒ 相邻两档之间**只差视觉短训步数**。
# ⚠ 但有一件**结构性的、消不掉的**混杂：视觉步数不同 ⇒ 致密化/剪枝历史不同
#   ⇒ **点数不同**。而 A× 头的逐点 embedding 是 P×32 ⇒ 点数差就是容量差。
#   所以 GATE 段把每档点数打进 `来源.txt`，出图时逐点标注；见 §「判据的边界」。
#
# 分段（SEGMENTS 控制；每段按产物存在幂等跳过，可中断续跑）
# -----------------------------------------------------------------------------
#   PREP     视觉短训 0→8000（GPU），在 500/1000/2000/3000/4000/8000 存 ckpt+ply
#   STEP0    建 first_iter=0 的 ckpt（GPU：distCUDA2 要 CUDA）
#   GATE     零 GPU 七闸（ckpt 齐 / first_iter 对 / spatial_lr_scale 全等 / 点数为正 …）
#   RRF      7 档 × [$RRF_METHODS] 的 A× 标量头，本地 1 万步
#   RRFEVAL  各节点 640 全量渲染 + 三指标
#
# 用法
# -----------------------------------------------------------------------------
#   全量：CUDA_VISIBLE_DEVICES=1 nohup bash tools/run_vis_slope_rrf.sh \
#           > /tmp/run_vis_slope_rrf.log 2>&1 &
#   冒烟：SEGMENTS="PREP STEP0 GATE" CUDA_VISIBLE_DEVICES=1 bash tools/run_vis_slope_rrf.sh
#   只跑 RRF：SEGMENTS="GATE RRF" CUDA_VISIBLE_DEVICES=1 RRF_METHODS=MVDR ...
# =============================================================================
set -u

# 频谱集有近千张图，`readColmapCameras` 会**逐张 open** 读尺寸 ⇒ fd 上限不够就
# `OSError: [Errno 24] Too many open files`（第一版就栽在这，14 个节点全灭）。
# 先试硬上限，再退回当前值 —— 拿不到就继续跑（不因为设不上就拒绝执行）。
ulimit -n 65535 2>/dev/null || ulimit -n "$(ulimit -Hn 2>/dev/null)" 2>/dev/null || true

SEGMENTS="${SEGMENTS:-PREP STEP0 GATE RRF RRFEVAL}"

ROOT=/data/yuexf/3dgs
cd "$ROOT" || exit 9
source ./load_config.sh
load_config full

TOP="${TOP:-$P_OUT/run_20260915_视觉质量因果斜率}"
LOG="$TOP/日志"
BASE_DIR="$TOP/物理起点"
RRF_DIR="$TOP/RRF"
mkdir -p "$LOG" "$BASE_DIR"

VIS_DS="${VIS_DS:-${P_VISUAL:-/data/yuexf/3dgs/datasets/RF-3DGS_dataset/blender_visual_dataset}}"
INIT_PLY="${INIT_PLY:-$P_CLOUDS/init.ply}"
PARITY_DIR="${PARITY_DIR:-$P_OUT/run_20260907_185429_parity档_视觉30k基座/rf-3dgs_1}"
PARITY_CKPT="$PARITY_DIR/chkpnt30000.pth"

VIS_ROOT="$TOP/视觉短训/rf-3dgs_vis"
VIS_STEPS_TRAIN="500 1000 2000 3000 4000 8000"   # 一次短训里的存档点
# 本实验真正跑的 RRF 档位。**可用环境变量覆盖**，用于补跑个别档而不重跑整条轴：
#     VIS_STEPS="0 500" SEGMENTS='RRF RRFEVAL' bash tools/run_vis_slope_rrf.sh
# 注意 STEP0 / GATE 段同样按本变量循环；只补跑时用 SEGMENTS 把这两段关掉
# （它们的产物 0/500 档早已存在，RRF 段的跳过判据是**文件级**的
#  `-f $MD/point_cloud/iteration_$ABS/point_cloud.ply`，空目录不会被误跳）。
VIS_STEPS="${VIS_STEPS:-0 500 1000 2000 4000 8000 30000}"
RRF_ITERS="${RRF_ITERS:-10000}"
RRF_METHODS="${RRF_METHODS:-MVDR CBF}"
DS="${DS:-${P_SPECTRUM:-}}"
PY="$PY"

echo "==== [$(date +%F_%T)] 视觉质量因果斜率  TOP=$TOP"
echo "     SEGMENTS='$SEGMENTS'"
echo "     CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-（未设，PREP/STEP0/RRF 段会报错）}'"
echo "     档位 VIS_STEPS='$VIS_STEPS'（RRF 本地 $RRF_ITERS 步）  RRF_METHODS='$RRF_METHODS'"
echo "     init 点云 INIT_PLY='$INIT_PLY'"
echo "     视觉集 VIS_DS='$VIS_DS'   频谱集 DS='$DS'"

FAIL=0
want() { case " $SEGMENTS " in *" $1 "*) return 0;; esac; return 1; }

# 档位 → (ckpt 路径, first_iter)。**唯一**的档位映射，所有段共用，避免各段各写一份。
# ⚠ `train.py` 把 ckpt 写在**模型根目录**（`os.path.join(model_path, "chkpnt%d.pth")`），
#   不是 `checkpoint/` 子目录 —— 本地那份历史产物里的 `checkpoint/` 是人手整理的，
#   照它写路径会全部落空（第一版就栽在这）。
arm_ckpt() {   # $1=步数；stdout = ckpt 绝对路径
  case "$1" in
    0)     echo "$BASE_DIR/vis0/chkpnt0.pth" ;;
    30000) echo "$PARITY_CKPT" ;;
    *)     echo "$VIS_ROOT/chkpnt$1.pth" ;;
  esac
}
arm_dir() { echo "$BASE_DIR/vis$1"; }

# ── 段 PREP（GPU）：视觉短训 0→8000，顺路在 6 个迭代存档 ─────────────────────
# 为什么一次跑完 8000 而不是分 6 次：ckpt 必须落在**同一条致密化轨迹**上，
#   「训到 500 存一份、再从 500 续到 1000」在 3DGS 里不是同一回事（opacity reset /
#   densify 的调度被截断）。一次跑完是最省事也最不会出错的做法。
# `--eval` **不开**：与 parity 的 30000 档同协议（cfg_args 里 eval=False ⇒ 全部相机进训练集）。
#   开了会让本轴的上端与 parity 的 chkpnt30000 不同协议，斜率的右端点就不是同一个东西。
if want PREP; then
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "PREP 段缺 CUDA_VISIBLE_DEVICES"; exit 2; }
  [ -f "$INIT_PLY" ] || { echo "init 点云缺失: $INIT_PLY"; exit 1; }
  [ -d "$VIS_DS" ]   || { echo "视觉数据集缺失: $VIS_DS"; exit 1; }
  echo; echo "[$(date +%F%T)] ==== 段 PREP：视觉短训 0→8000（卡 $CUDA_VISIBLE_DEVICES）===="
  cd "$P_CODE" || exit 9
  LAST="$(echo $VIS_STEPS_TRAIN | awk '{print $NF}')"
  if [ -f "$VIS_ROOT/point_cloud/iteration_${LAST}/point_cloud.ply" ]; then
    echo "  [T] 末档 iteration_$LAST 已在，跳过短训"
  else
    SAVE_ARG="$(echo $VIS_STEPS_TRAIN | tr ' ' ',')"
    export RF_POINTCLOUD="$INIT_PLY"
    echo "  RF_POINTCLOUD=$RF_POINTCLOUD"
    echo "  iterations=$LAST  save/checkpoint=[$SAVE_ARG]  resolution=2  eval=False"
    "$PY" train.py -s "$VIS_DS" -m "$VIS_ROOT" \
        --iterations "$LAST" \
        --save_iterations ${VIS_STEPS_TRAIN} \
        --checkpoint_iterations ${VIS_STEPS_TRAIN} \
        --resolution 2 --data_device cuda --quiet \
        > "$LOG/prep_vis_shorttrain.txt" 2>&1 \
      || { echo "  PREP_FAIL：短训失败"; tail -30 "$LOG/prep_vis_shorttrain.txt"; FAIL=1; }
    unset RF_POINTCLOUD
    grep -aE 'Number of points at initialisation|Training complete' \
        "$LOG/prep_vis_shorttrain.txt" | sed 's/^/    /' | head -3
  fi
  # 逐档点名：少一个档后面整条斜率就少一个点，必须当场看见
  for s in $VIS_STEPS_TRAIN; do
    [ -f "$(arm_ckpt $s)" ] \
      || { echo "  [PREP] 缺 chkpnt$s.pth"; FAIL=1; }
    [ -f "$VIS_ROOT/point_cloud/iteration_$s/point_cloud.ply" ] \
      || { echo "  [PREP] 缺 iteration_$s 的点云"; FAIL=1; }
  done
fi

# ── 段 STEP0（GPU）：纯前馈、零视觉优化的第 0 档 ─────────────────────────────
if want STEP0; then
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "STEP0 段缺 CUDA_VISIBLE_DEVICES"; exit 2; }
  echo; echo "[$(date +%F%T)] ==== 段 STEP0：first_iter=0 的物理起点 ===="
  cd "$P_CODE" || exit 9
  OUT0="$(arm_ckpt 0)"
  if [ -f "$OUT0" ]; then
    echo "  [T] $OUT0 已在，跳过"
  else
    REF=""; [ -f "$PARITY_CKPT" ] && REF="$PARITY_CKPT"
    [ -n "$REF" ] || echo "  [warn] 参考 ckpt 不在，spatial_lr_scale 将无佐证"
    "$PY" tools/make_step0_ckpt.py \
        --init_ply "$INIT_PLY" ${REF:+--ref_ckpt "$REF"} \
        --sh_degree 3 --out "$OUT0" \
        > "$LOG/step0_make.txt" 2>&1 \
      || { echo "  STEP0_FAIL"; tail -30 "$LOG/step0_make.txt"; FAIL=1; }
    tail -6 "$LOG/step0_make.txt" | sed 's/^/    /'
  fi
fi

# ── 段 GATE（零 GPU）：七闸。不过不许进 GPU ──────────────────────────────────
if want GATE; then
  echo; echo "[$(date +%F%T)] ==== 段 GATE：零 GPU 七闸 ===="
  cd "$P_CODE" || exit 9
  "$PY" - "$VIS_STEPS" "$TOP" "$RRF_DIR" "$PARITY_CKPT" "$VIS_DS" "$DS" "$RRF_METHODS" <<'PYEOF'
import json, os, sys, hashlib
steps = sys.argv[1].split(); top, rrf_dir, parity, vis_ds, ds = sys.argv[2:7]
methods = sys.argv[7].split()
import torch

def ckpt_of(s):
    if s == "0":     return os.path.join(top, "物理起点/vis0/chkpnt0.pth")
    if s == "30000": return parity
    # 与 arm_ckpt() 必须一致：train.py 写在模型根目录，不是 checkpoint/ 子目录
    return os.path.join(top, "视觉短训/rf-3dgs_vis/chkpnt%s.pth" % s)

bad = 0
info = {}
for s in steps:
    p = ckpt_of(s)
    if not os.path.isfile(p):
        print("  [G1] FAIL  档 %s：ckpt 缺失 %s" % (s, p)); bad += 1; continue
    m, fi = torch.load(p, map_location="cpu", weights_only=False)
    d = dict(step=int(s), path=p, first_iter=int(fi), P=int(m[1].shape[0]),
             s_lr=float(m[11]), P_mB=os.path.getsize(p) / 1e6,
             md5=hashlib.md5(open(p, "rb").read(1 << 20)).hexdigest()[:8])
    info[s] = d
    # G1 结构：12 项、first_iter 必须等于档位、行数一致
    if len(m) != 12:      print("  [G1] FAIL  档 %s：model_params %d 项 != 12" % (s, len(m))); bad += 1
    if int(fi) != int(s): print("  [G1] FAIL  档 %s：first_iter=%s != %s" % (s, fi, s)); bad += 1
    if d["P"] <= 0:       print("  [G1] FAIL  档 %s：点数 %d" % (s, d["P"])); bad += 1
    xs = [m[i] for i in (2, 4, 5)]
    if any(not torch.isfinite(t).all() for t in xs):
        print("  [G1] FAIL  档 %s：SH/scaling/rotation 里有非有限值" % s); bad += 1
    if not torch.isfinite(m[1]).all():
        print("  [G1] FAIL  档 %s：xyz 里有非有限值" % s); bad += 1
    print("  [G1] 档 %-6s P=%-9d first_iter=%-6d s_lr=%.6f  %.0f MB" %
          (s, d["P"], d["first_iter"], d["s_lr"], d["P_mB"]))

# G2 坐标系：spatial_lr_scale 全等。它经头部的 _dist() 真的进输入，
#    各档不同 ⇒ 各档的输入语义就不同 ⇒ 斜率无意义。
sl = {v["s_lr"] for v in info.values()}
if len(sl) != 1:
    print("  [G2] FAIL  spatial_lr_scale 不全等：%s" % sorted(sl)); bad += 1
else:
    print("  [G2] spatial_lr_scale 全等 = %.6f ✓" % sl.pop())

# G3 点数差异**必须打印**：它是这个设计消不掉的混杂（P×32 容量随 P 变）
Ps = [info[s]["P"] for s in steps if s in info]
if Ps:
    print("  [G3] 点数跨度 %d ~ %d（差 %.1f%%）—— P×32 逐点容量跟着变，"
          "这是本轴消不掉的混杂，出图必须逐点标注" % (min(Ps), max(Ps), 100.0 * (max(Ps) - min(Ps)) / min(Ps)))

# G4 数据源
for M in methods:
    p = os.path.join(ds, "3dgs_%s_100" % M)
    print("  [G4] %s 频谱集 %s" % ("OK  " if os.path.isdir(p) else "FAIL", p))
    if not os.path.isdir(p): bad += 1
print("  [G5] 视觉集 %s %s" % ("OK  " if os.path.isdir(vis_ds) else "FAIL", vis_ds))
if not os.path.isdir(vis_ds): bad += 1

# G6 RRF 绝对末档 = first_iter + 本地步数；逐档打出来，免得后面写错
RRF_ITERS = 10000
print("  [G6] 各档 RRF 绝对末档 = first_iter + %d：" % RRF_ITERS)
for s in steps:
    if s in info:
        print("        档 %-6s → abs %d" % (s, info[s]["first_iter"] + RRF_ITERS))

# G7 产物侧写：来源.txt（一次写全，后面出图/写文档只许读它，不许各自重算）
src = os.path.join(top, "来源.txt")
with open(src, "w", encoding="utf-8") as f:
    f.write("视觉质量因果斜率（方案C 轴1）\n")
    f.write("生成 %s\n\n" % __import__("datetime").datetime.now().strftime("%F %T"))
    f.write("RRF 本地步数 = %d；RRF_METHODS = %s\n" % (RRF_ITERS, " ".join(methods)))
    f.write("init 点云 = /data/yuexf/3dgs/results/mvsplat_clouds/init.ply\n")
    f.write("视觉集 = %s（eval=False，与 parity 30000 档同协议）\n" % vis_ds)
    f.write("频谱集 = %s\n\n" % ds)
    f.write("%-8s %-12s %-10s %-12s %-10s %s\n" %
            ("档位", "first_iter", "点数P", "s_lr", "ckpt_MB", "md5(前1MB)"))
    for s in steps:
        d = info.get(s)
        if d:
            f.write("%-8s %-12d %-10d %-12.6f %-10.0f %s\n" %
                    (s, d["first_iter"], d["P"], d["s_lr"], d["P_mB"], d["md5"]))
        else:
            f.write("%-8s 缺\n" % s)
    f.write("\n判据（预登记）：首末档 ΔPSNR 跨度 > 3× 抖动（MVDR 0.0418 / CBF 0.2461 dB）"
            "⇒ 斜率显著；否则平坦，视觉对 RF 无可测因果贡献。\n")
print("  [G7] 写 %s" % src)
print("  GATE %s（bad=%d）" % ("FAIL" if bad else "PASS", bad))
sys.exit(1 if bad else 0)
PYEOF
  [ $? -ne 0 ] && { echo "  GATE FAIL ⇒ 不进 GPU 段"; exit 1; }
fi

# ── 段 RRF（GPU）：7 档 × [$RRF_METHODS] ─────────────────────────────────────
if want RRF; then
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "RRF 段缺 CUDA_VISIBLE_DEVICES"; exit 2; }
  echo; echo "[$(date +%F%T)] ==== 段 RRF：档位 [$VIS_STEPS] × [$RRF_METHODS]（卡 $CUDA_VISIBLE_DEVICES）===="
  cd "$P_CODE" || exit 9
  for M in $RRF_METHODS; do
    case "$M" in
      MVDR|CBF) ;;
      *) echo "  [拒绝] RRF_METHODS 里有 '$M'：本脚本只跑标量头（MVDR / CBF）。"
         echo "         MPC / Delay 是 mlp 引擎，走 train_rrf_mlp.py 的另一份驱动 ——"
         echo "         在这里静默跑成标量头会变成「同一方法两种色头」的隐性变量。"
         exit 3;;
    esac
  done
  for s in $VIS_STEPS; do
    BASE="$(arm_ckpt $s)"
    [ -f "$BASE" ] || { echo "  [档 $s] 物理起点缺失: $BASE"; FAIL=1; continue; }
    ABS=$((s + RRF_ITERS))
    for M in $RRF_METHODS; do
      DATA="$DS/3dgs_${M}_100"
      [ -d "$DATA" ] || { echo "  [$M] 频谱数据缺失: $DATA"; FAIL=1; continue; }
      MD="$RRF_DIR/vis${s}/$M"; mkdir -p "$MD"
      if [ -f "$MD/point_cloud/iteration_${ABS}/point_cloud.ply" ]; then
        echo "  [vis$s/$M] 末档已在，跳过训练"
      else
        echo "[$(date +%F%T)] RRF vis$s / $M（起点 $(basename "$BASE")，绝对 $ABS = 本地 $RRF_ITERS）"
        # 频谱集的 init 点云由本脚本自己管（RF_POINTCLOUD 只对阶段一有意义）
        unset RF_POINTCLOUD
        "$PY" train_rrf_scalar.py -s "$DATA" -m "$MD" \
            --start_checkpoint "$BASE" --iterations "$ABS" \
            --save_iterations "$ABS" --densify_until_iter 0 --data_device cpu --quiet \
            --embed_dim 32 --mlp_hidden 128 --mlp_depth 2 \
            --lr_schedule exp --mlp_lr_init 8e-3 --mlp_lr_final 8e-5 --lr_max_steps 40000 \
            --pe_xyz_L 5 --pe_dir_L 3 --use_dist --n_bands 2 --band_mix mult \
            > "$LOG/rrf_vis${s}_${M}.txt" 2>&1 \
          || { echo "  RRF_FAIL vis$s/$M"; tail -30 "$LOG/rrf_vis${s}_${M}.txt"; FAIL=1; continue; }
        grep -aE 'Number of points|头参数量|Training complete' "$LOG/rrf_vis${s}_${M}.txt" \
          | sed 's/^/    /' | head -4
      fi
      [ -f "$MD/point_cloud/iteration_${ABS}/point_cloud.ply" ] \
        || { echo "  [vis$s/$M] 缺末档 ply"; FAIL=1; }
    done
  done
fi

# ── 段 RRFEVAL（GPU）：640 全量渲染 + 三指标 ─────────────────────────────────
# LPIPS 是主判据，PSNR / SSIM 并报。**必须 640 张全量**（--subset 0）。
if want RRFEVAL; then
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "RRFEVAL 段缺 CUDA_VISIBLE_DEVICES"; exit 2; }
  echo; echo "[$(date +%F%T)] ==== 段 RRFEVAL：[$VIS_STEPS] × [$RRF_METHODS] 的 640 全量渲染 + 三指标 ===="
  cd "$P_CODE" || exit 9
  pngn() { find "$1" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l; }
  for s in $VIS_STEPS; do
    ABS=$((s + RRF_ITERS))
    for M in $RRF_METHODS; do
      DATA="$DS/3dgs_${M}_100"; MD="$RRF_DIR/vis${s}/$M"
      [ -d "$MD/point_cloud/iteration_${ABS}" ] || { echo "  [skip] vis$s/$M 无 abs$ABS"; continue; }
      unset RF_POINTCLOUD
      if [ "$(pngn "$MD/test/ours_${ABS}/renders")" -lt 640 ]; then
        echo "[$(date +%F%T)] 渲染 vis$s/$M 640 张"
        "$PY" render_rrf_scalar.py -s "$DATA" -m "$MD" --iteration "$ABS" \
              --eval --skip_train --subset 0 --quiet \
            > "$LOG/rrfrender_vis${s}_${M}.txt" 2>&1 \
          || { echo "  RENDER_FAIL vis$s/$M"; tail -20 "$LOG/rrfrender_vis${s}_${M}.txt"; FAIL=1; continue; }
      fi
      [ "$(pngn "$MD/test/ours_${ABS}/renders")" -ge 640 ] \
        || { echo "  [vis$s/$M] 渲染不足 640 张（$(pngn "$MD/test/ours_${ABS}/renders")）"; FAIL=1; }
      "$PY" "$P_TOOLS/metrics_fast.py" -m "$MD" --iterations "$ABS" --subset 0 \
            --out "$MD/results.json" > "$LOG/rrfmetrics_vis${s}_${M}.txt" 2>&1 \
        || { echo "  METRICS_FAIL vis$s/$M"; tail -20 "$LOG/rrfmetrics_vis${s}_${M}.txt"; FAIL=1; continue; }
      "$PY" - "$MD" "$ABS" "$s" "$M" <<'PYEOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1] + "/results.json", encoding="utf-8"))
for k, v in d.items():
    if k.startswith("ours_" + sys.argv[2]) and "PSNR" in v:
        print("    档 %-6s %-6s PSNR %.4f  SSIM %.5f  LPIPS %.5f"
              % (sys.argv[3], sys.argv[4], v["PSNR"], v["SSIM"], v["LPIPS"]))
PYEOF
    done
  done
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "==== [$(date +%F_%T)] 全部段通过 ✓  产物在 $TOP"
else
  echo "==== [$(date +%F_%T)] 有失败项（FAIL=1），看上面逐行 [xxx_FAIL] 与 $LOG"
fi
exit "$FAIL"
