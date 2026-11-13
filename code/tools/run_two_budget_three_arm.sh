#!/bin/bash
# =============================================================================
# run_two_budget_three_arm.sh —— 两组预算 × 三臂端到端驱动
# =============================================================================
# 命题：把「MVSplat 点云初始化 → 短程物理重建 → RRF 频谱重建」两段接起来，做
#       两组预算下的三臂对比（每臂 = 物理来源 × RRF 颜色头 的组合）：
#
#   臂              物理重建                      RRF 颜色头
#   A1 原版物理+原版RRF  上游 orig_seed_chkpnt30000   train.py 原版逐点 SH（引擎 SH）
#   A2 原版物理+新RRF    同上                        MPC/Delay=train_rrf_mlp（V14）
#                                                   MVDR/CBF=train_rrf_scalar（形态 A×）
#   A3 新物理+新RRF      MVSplat init + 短训         同 A2（四方法同一份 A* 协议）
#
#   组1 物理30000 / RRF30000（等预算对照，唯一变量 = init 来源）
#   组2 物理10000 / RRF20000（省预算命题：1/3 物理预算 + 2/3 RRF 预算）
#
# 口径（每条都从代码/仓库既有命名里定，不靠猜）
# -----------------------------------------------------------------------------
# * 「原版RRF」= code/train.py 的原版逐点 SH 色头。依据是仓库自己的双引擎命名
#   （run_rrf40k_driver.sh:7-8「引擎 SH = train.py（原版逐点 SH 色头）」）。
# * 「新RRF」= 统一 A* 协议（docs/RRF-MLP改进方案_对照Scaffold-GS.md:415）：
#     --lr_schedule exp --mlp_lr_init 8e-3 --mlp_lr_final 8e-5 --lr_max_steps 40000
#     --pe_xyz_L 5 --pe_dir_L 3 --use_dist --embed_dim 32 --mlp_hidden 128 --mlp_depth 2
#   四方法同一份协议 ⇒「新RRF」在三臂里是单一变量，不把协议差混进头部差。
# * RRF 步数按**本地步数**：--iterations 是绝对迭代数，实际新训 = iterations − first_iter
#   （train.py:38-40 解出 first_iter，:62 从 first_iter+1 起循环）。
#   ⇒ 组1 A1/A2/A3 都跑 --iterations 60000（本地 30000）；组2 A1/A2 跑 50000（本地 20000）、
#     组2 A3 跑 30000（10000+20000）。LR 调度位置 t = 本地步数/40000，三臂一致。
# * 组2 的 10000 步物理重建要避开「正好停在透明度重置点上」：
#     --densify_until_iter 5000 --opacity_reset_interval 1000
#   ⇒ 重置发生在 1000/2000/3000/4000，后 5000 步不重置（可验证，见 GATE 段）。
#   代价（如实登记）：该守卫同时管增密（train.py:124 包住 :129/:133），故 10000 步的
#   物理重建在 5000 步后不再增密。要解耦必须改 train.py，与「不改引擎内核」冲突。
#
# 只读复用（不重跑、不写基线）
# -----------------------------------------------------------------------------
#   组1 A1 = run_rrf40k_sh/<M> 的 abs60000 档；组2 A1 = 同 run 的 abs50000 档
#            （RRF 阶段几何冻结 + densify_until_iter 0 ⇒ 长跑中间档 = 独立跑到该档的状态）
#   组1/组2 A2 的 MVDR/CBF = run_20260912_多频段标量分支/<M> 的 60000/50000 档
#   组1 A3 的物理基座 = parity run 的 chkpnt30000.pth（MVSplat init 续训到 30000 步）
#   **上述目录一律只读**：需要渲染时在本 run 树下建「软链视图」
#   （point_cloud 软链 + cfg_args/cameras.json 拷贝），渲染与评测只往本 run 树里写。
#   已存在的 640 张全量渲染直接复用（metrics_fast.py 对 -m 只读，--out 指向本树）。
#
#   ⚠ **不能复用 `run_rrf40k_mlp`**：它的 cfg_args 只有 embed_dim/mlp_hidden/mlp_depth，
#   没有 lr_schedule=exp / pe_xyz_L / pe_dir_L / use_dist ⇒ 是 **V-base** 而非 A* 协议。
#   按口径 §2「四方法同一份协议」，A2/A3 的 MPC/Delay 必须**新训**，否则「新RRF」这条
#   腿就混进了协议差，三臂不再是单变量对照。（该 run 只借来做闸②的 MLP 色头相机等价。）
#
#   存档档位默认**只存末档**（--save_iterations <绝对末迭代>）：本交付物只要末档的
#   柱状图/图像对比/视频，收敛曲线不在范围内，故不留 9 档 ×12 run ≈ 26 GB 的中间云。
#   要曲线时用 SAVE_ITERS="45000 60000" 之类覆盖即可。
#
# 分段（SEGMENTS 控制；每段按产物存在幂等跳过，可中断续跑、可多卡分片）
# -----------------------------------------------------------------------------
#   A      阶段A：MVSplat init 点云（含 md5 与 parity 对拍）
#   B      阶段B：组2 的 10000 步物理重建
#   GATE   两道便宜闸：①重置时刻的**实证**验证 ②三色头相机等价性自检
#   RRF    12 次 RRF 训练（组×臂×方法；A1 全复用，实际新跑 6+6 里的 5+6）
#   RENDER 按方法匹配的渲染脚本渲 640 全量 + metrics_fast 出每臂 results.json
#   WALK   环绕/推轨漫游帧（三种颜色头）
#   REAL   真实机位四栏帧（GT + 三臂，逐帧严格同机位；只拷文件，不渲染）
#   COV    阶段1 覆盖度（覆盖率.png + 验证数据.json）
#   MANIFEST 在 RRF/RENDER/WALK 之后由它们各自写；本段只补写缺失节点
#
# 用法（服务器，cwd=/data/yuexf/3dgs）
# -----------------------------------------------------------------------------
#   串行跑全部（占一卡，约 4-5 GPU 小时）：
#     CUDA_VISIBLE_DEVICES=1 nohup bash code/tools/run_two_budget_three_arm.sh \
#       > /data/yuexf/3dgs/results/run_20260912_两组预算_三臂对比/日志/驱动.log 2>&1 &
#   分片快跑（先 A B 一次，再按方法四卡并行 RRF/RENDER/WALK）：
#     CUDA_VISIBLE_DEVICES=1 SEGMENTS="A B GATE" bash code/tools/run_two_budget_three_arm.sh
#     for i in 0 1 2 3; do M=(MPC CBF Delay MVDR)[$i]; \
#       CUDA_VISIBLE_DEVICES=$i SEGMENTS="RRF RENDER WALK" METHODS="$M" \
#       nohup bash code/tools/run_two_budget_three_arm.sh > .../日志/分片_$M.log 2>&1 & done
#   只过闸不训练：
#     CUDA_VISIBLE_DEVICES=1 SEGMENTS="A GATE" bash code/tools/run_two_budget_three_arm.sh
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
# fd 上限必须**真的**提上去：dataset_readers.readColmapCameras 读相机时会同时开上千个
# PNG，soft=1024 会以 `OSError: [Errno 24] Too many open files` 崩在场景加载里。
# 那个错看起来像"数据集坏了"，实际只是 shell 限制 —— 闸② 的 MLP 一测就曾被它弄成假失败。
FD_NOW=$(ulimit -n 2>/dev/null || echo 0)
[ "$FD_NOW" -ge 8192 ] 2>/dev/null \
  || echo "[警告] fd 上限只有 $FD_NOW（未提到 8192 以上）：场景加载可能 EMFILE 假失败"
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "缺 CUDA_VISIBLE_DEVICES"; exit 2; }

METHODS="${METHODS:-MPC CBF Delay MVDR}"
SEGMENTS="${SEGMENTS:-A B GATE V4 RRF RENDER WALK REAL COV}"
SAVE_ITERS_OVERRIDE="${SAVE_ITERS:-}"        # 空 = 每臂只存末档（见下方注释）
WALK_FRAMES="${WALK_FRAMES:-120}"
TRAJ="${TRAJ:-orbit dolly}"
SUBSET_FULL=0
# 分辨率与 config.yaml:stage_b.resolution 同值。**必须在这里定义**：`set -u` 下漏定义
# 会在跑到那一步时整脚本猝死（本脚本已经因为 echo 里裸引 $RES 死过一次），
# 而 `set -u` 的死法不留产物、只留一行 unbound variable，看起来像训练崩了。
RES="${RES:-2}"

# ---- 服务器路径（取自 /data/yuexf/3dgs/config.yaml，本脚本不写它）------------
PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
DS=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum
VIS=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/blender_visual_dataset
CKPT=/data/yuexf/3dgs/checkpoints/re10k.ckpt
SEED=/data/yuexf/3dgs/results/orig_seed_chkpnt30000.pth
PARITY=/data/yuexf/3dgs/results/run_20260907_185429_parity档_视觉30k基座
A1_SRC=/data/yuexf/3dgs/results/run_rrf40k_sh
A2_SRC=/data/yuexf/3dgs/results/run_20260912_多频段标量分支
# 只用于闸②的「MLP 色头相机等价」一测（**不**当 RRF 臂复用：run_rrf40k_mlp 的 cfg_args
# 无 lr_schedule=exp/pe_*/use_dist，是 V-base 而非 A* 协议，与「新RRF」口径不符）
MLP_SRC=/data/yuexf/3dgs/results/run_rrf40k_mlp
RUN="${RUN:-/data/yuexf/3dgs/results/run_20260912_两组预算_三臂对比}"
LOG="$RUN/日志"
MANI="$RUN/清单"
# REAL 段的选帧用哪个 cameras.json —— 四方法的数据集 `sparse/0/{images,cameras}.txt`
# 与各自 run 的 cameras.json **逐字节相同**（实测 md5 一致），故一份选择表四方法通用。
# **但 GT 伪彩图逐方法不同**：各方法的 images/<name>.png md5 互不相同 ⇒ GT 栏必须逐方法
# 各取各的（用该臂自己的 `test/ours_<档>/gt/`），绝不能四方法共用一张。
# ⚠ 这一行**必须在 A1_SRC 之后**：`set -u` 下引用未定义变量是**直接猝死**，
#   不留产物、只留一行 unbound variable（本行第一版放在 TRAJ 旁，就是这么死的）。
CAM_JSON="${CAM_JSON:-$A1_SRC/MPC/cameras.json}"
# 阶段A 的分组超参：与 config.yaml 的 full profile 逐字一致（K16/V4/G2/1.5M）
export RF_NUM_CLUSTERS=16 RF_VIEWS_PER_CLUSTER=4 RF_GAZES_PER_CLUSTER=2
export RF_MAX_GAUSSIANS=1500000 RF_VOXEL_SIZE=0.0 RF_OPACITY_THRESHOLD=0.05
export RF_MAX_DEPTH=15.0 RF_CLUSTERS_LIMIT=0 RF_BACKGROUND=black
export RF_NEAR=1.0 RF_FAR=100.0 RF_TARGET_FRAMES_PER_CLUSTER=0
export RF_SOURCE_PATH="$VIS"

# A* 协议（口径 §2）—— 四方法逐字相同
A_INIT=8e-3; A_FINAL=8e-5; A_MAXSTEPS=40000
EMBED_DIM=32; MLP_HIDDEN=128; MLP_DEPTH=2; PE_XYZ_L=5; PE_DIR_L=3
NBANDS=2; BANDMIX=mult

# 臂的中文目录名（交付树用；视频标签另用纯 ASCII，见 WALK 段注释）
ARM_DIR_A1=A1_原版物理_原版RRF
ARM_DIR_A2=A2_原版物理_新RRF
ARM_DIR_A3=A3_新物理_新RRF
ARM_LAB_A1=A1_yuanban_RRF
ARM_LAB_A2=A2_yuanban_newRRF
ARM_LAB_A3=A3_MVSplat_newRRF
# 真实机位四栏视频的第 1 栏（原始数据真值）。纯 ASCII，理由同 WALK 段的标签注。
ARM_LAB_GT=GT_zhenzhi

for p in "$DS" "$VIS" "$CKPT" "$SEED"; do
  [ -e "$p" ] || { echo "缺依赖: $p"; exit 1; }
done
cd "$CODE" || exit 9
mkdir -p "$LOG" "$MANI"
echo "==== [$(date +%F_%T)] 两组预算三臂 卡=$CUDA_VISIBLE_DEVICES methods='$METHODS' ===="
echo "     SEGMENTS='$SEGMENTS'  RUN=$RUN"
FAIL=0
seg() { case " $SEGMENTS " in *" $1 "*) return 0;; *) return 1;; esac; }

# 组参数：G_NAME G_PHYS G_RRF G_ABS G_ABS_A3 G_BASE_A3
#
# **三臂的绝对末迭代并不相同**，别被 G_ABS 这个名字骗了：
#   A1/A2 从 orig_seed（上游 30000 步视觉基座）续训 ⇒ 绝对末档 = 30000 + G_RRF
#   A3   从本组**物理基座**续训 ⇒ 绝对末档 = G_PHYS + G_RRF
# 两者要跑出同样的**本地** G_RRF 步，绝对档就必然不同。组1 恰好相等（30000+30000 = 60000
# = 30000+30000）所以看不出来；组2 差得很远：A1/A2 = 50000，A3 = 10000+20000 = **30000**。
# 首轮把三臂一律当 G_ABS 用 ⇒ 组2 A3 被训成本地 40000 步、LR 调度跑到 t=1.0，
# 既不是「省预算」命题，也和 A2 的 t=0.5 不是同一个调度位置（这类错比崩溃难查得多）。
G_NAME=""; G_PHYS=0; G_RRF=0; G_ABS=0; G_ABS_A3=0; G_BASE_A3=""
group_def() {
  case "$1" in
    1) G_NAME=组1_物理30000_RRF30000; G_PHYS=30000; G_RRF=30000; G_ABS=60000
       G_BASE_A3="$PARITY/rf-3dgs_1/chkpnt30000.pth";;
    2) G_NAME=组2_物理10000_RRF20000; G_PHYS=10000; G_RRF=20000; G_ABS=50000
       # **必须带 `rf-3dgs_1/` 这一层**：段B 用 `train.py -m "$G2D/rf-3dgs_1"`，
       # 产物落在 `$G2D/rf-3dgs_1/` 下。首轮跑时这里漏了这层 ⇒ 组2 的 A3 四方法
       # 全部「起点缺失」跳过（三臂里整整少一臂），而 [V3]/其余闸都不会为此报错 ——
       # 是 RENDER 段那句 `[skip] 2 A3 MPC 无 abs50000` 才暴露的。
       G_BASE_A3="$RUN/组2_物理10000_RRF20000/阶段2_物理重建/rf-3dgs_1/chkpnt10000.pth";;
    *) echo "未知组: $1"; return 1;;
  esac
  G_DIR="$RUN/$G_NAME"
  G_ABS_A3=$((G_PHYS + G_RRF))     # A3 的绝对末档 = 本组物理步数 + RRF 本地步数
  return 0
}

# 方法 → 引擎（A2/A3 用；A1 恒为 sh）
method_engine() { case "$1" in MPC|Delay) echo mlp;; MVDR|CBF) echo scalar;; *) echo "";; esac; }

# 臂 → 该臂的绝对末迭代（A1/A2 用 G_ABS，A3 用 G_ABS_A3）。**所有按臂取的档位都必须走这个**，
# 直接写 $G_ABS 就是把「三臂绝对档相同」这个错误假设又请回来了。
arm_abs() { [ "$1" = A3 ] && echo "$G_ABS_A3" || echo "$G_ABS"; }

pngn() { find "$1" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l; }
has_full_renders() {  # $1=模型目录 $2=绝对档；640 张 renders + 640 张 gt 才算全量
  local od="$1/test/ours_$2"
  [ "$(pngn "$od/renders")" -ge 640 ] && [ "$(pngn "$od/gt")" -ge 640 ]
}

# 软链视图：在本 run 树里借源模型的 point_cloud 渲自己的渲染（只读源，绝不写源）
ensure_view() {  # $1=本树臂目录 $2=源模型目录
  local ours="$1" src="$2"
  [ -e "$ours/point_cloud" ] && return 0
  mkdir -p "$ours"
  ln -sfn "$src/point_cloud" "$ours/point_cloud"
  cp -f "$src/cfg_args" "$ours/cfg_args"
  for f in cameras.json input.ply per_view.json; do
    [ -f "$src/$f" ] && ln -sfn "$src/$f" "$ours/$f"
  done
  echo "  [view] 软链视图 $ours → $src（源只读）"
}

# 渲染脚本按引擎分派（**统一 --eval --skip_train**，与方法匹配的那个脚本）
# 注意：render.py（引擎内核）**没有** --subset —— 它本来就是全量渲，故 sh 分支不传该参数。
render_engine() {  # $1=引擎 $2=数据目录 $3=模型目录 $4=绝对档
  local eng="$1" data="$2" md="$3" i="$4" sub="${RENDER_SUBSET:-0}"
  case "$eng" in
    sh)     "$PY" render.py            -s "$data" -m "$md" --iteration "$i" --eval --skip_train --quiet;;
    mlp)    "$PY" render_rrf_mlp.py    -s "$data" -m "$md" --iteration "$i" --eval --skip_train --subset "$sub" --quiet;;
    scalar) "$PY" render_rrf_scalar.py -s "$data" -m "$md" --iteration "$i" --eval --skip_train --subset "$sub" --quiet;;
    *) return 2;;
  esac
}

# 写单节点清单（每节点一个文件 ⇒ 分片安全、部分跑完也有可用的清单）
write_mani() {  # $1=组 $2=臂键 $3=方法 $4=引擎 $5=绝对档 $6=源模型目录 $7=本树臂目录 $8=渲染来源目录 $9=是否复用
  local g="$1" ak="$2" M="$3" eng="$4" abs="$5" src="$6" ours="$7" rend="$8" reused="$9"
  local f="$MANI/${g}_${ak}_${M}.json"
  G="$g" AK="$ak" M="$M" ENG="$eng" ABS="$abs" SRC="$src" OURS="$ours" REND="$rend" \
  REUSED="$reused" RUNROOT="$RUN" F="$f" CODE="$CODE" "$PY" - <<'PYEOF'
import json, os, re, sys
ply = os.path.join(os.environ["SRC"], "point_cloud", "iteration_%s" % os.environ["ABS"], "point_cloud.ply")
n = 0
try:
    with open(ply, "rb") as fh:
        m = re.search(rb"element vertex (\d+)", fh.read(4096))
        n = int(m.group(1)) if m else 0
except OSError:
    pass
abs_i, ak, M, eng = os.environ["ABS"], os.environ["AK"], os.environ["M"], os.environ["ENG"]
it = os.path.join(os.environ["SRC"], "point_cloud", "iteration_%s" % abs_i)
side = {"scalar": "color_scalar_%s.pth" % abs_i, "mlp": "color_mlp_%s.pth" % abs_i}.get(eng)
rec = {
    "组": os.environ["G"], "臂": ak, "方法": M, "引擎": eng, "绝对迭代": int(abs_i),
    "源模型目录": os.environ["SRC"], "本树臂目录": os.environ["OURS"],
    # ⚠ `是否复用` 单看有两个意思，必须拆开 —— 它指的是**640 张渲染**是否直接取自基线，
    # **不是**指模型是否复用。二者确实会不一致：组2 A2 的 MVDR/CBF **模型**是只读复用
    # `run_20260912_多频段标量分支` 的，但那个 run 只有 48 张渲染，640 张是本树新渲的
    # ⇒ 模型复用而渲染不复用。只留一个 `是否复用` 会让读清单的人得出相反结论。
    "是否复用": os.environ["REUSED"] == "1",          # 保留：= 渲染是否复用（既有消费者沿用）
    "渲染是否复用": os.environ["REUSED"] == "1",
    "模型是否复用": os.path.normpath(os.environ["SRC"]) != os.path.normpath(os.environ["OURS"]),
    "渲染来源目录": os.environ["REND"],
    "结果json": os.path.join(os.environ["OURS"], "results.json"),
    "定性抽帧": os.path.join(os.environ["OURS"], "定性抽帧"),
    "点云": ply, "侧车": os.path.join(it, side) if side else None,
    "高斯数": n, "本树相对路径": os.path.relpath(os.environ["OURS"], os.environ["RUNROOT"]),
}
rec["复用说明"] = (
    "模型与渲染都复用基线" if rec["模型是否复用"] and rec["渲染是否复用"] else
    "模型只读复用基线，640 渲染为本树新渲（基线只有部分渲染）" if rec["模型是否复用"] else
    "模型与渲染都是本树产物（渲染可能复用了同一条臂已渲的档）" if rec["渲染是否复用"] else
    "模型与渲染都是本树新产物")
# 头部参数量：三臂参数量不同，柱状图旁必须给出来，免得把「参数更多」读成「方法更好」。
# 引擎 sh 的参数量记 0 并附说明（颜色就在 ply 的 SH 系数里，不是独立头部）。
rec["头部参数"] = 0 if eng == "sh" else None
rec["头部参数说明"] = ("颜色 = ply 里的逐步 SH 系数（f_dc + f_rest），无独立颜色头"
                       if eng == "sh" else "")
if side:
    try:
        sys.path.insert(0, os.environ.get("CODE", "."))
        if eng == "scalar":
            from splat.color_scalar import load_scalar_state
            h = load_scalar_state(rec["侧车"], device="cpu")[0]
        else:
            from splat.color_mlp import load_color_state
            h = load_color_state(rec["侧车"], device="cpu")[0]
        rec["头部参数"] = int(sum(p.numel() for p in h.parameters()))
    except Exception as e:                      # 拿不到就如实记 None，不编数
        rec["头部参数说明"] = "读取失败: %s" % e
os.makedirs(os.path.dirname(os.environ["F"]), exist_ok=True)
with open(os.environ["F"], "w", encoding="utf-8") as fh:
    json.dump(rec, fh, ensure_ascii=False, indent=2)
print("  [清单] %s %s %s  高斯=%d  复用=%s" % (
    rec["组"], ak, M, n, "是" if rec["是否复用"] else "否"))
PYEOF
}

# =============================================================================
# 段 A：阶段A MVSplat init 点云
# =============================================================================
if seg A; then
  echo; echo "[$(date +%F_%T)] ==== 段A：MVSplat init 点云 ===="
  A_PLY="$CODE/outputs/rf_init/init.ply"
  if [ -f "$RUN/阶段1_MVSplat初始化点云/init.ply" ]; then
    echo "  阶段A 产物已存在，跳过"
  else
    [ -f "$A_PLY" ] && echo "  [旧] 现存 outputs/rf_init/init.ply md5=$(md5sum "$A_PLY" | cut -d' ' -f1)（随后被覆盖）"
    rm -rf outputs/rf_init
    if "$PY" init_cloud.py mode=test +experiment=re10k checkpointing.load="$CKPT" \
         > "$LOG/stageA_mvsplat.log" 2>&1 && [ -f "$A_PLY" ]; then
      mkdir -p "$RUN/阶段1_MVSplat初始化点云"
      cp "$A_PLY" "$RUN/阶段1_MVSplat初始化点云/init.ply"
      cp outputs/rf_init/init_view.ply "$RUN/阶段1_MVSplat初始化点云/init_view.ply"
      cp outputs/rf_init/clusters.json "$RUN/阶段1_MVSplat初始化点云/clusters.json"
    else
      echo "  STAGE_A_FAIL"; tail -30 "$LOG/stageA_mvsplat.log"; FAIL=1
    fi
  fi
  # V1：与 parity 的云对拍（init_cloud 已 torch.manual_seed(0) + kmeans(seed=0)，应逐位相同）
  NEW_MD=$(md5sum "$RUN/阶段1_MVSplat初始化点云/init.ply" 2>/dev/null | cut -d' ' -f1)
  PAR_MD=$(md5sum "$PARITY/mvsplat/点云/init.ply" 2>/dev/null | cut -d' ' -f1)
  OLD_MD=$(md5sum /data/yuexf/3dgs/results/mvsplat_clouds/init.ply 2>/dev/null | cut -d' ' -f1)
  echo "  [V1] 本次 init.ply      md5=${NEW_MD:-缺}"
  echo "  [V1] parity run init.ply md5=${PAR_MD:-缺}"
  echo "  [V1] mvsplat_clouds(服务器) md5=${OLD_MD:-缺}"
  [ "${NEW_MD:-x}" = "${PAR_MD:-y}" ] \
    && echo "  [V1] PASS：阶段A 可复现，组1/组2 的新物理起点云同源" \
    || { echo "  [V1] FAIL：与 parity 不一致 —— 有静默配置漂移，必须查清（不阻断后续，如实登记）"; FAIL=1; }
fi

# =============================================================================
# 段 B：组2 的 10000 步物理重建（重置只在前半；不带 --checkpoint_iterations 拿不到 A3 的基座）
# =============================================================================
if seg B; then
  echo; echo "[$(date +%F_%T)] ==== 段B：组2 物理重建 10000 步 ===="
  G2D="$RUN/组2_物理10000_RRF20000/阶段2_物理重建"
  mkdir -p "$G2D"
  if [ -f "$G2D/rf-3dgs_1/chkpnt10000.pth" ]; then
    echo "  阶段B 产物已存在，跳过"
  else
    export RF_POINTCLOUD="$RUN/阶段1_MVSplat初始化点云/init.ply"
    [ -f "$RF_POINTCLOUD" ] || { echo "  缺 init.ply，段A 未完成"; FAIL=1; }
    if [ -f "$RF_POINTCLOUD" ]; then
      if [ -n "${RF_VRAM_GB:-}" ]; then TRAIN_PY=( "$PY" "$CODE/tools/_prealloc.py" train.py )
      else TRAIN_PY=( "$PY" train.py ); fi
      echo "  init 云=$RF_POINTCLOUD  分辨率=$RES  重置窗口=densify_until_iter 5000 / interval 1000"
      if "${TRAIN_PY[@]}" -s "$VIS" -m "$G2D/rf-3dgs_1" --resolution "$RES" \
            --iterations 10000 --save_iterations 5000 10000 \
            --checkpoint_iterations 10000 \
            --densify_until_iter 5000 --opacity_reset_interval 1000 \
            > "$LOG/stageB_visual_10000.log" 2>&1 \
         && [ -f "$G2D/rf-3dgs_1/chkpnt10000.pth" ]; then
        echo "  阶段B 完成"
      else
        echo "  STAGE_B_FAIL"; tail -30 "$LOG/stageB_visual_10000.log"; FAIL=1
      fi
    fi
    unset RF_POINTCLOUD
  fi
  # [V2a 修正] 这里**不能**断言 cfg_args 里有 densify_until_iter / opacity_reset_interval：
  # code/train.py:33 把 `dataset`（= lp.extract(args)，**只有 ModelParams 那一组**）交给
  # prepare_output_and_logger，:156 写下的就是这个 namespace ⇒ 本仓库的 cfg_args 里
  # 永远只有 sh_degree/source_path/model_path/images/resolution/white_background/
  # data_device/eval 八个字段，**不含任何优化超参**。
  # 原先这里 grep 这两个键并判 FAIL —— 那是个**永不通过的假闸**（与「grep 日志找 reset」
  # 同一个错误类：查了一个根本不承载该信息的数据源）。真正的证据是闸①：它用 5001 步
  # 探针**实测** opacity 的塌陷位置，验的是执行路径而不是参数值。故这里只登记、不判定。
  G2C="$G2D/rf-3dgs_1/cfg_args"
  echo "  [V2a] cfg_args（只含 ModelParams，非本闸证据）: $(tr -d '\n' < "$G2C" 2>/dev/null | tr -s ' ')"
  echo "  [V2a] 重置窗口的判定见段GATE 的闸①（5001 步探针实测 opacity max 的塌陷位置）"
  [ -f "$G2D/rf-3dgs_1/chkpnt10000.pth" ] \
    && echo "  [V2a] chkpnt10000.pth 存在 ✓（组2 A3 的物理基座）" \
    || { echo "  [V2a] FAIL：chkpnt10000.pth 缺失 —— 组2 A3 无基座"; FAIL=1; }
fi

# =============================================================================
# 段 SRC：写每组的「来源.txt」（复用臂与本组物理基座的只读来源 + md5）
# =============================================================================
# 复用臂不复制模型（模型 240 MB/档 ×8），只留一份可核对的来源说明；本脚本绝不写那些目录。
if seg SRC || seg RRF; then
  echo; echo "[$(date +%F_%T)] ==== 段SRC：来源.txt ===="
  md5f() { [ -f "$1" ] && md5sum "$1" | cut -d' ' -f1 || echo "缺"; }
  for G in 1 2; do
    group_def "$G" || continue
    mkdir -p "$G_DIR/阶段1_MVSplat初始化点云" "$G_DIR/阶段2_物理重建"
    {
      echo "组$G 物理基座来源（只读，未复制）"
      echo "======================================"
      if [ "$G" = 1 ]; then
        echo "物理重建 = 复用 parity run 的 30000 步基座（本组不新训物理）"
        echo "  路径: $PARITY/rf-3dgs_1/chkpnt30000.pth"
        echo "  md5 : $(md5f "$PARITY/rf-3dgs_1/chkpnt30000.pth")"
        echo "  其 init 点云: $PARITY/mvsplat/点云/init.ply"
        echo "  md5 : $(md5f "$PARITY/mvsplat/点云/init.ply")"
        echo "  （该基座即 MVSplat init 续训到 30000 步的产物；本组 A3 从它起跑 RRF）"
      else
        echo "物理重建 = 本 run 新训的 10000 步（段B）"
        echo "  路径: $G_DIR/阶段2_物理重建/rf-3dgs_1/chkpnt10000.pth"
        echo "  md5 : $(md5f "$G_DIR/阶段2_物理重建/rf-3dgs_1/chkpnt10000.pth")"
        echo "  其 init 点云: $RUN/阶段1_MVSplat初始化点云/init.ply"
        echo "  md5 : $(md5f "$RUN/阶段1_MVSplat初始化点云/init.ply")"
        echo "  重置窗口: --densify_until_iter 5000 --opacity_reset_interval 1000（后 5000 步不重置）"
      fi
    } > "$G_DIR/阶段2_物理重建/来源.txt"
    echo "  [SRC] $G_DIR/阶段2_物理重建/来源.txt"
    {
      echo "阶段1 MVSplat init 点云（本组的口径来源）"
      echo "======================================"
      if [ "$G" = 1 ]; then
        echo "本组物理基座是历史 parity run，其 init 云为:"
        echo "  $PARITY/mvsplat/点云/init.ply"
        echo "  md5: $(md5f "$PARITY/mvsplat/点云/init.ply")"
        echo "  （组1 不重跑阶段A；组2 的 init 云在 $RUN/阶段1_MVSplat初始化点云/）"
      else
        echo "本组用的 init 云 = 本 run 阶段A 产物:"
        echo "  $RUN/阶段1_MVSplat初始化点云/init.ply"
        echo "  md5: $(md5f "$RUN/阶段1_MVSplat初始化点云/init.ply")"
        echo "  （init.ply / init_view.ply 只放在本 run 根目录下这一份，不复制 34 MB 点云）"
      fi
    } > "$G_DIR/阶段1_MVSplat初始化点云/来源.txt"
    # 复用臂来源
    for M in $METHODS; do
      for AK in A1 A2; do
        case "$AK" in
          A1) SRC="$A1_SRC/$M";;
          A2) [ "$M" = MVDR ] || [ "$M" = CBF ] || continue; SRC="$A2_SRC/$M";;
        esac
        [ -d "$SRC" ] || continue
        eval "AD=\$ARM_DIR_$AK"
        D="$G_DIR/阶段3_RRF重建/$AD/$M"
        [ -d "$D" ] || continue
        { echo "本臂模型为**只读复用**，未复制（渲染/评测只往本 run 树写）"
          echo "  源: $SRC"
          echo "  档: iteration_$G_ABS（绝对迭代）"
          echo "  复用的渲染: $SRC/test/ours_$G_ABS/"
          echo "  results.json: 本树 $D/results.json（由 metrics_fast.py 对上面的渲染现场算）"
        } > "$D/来源.txt"
      done
    done
  done
fi

# =============================================================================
# 段 GATE：①重置时刻的实证验证 ②三色头相机等价性自检
# =============================================================================
# V2 的由来：原计划的「grep 训练日志断言 reset 只出现在 1000..4000」**做不到** ——
# gaussian_model.py:210 reset_opacity() 一行都不打印，日志里没有 reset 事件。
# 改成**实证**：同一份配置在 5000 步刻度附近存档，直接量不透明度上限的跳变。重置会把
# 所有高斯压到 ≤0.01（gaussian_model.py:211），于是：
#   * 4000→4001 间若有重置，max(opacity) 必须从"训练后值"塌到 ≈0.01
#   * 4999→5000→5001 间若无重置，max(opacity) 应保持高位、不塌
# 这比读日志强：它检验的是**真实执行路径**，不是参数值。
if seg GATE; then
  echo; echo "[$(date +%F_%T)] ==== 段GATE ===="
  GD="$RUN/_闸_重置验证"
  if [ -f "$GD/done" ]; then
    echo "  闸① 已跑过，跳过"
  else
    mkdir -p "$GD"
    export RF_POINTCLOUD="$RUN/阶段1_MVSplat初始化点云/init.ply"
    if "${PY}" train.py -s "$VIS" -m "$GD/rf-3dgs_1" --resolution "$RES" \
          --iterations 5001 --save_iterations 3999 4000 4001 4999 5000 5001 \
          --densify_until_iter 5000 --opacity_reset_interval 1000 \
          > "$LOG/gate_reset.log" 2>&1; then
      "$PY" - "$GD/rf-3dgs_1" <<'PYEOF'
import sys, glob, os, json
import numpy as np
from plyfile import PlyData
md = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(md, "point_cloud", "iteration_*", "point_cloud.ply")),
                key=lambda x: int(x.split("iteration_")[1].split("/")[0])):
    it = int(p.split("iteration_")[1].split("/")[0])
    op = np.asarray(PlyData.read(p).elements[0]["opacity"])
    rows.append((it, float(op.max()), float((op < 0.02).mean())))
print("  [闸①] 重置时刻实证（opacity 是 ply 里的原始值，未过 sigmoid）")
print("        档位   max(opacity)   占比<0.02")
for it, mx, frac in rows:
    print("        %5d   %10.4f   %8.1f%%" % (it, mx, frac * 100))
d = dict((it, mx) for it, mx, _ in rows)
verdict = []
if 4000 in d and 4001 in d:
    verdict.append(("4000→4001 有重置（max 塌到 ≤0.05）", bool(d[4001] <= 0.05)))
for a, b in ((4999, 5000), (5000, 5001)):
    if a in d and b in d:
        verdict.append(("%d→%d 无重置（max 保持 >0.05）" % (a, b), bool(d[b] > 0.05)))
ok = all(v for _, v in verdict)
for k, v in verdict:
    print("        [%s] %s" % ("PASS" if v else "FAIL", k))
print("  [闸①] %s" % ("PASS：重置只在 1000..4000，5000 之后零次" if ok else "FAIL：重置窗口与预期不符"))
gd = os.path.dirname(md)
with open(os.path.join(gd, "done"), "w") as fh:
    fh.write("ok" if ok else "fail")
with open(os.path.join(gd, "闸1_读数.json"), "w", encoding="utf-8") as fh:
    json.dump({
        "说明": "reset_opacity()（gaussian_model.py:210）不打印任何日志，故不读日志、直接量"
                "不透明度上限在存档刻度上的跳变；opacity 为 ply 原始值（未过 sigmoid）",
        "配置": {"densify_until_iter": 5000, "opacity_reset_interval": 1000,
                 "iterations": 5001, "resolution": 2},
        "读数": [{"档位": it, "max_opacity": mx, "占比_lt_0.02": fr} for it, mx, fr in rows],
        "判据": [{"断言": k, "通过": v} for k, v in verdict],
        "结论": "PASS" if ok else "FAIL",
    }, fh, ensure_ascii=False, indent=2)
raise SystemExit(0 if ok else 1)
PYEOF
      [ $? -eq 0 ] || FAIL=1
    else
      echo "  闸① 训练失败"; tail -20 "$LOG/gate_reset.log"; FAIL=1
    fi
    unset RF_POINTCLOUD
  fi

  # 闸② 三色头相机等价性（每头各跑一次；阈值 1e-6，**不与已存 PNG 比**）
  echo "  闸② 三色头相机等价性自检"
  gate_selfcheck() {  # $1=标签 $2=模型目录 $3=绝对档 $4=数据集 $5=head  $6=日志名(ASCII)
    local lab="$1" md="$2" i="$3" data="$4" head="$5" tag="$6"
    local out
    # **全量输出落盘**：原先只把匹配行塞进 $out、其余(含 traceback)丢弃 ⇒ 闸② 报 FAIL 时
    # 看不到原因，只能手动重跑一遍才查出是 TypeError。错误现场必须留档。
    "$PY" tools/render_rrf_walk.py -s "$data" -m "$md" --iteration "$i" \
          --eval --head "$head" --selfcheck --quiet \
          > "$LOG/gate2_${tag}.log" 2>&1
    out=$(grep -a "颜色头\|R_syn\|FoV\|两次渲染\|selfcheck \[" "$LOG/gate2_${tag}.log")
    # 分隔符用 `|` 而不是 `/`：标签里有 run_rrf40k_sh/MPC 这样的斜杠，
    # 用 s/.../.../ 会被 shell 变量里的斜杠搞成 "unknown option to `s'"（踩过）
    echo "$out" | sed "s|^|      [$lab] |"
    if echo "$out" | grep -q "selfcheck \[.*\] PASS"; then
      echo "      [$lab] 闸② PASS（全量日志 $LOG/gate2_${tag}.log）"
    else
      echo "      [$lab] 闸② FAIL —— 原始输出末尾："
      tail -12 "$LOG/gate2_${tag}.log" | sed "s|^|        |"
      FAIL=1
    fi
  }
  gate_selfcheck "SH    (run_rrf40k_sh/MPC)"   "$A1_SRC/MPC"  60000 "$DS/3dgs_MPC_100"  sh   sh
  gate_selfcheck "MLP   (run_rrf40k_mlp/MPC)"  "$MLP_SRC/MPC" 60000 "$DS/3dgs_MPC_100"  auto mlp
  gate_selfcheck "scalar(多频段/MVDR)"          "$A2_SRC/MVDR" 60000 "$DS/3dgs_MVDR_100" auto scalar
fi

# =============================================================================
# 段 V4：三臂口径审计（证据取自 sidecar 自身，**不**取 cfg_args）
# =============================================================================
# 为什么换了证据源：`code/train.py:33` 交给 `prepare_output_and_logger` 的是 `dataset`
# （= ModelParams 那一组），`:156` 写下的 cfg_args 只有 sh_degree/source_path/model_path/
# images/resolution/white_background/data_device/eval 八个字段 ⇒ **不含任何超参**。
# 所以「这一臂是 SH 还是 MLP/标量头、是不是 A* 协议」只能这么取证：
#   · 引擎 = sh  ⇒ 取证于 **没有颜色 sidecar**（颜色只可能来自 ply 的逐步 SH 系数）
#   · 引擎 = mlp/scalar ⇒ 取证于 `color_{mlp,scalar}_<abs>.pth` 存在，且它**自己存的 cfg**
#     里 pe_xyz_L/pe_dir_L/use_dist 等 A* 标记、以及 head_type/n_bands/band_mix 对得上
# 这条闸**不是走过场**：A2 的 MVDR/CBF 是复用 `run_20260912_多频段标量分支` 的，
# 它到底是不是 A× 配置（n_bands=2 / band_mix=mult）此前只是"我认定"，这一步才真的验。
if seg V4; then
  echo; echo "[$(date +%F_%T)] ==== 段V4：三臂口径审计 ===="
  V4D="$RUN/_闸_口径审计"; mkdir -p "$V4D"
  rm -f "$V4D/口径审计.jsonl"     # jsonl 是追加写的（为将来并行分片留的），重跑必须先清
                                # 否则上一轮的条目会和这一轮叠在一起，汇总里"条目数"翻倍
  V4_MISS=0
  for G in 1 2; do
    group_def "$G" || { FAIL=1; continue; }
    for M in $METHODS; do
      for AK in A1 A2 A3; do
        eval "AD=\$ARM_DIR_$AK"
        ABS_IT="$(arm_abs "$AK")"
        case "$AK" in
          A1) SRC="$A1_SRC/$M"; ENG=sh;;
          A2) if [ "$M" = MVDR ] || [ "$M" = CBF ]; then SRC="$A2_SRC/$M"
              else SRC="$G_DIR/阶段3_RRF重建/$AD/$M"; fi
              ENG=$(method_engine "$M");;
          A3) SRC="$G_DIR/阶段3_RRF重建/$AD/$M"; ENG=$(method_engine "$M");;
        esac
        if [ ! -d "$SRC/point_cloud/iteration_${ABS_IT}" ]; then
          # **缺失即 FAIL，不 skip**：三臂对比里少一臂比报错危险得多 —— 柱子照画、
          # 缺的那根只是"矮一截"或空缺，混进交付物看不出来。
          # 首轮就是这里以一句极不显眼的 `[skip] 2 A3 MPC 无 abs50000` 掩盖了
          # 「组2 A3 四方法全没跑」（G_BASE_A3 少一层 rf-3dgs_1 所致）。
          echo "  [V4] FAIL 组$G $AK $M 缺 abs$ABS_IT 档: $SRC"
          echo "       —— 三臂缺一臂，交付物不完整，必须补齐后重跑本段"
          FAIL=1; V4_MISS=$((V4_MISS + 1)); continue
        fi
        if CODE="$CODE" "$PY" - "$G" "$AK" "$M" "$ENG" "$ABS_IT" "$G_RRF" "$SRC" "$V4D" <<'PYEOF'
import glob, json, os, sys
g, ak, m, eng, absit, rrf, src, outd = sys.argv[1:9]
absit, rrf = int(absit), int(rrf)
D = os.path.join(src, "point_cloud", "iteration_%d" % absit)
rec = {"组": int(g), "臂": ak, "方法": m, "期望引擎": eng, "绝对迭代": absit,
       "本地步数": rrf, "模型目录": src, "档目录": D}
errs = []
rec["有 point_cloud.ply"] = os.path.isfile(os.path.join(D, "point_cloud.ply"))
if not rec["有 point_cloud.ply"]:
    errs.append("缺 point_cloud.ply")
rec["color_mlp_*.pth"] = sorted(os.path.basename(x) for x in glob.glob(os.path.join(D, "color_mlp_*.pth")))
rec["color_scalar_*.pth"] = sorted(os.path.basename(x) for x in glob.glob(os.path.join(D, "color_scalar_*.pth")))

def read_cfg(path):
    import torch
    try:
        sd = torch.load(path, map_location="cpu")
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    return (sd.get("cfg") or {}) if isinstance(sd, dict) else {}

if eng == "sh":
    rec["口径"] = "原版逐点 SH（train.py 引擎）：颜色只在 ply 里，无独立颜色头"
    if rec["color_mlp_*.pth"] or rec["color_scalar_*.pth"]:
        errs.append("A1 本该没有颜色 sidecar，却有 %s —— 颜色未必来自 ply 的 SH"
                    % (rec["color_mlp_*.pth"] + rec["color_scalar_*.pth"]))
else:
    want = "color_%s_%d.pth" % (eng, absit)
    got = rec["color_mlp_*.pth"] if eng == "mlp" else rec["color_scalar_*.pth"]
    rec["期望 sidecar"] = want
    if want not in got:
        errs.append("缺 sidecar %s（该档实有 %s）" % (want, got))
    else:
        cfg = read_cfg(os.path.join(D, want))
        rec["sidecar cfg"] = {k: cfg.get(k) for k in
                              ("head_type", "embed_dim", "mlp_hidden", "mlp_depth",
                               "pe_xyz_L", "pe_dir_L", "use_dist", "n_bands",
                               "band_mix", "init_u", "band_lambda", "lut_n")}
        # A* 协议标记（[[docs/RRF-MLP改进方案_对照Scaffold-GS.md:415]]）
        a_star = (cfg.get("embed_dim") == 32 and cfg.get("mlp_hidden") == 128
                  and cfg.get("mlp_depth") == 2 and cfg.get("pe_xyz_L") == 5
                  and cfg.get("pe_dir_L") == 3 and bool(cfg.get("use_dist")) is True)
        rec["是 A* 协议"] = bool(a_star)
        if not a_star:
            errs.append("sidecar cfg 不是 A* 协议（pe/embed/dist 对不上）: %s" % rec["sidecar cfg"])
        if eng == "scalar":
            rec["口径"] = "A× 标量头（多频段，先混合再查表）"
            if cfg.get("head_type") != "scalar_jet":
                errs.append("head_type 不是 scalar_jet: %r" % cfg.get("head_type"))
            if cfg.get("n_bands") != 2 or cfg.get("band_mix") != "mult":
                errs.append("不是形态 A×（n_bands=%r band_mix=%r，期望 2/mult）"
                            % (cfg.get("n_bands"), cfg.get("band_mix")))
        else:
            rec["口径"] = "MLP 颜色头（V14：A* LR + PE(5,3) + dist 条件）"
        # 参数量（与清单同源，供柱状图旁标注：三臂参数量不同，别把"参数多"读成"方法好"）
        import torch
        sd = torch.load(os.path.join(D, want), map_location="cpu")
        rec["头部参数"] = int(sum(v.numel() for v in sd["head"].values()))
        rec["embedding 行数"] = int(sd["embedding"].shape[0])

rec["结论"] = "PASS" if not errs else "FAIL"
rec["问题"] = errs
line = "[V4] %s 组%d %s %s 引擎=%s 口径=%s%s" % (
    rec["结论"], rec["组"], rec["臂"], rec["方法"], eng, rec.get("口径", "—"),
    ("  问题: " + " / ".join(errs)) if errs else "")
print(line, flush=True)
f = os.path.join(outd, "口径审计.jsonl")
with open(f, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
sys.exit(0 if not errs else 1)
PYEOF
        then :; else FAIL=1; fi
      done
    done
  done
  # 收集成一份 JSON（jsonl 便于并行追加，最后合并；重复运行前先清空）
  if [ -f "$V4D/口径审计.jsonl" ]; then
    "$PY" - "$V4D/口径审计.jsonl" "$V4D/口径审计.json" <<'PYEOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
bad = [r for r in recs if r["结论"] != "PASS"]
json.dump({"条目数": len(recs), "通过": len(recs) - len(bad),
           "未通过": len(bad), "明细": recs},
          open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("  [V4] 汇总 %d 条，通过 %d，未通过 %d -> %s"
      % (len(recs), len(recs) - len(bad), len(bad), sys.argv[2]))
PYEOF
  fi
  # 完整性：2 组 × 3 臂 × 4 方法 = 24 个节点，一个都不能少
  N_EXP=24
  if [ "$V4_MISS" = 0 ]; then
    echo "  [V4] 完整性 PASS：$N_EXP 个节点（2组 × 3臂 × $((N_EXP / 6))方法）全在"
  else
    echo "  [V4] 完整性 FAIL：缺 $V4_MISS 个节点（期望 $N_EXP）—— 交付物少臂，必须补跑 RRF 后重跑本段"
    FAIL=1
  fi
fi

# =============================================================================
# 段 RRF：12 个节点（组×臂×方法）。A1 全复用；A2 的 MVDR/CBF 复用；其余新训。
# =============================================================================
rrf_train() {  # $1=数据目录 $2=模型目录 $3=起点chkpnt $4=绝对末迭代 $5=引擎 $6=日志名 $7=起点绝对迭代
  local data="$1" md="$2" base="$3" abs="$4" eng="$5" lg="$6" first="$7"
  local save="${SAVE_ITERS_OVERRIDE:-$abs}"
  if [ -f "$md/point_cloud/iteration_${abs}/point_cloud.ply" ]; then
    echo "  [T] 末档已存在，跳过训练"; return 0
  fi
  echo "  [T] 引擎=$eng  起点=$(basename "$base")  绝对 $abs（本地 $((abs - first))，LR 调度 t=$((abs - first))/40000）存档='$save'"
  case "$eng" in
    mlp) "$PY" train_rrf_mlp.py -s "$data" -m "$md" \
            --start_checkpoint "$base" --iterations "$abs" --save_iterations $save \
            --densify_until_iter 0 --data_device cpu --quiet \
            --embed_dim "$EMBED_DIM" --mlp_hidden "$MLP_HIDDEN" --mlp_depth "$MLP_DEPTH" \
            --lr_schedule exp --mlp_lr_init "$A_INIT" --mlp_lr_final "$A_FINAL" \
            --lr_max_steps "$A_MAXSTEPS" --pe_xyz_L "$PE_XYZ_L" --pe_dir_L "$PE_DIR_L" --use_dist \
            > "$LOG/$lg" 2>&1;;
    scalar) "$PY" train_rrf_scalar.py -s "$data" -m "$md" \
            --start_checkpoint "$base" --iterations "$abs" --save_iterations $save \
            --densify_until_iter 0 --data_device cpu --quiet \
            --embed_dim "$EMBED_DIM" --mlp_hidden "$MLP_HIDDEN" --mlp_depth "$MLP_DEPTH" \
            --lr_schedule exp --mlp_lr_init "$A_INIT" --mlp_lr_final "$A_FINAL" \
            --lr_max_steps "$A_MAXSTEPS" --pe_xyz_L "$PE_XYZ_L" --pe_dir_L "$PE_DIR_L" --use_dist \
            --n_bands "$NBANDS" --band_mix "$BANDMIX" \
            > "$LOG/$lg" 2>&1;;
    *) echo "  未知引擎 $eng"; return 2;;
  esac
}

if seg RRF; then
  echo; echo "[$(date +%F_%T)] ==== 段RRF（12 节点）===="
  for G in 1 2; do
    group_def "$G" || { FAIL=1; continue; }
    for M in $METHODS; do
      DATA="$DS/3dgs_${M}_100"
      [ -d "$DATA" ] || { echo "  [$M] 数据缺失"; FAIL=1; continue; }
      ENG_M=$(method_engine "$M"); [ -n "$ENG_M" ] || { echo "  [$M] 未知方法"; FAIL=1; continue; }
      for AK in A1 A2 A3; do
        eval "AD=\$ARM_DIR_$AK"
        ABS_IT="$(arm_abs "$AK")"
        # ---- A1：全复用 run_rrf40k_sh，只核对 ----
        if [ "$AK" = A1 ]; then
          has_full_renders "$A1_SRC/$M" "$ABS_IT" \
            || echo "  [警告] A1 $M abs$ABS_IT 源渲染不是 640 张，RENDER 段会补渲"
          continue
        fi
        # ---- A2：MVDR/CBF 复用多频段 run（48 张渲染，RENDER 段补渲到 640）----
        if [ "$AK" = A2 ] && { [ "$M" = MVDR ] || [ "$M" = CBF ]; }; then
          [ -d "$A2_SRC/$M/point_cloud/iteration_${ABS_IT}" ] \
            || { echo "  [$M] 复用源缺 abs$ABS_IT: $A2_SRC/$M"; FAIL=1; }
          continue
        fi
        # ---- 需要新训 ----
        MD="$G_DIR/阶段3_RRF重建/$AD/$M"
        BASE="$SEED"; FIRST_ITER=30000
        [ "$AK" = A3 ] && { BASE="$G_BASE_A3"; FIRST_ITER="$G_PHYS"; }
        [ -f "$BASE" ] || { echo "  [$AK $M] 起点缺失: $BASE"; FAIL=1; continue; }
        unset RF_POINTCLOUD     # 频谱集用自身 colmap 点云；init 云会污染（run_end_to_end.sh:142 的坑）
        mkdir -p "$MD"
        rrf_train "$DATA" "$MD" "$BASE" "$ABS_IT" "$ENG_M" "rrf_${G}_${AK}_${M}.log" "$FIRST_ITER" || {
          echo "  RRF_FAIL ${G}_${AK}_${M}"; tail -20 "$LOG/rrf_${G}_${AK}_${M}.log"; FAIL=1; continue; }
        # V3：本地步数 = 绝对 − first_iter，必须与组定义一致
        echo "  [V3] ${G}_${AK}_${M}: 绝对 $ABS_IT − first_iter $FIRST_ITER = $((ABS_IT - FIRST_ITER))（期望 $G_RRF）"
        [ $((ABS_IT - FIRST_ITER)) -eq "$G_RRF" ] || { echo "  [V3] FAIL：本地步数与组定义不符"; FAIL=1; }
        # 起点核对：日志里的 start_checkpoint 必须是预期的那个 base
        grep -aq "$(basename "$BASE")" "$LOG/rrf_${G}_${AK}_${M}.log" \
          && echo "  [V3] 起点含 $(basename "$BASE") ✓" \
          || echo "  [V3] 警告：日志里找不到起点文件名（train_rrf_*.py 未必打印，非致命）"
        echo "  [T] ${G}_${AK}_${M} 完成"
      done
    done
  done
fi

# =============================================================================
# 段 RENDER：每节点 640 全量渲染 + 每臂 results.json（渲染能复用就复用，只读）
# =============================================================================
if seg RENDER; then
  echo; echo "[$(date +%F_%T)] ==== 段RENDER ===="
  for G in 1 2; do
    group_def "$G" || { FAIL=1; continue; }
    for M in $METHODS; do
      DATA="$DS/3dgs_${M}_100"
      ENG_M=$(method_engine "$M")
      for AK in A1 A2 A3; do
        eval "AD=\$ARM_DIR_$AK"; eval "AL=\$ARM_LAB_$AK"
        ABS_IT="$(arm_abs "$AK")"
        case "$AK" in
          A1) SRC="$A1_SRC/$M"; ENG=sh;;
          A2) if [ "$M" = MVDR ] || [ "$M" = CBF ]; then SRC="$A2_SRC/$M"; else
                SRC="$G_DIR/阶段3_RRF重建/$AD/$M"; fi
              ENG="$ENG_M";;
          A3) SRC="$G_DIR/阶段3_RRF重建/$AD/$M"; ENG="$ENG_M";;
        esac
        OURS="$G_DIR/阶段3_RRF重建/$AD/$M"
        [ -d "$SRC/point_cloud/iteration_${ABS_IT}" ] \
          || { echo "  [skip] $G $AK $M 无 abs$ABS_IT: $SRC"; FAIL=1; continue; }
        REUSED=1
        if [ "$SRC" = "$OURS" ] || ! has_full_renders "$SRC" "$ABS_IT"; then
          REUSED=0
          ensure_view "$OURS" "$SRC"
          if ! has_full_renders "$OURS" "$ABS_IT"; then
            echo "  [R] $G $AK $M 渲 640（引擎 $ENG）"
            render_engine "$ENG" "$DATA" "$OURS" "$ABS_IT" \
              > "$LOG/render_${G}_${AK}_${M}.txt" 2>&1 \
              || { echo "  RENDER_FAIL ${G}_${AK}_${M}"; tail -12 "$LOG/render_${G}_${AK}_${M}.txt"; FAIL=1; continue; }
          fi
          REND="$OURS"
        else
          echo "  [R] $G $AK $M 复用源 640 张渲染（只读）: $SRC"
          REND="$SRC"
        fi
        mkdir -p "$OURS"
        if [ ! -f "$OURS/results.json" ]; then
          "$PY" tools/metrics_fast.py -m "$REND" --iterations "$ABS_IT" --subset "$SUBSET_FULL" \
              --out "$OURS/results.json" > "$LOG/metrics_${G}_${AK}_${M}.txt" 2>&1 \
            || { echo "  METRICS_FAIL ${G}_${AK}_${M}"; tail -12 "$LOG/metrics_${G}_${AK}_${M}.txt"; FAIL=1; continue; }
        fi
        echo "  [640] $G $AK $M $(cat "$OURS/results.json" 2>/dev/null | tr -d '\n ')"
        # 定性抽帧：确定性 4 帧（步长 640/4=160 → 0/160/320/480）拷进本树。
        # 复用臂的渲染在基线目录里（只读），拷这 4 帧就够出对照图，之后拉取只需 tar 本树。
        QD="$OURS/定性抽帧"
        if [ ! -d "$QD/renders" ]; then
          mkdir -p "$QD/renders" "$QD/gt"
          for n in 00000 00160 00320 00480; do
            cp -f "$REND/test/ours_${ABS_IT}/renders/$n.png" "$QD/renders/" 2>/dev/null
            cp -f "$REND/test/ours_${ABS_IT}/gt/$n.png" "$QD/gt/" 2>/dev/null
          done
        fi
        echo "  [抽帧] $G $AK $M renders=$(pngn "$QD/renders") gt=$(pngn "$QD/gt") $REND"
        [ "$(pngn "$QD/renders")" -ge 4 ] || { echo "  [抽帧] FAIL：取不到 4 帧"; FAIL=1; }
        write_mani "$G" "$AK" "$M" "$ENG" "$ABS_IT" "$SRC" "$OURS" "$REND" "$REUSED"
      done
    done
  done
fi

# =============================================================================
# 段 WALK：环绕/推轨漫游（三种颜色头共用同一批合成相机）
# =============================================================================
# 标签用纯 ASCII（A1_yuanban_RRF 等）：make_walk_video.py 的标签栏字体若无 CJK
# 会把 `×`/中文渲成豆腐块。目录名仍是中文（见 ARM_DIR_*）。
if seg WALK; then
  echo; echo "[$(date +%F_%T)] ==== 段WALK（$TRAJ，$WALK_FRAMES 帧）===="
  for G in 1 2; do
    group_def "$G" || { FAIL=1; continue; }
    VROOT="$G_DIR/视频"
    for M in $METHODS; do
      DATA="$DS/3dgs_${M}_100"
      for AK in A1 A2 A3; do
        eval "AD=\$ARM_DIR_$AK"; eval "AL=\$ARM_LAB_$AK"
        ABS_IT="$(arm_abs "$AK")"
        case "$AK" in
          A1) SRC="$A1_SRC/$M"; HEADF=sh;;
          A2) if [ "$M" = MVDR ] || [ "$M" = CBF ]; then SRC="$A2_SRC/$M"; else
                SRC="$G_DIR/阶段3_RRF重建/$AD/$M"; fi
              HEADF=auto;;
          A3) SRC="$G_DIR/阶段3_RRF重建/$AD/$M"; HEADF=auto;;
        esac
        [ -d "$SRC/point_cloud/iteration_${ABS_IT}" ] && continue_ok=1 || continue_ok=0
        [ "$continue_ok" = 1 ] || { echo "  [skip] WALK $G $AK $M 无 abs$ABS_IT"; continue; }
        for T in $TRAJ; do
          [ -f "$VROOT/$M/漫游/$AL/$T/轨迹.json" ] && { echo "  [W] $G $AK $M $T 已存在"; continue; }
          echo "  [W] $G $AK $M $T（head=$HEADF）"
          "$PY" tools/render_rrf_walk.py -s "$DATA" -m "$SRC" --iteration "$ABS_IT" \
              --head "$HEADF" --traj "$T" --frames "$WALK_FRAMES" \
              --out "$VROOT/$M/漫游" --arm-out "$AL" --quiet \
              > "$LOG/walk_${G}_${AK}_${M}_${T}.txt" 2>&1 \
            || { echo "  WALK_FAIL ${G}_${AK}_${M}_${T}"; tail -12 "$LOG/walk_${G}_${AK}_${M}_${T}.txt"; FAIL=1; }
        done
      done
    done
  done
fi

# =============================================================================
# 段 REAL：真实机位四栏视频的帧（GT + 三臂，逐帧严格同机位）
# =============================================================================
# 为什么单独一段：环绕/推轨是**数据集没拍过的新机位**，那里**没有真值** ——
# `render_rrf_walk.py --traj orbit` 的合成机位到最近真实机位的中位距离 0.88 m、
# 最大 2.65 m，拿最近的 GT 顶上去是**视差误导**。所以「三方法 + 原始真实结果」
# 的四栏对比不能建在那两条轨迹上，得换一条**真实机位**的轨迹。
#
# 真值一直都在：`test/ours_<档>/gt/<i>.png` 就是数据集原图（实测与
# `images/<test_index[i]>.png` 下采样 2× 后**逐像素差 0、最大差 0**），而三臂在同一批
# 机位上的 640 张渲染也已经有了 ⇒ **本段只按选择表拷文件，不渲染、不训练、不占 GPU**。
#
# 选择规则（`tools/pick_real_views.py`，规则原文写进 真实机位_选择.json 的「规则」字段）：
#   640 台 test 机位 = 160 个位置 × 4 个**固定世界朝向**（实测 {270°,0°,90°,270°}，
#   与位置无关，像一台四向机架）；按渲染索引把 640 台聚成 160 个位置点，每个位置取
#   **最朝向场景质心**的那一台，再把 160 个位置**按绕质心的方位角升序**排序。
#   ⇒ 每一帧的位置/朝向/图像全部来自数据集，只有**排序**是本脚本加的。
#
# 观感上要知道的两点（不粉饰）：① 朝向被量化到 4 个方向 ⇒ 环绕一圈里视线会分 4 次
# 各转 90°（与内向夹角最大 45°，这是**数据的固有分辨率**，不是我们造的误差）；
# ② 真实覆盖在方位角 11.9°→62.5° 之间有个缺口 ⇒ 第 7 帧有一次 5.4 m 的跳位。
if seg REAL; then
  echo; echo "[$(date +%F_%T)] ==== 段REAL：真实机位四栏帧（GT + 三臂）===="
  SEL="$RUN/真实机位_选择.json"; IDX="$RUN/真实机位_索引.txt"
  if [ ! -f "$IDX" ]; then
    "$PY" tools/pick_real_views.py --dataset "$DS/3dgs_MPC_100" --cameras-json "$CAM_JSON" \
        --out "$SEL" --index-out "$IDX" > "$LOG/real_pick.log" 2>&1 \
      || { echo "  REAL_PICK_FAIL"; tail -15 "$LOG/real_pick.log"; FAIL=1; }
  fi
  if [ ! -f "$IDX" ]; then
    echo "  [REAL] 无选择表，跳过（选帧失败）"
  else
    REAL_FRAMES=$(wc -l < "$IDX")
    echo "  [选帧] $REAL_FRAMES 帧；$(grep '^\[stat\] 半径' "$LOG/real_pick.log" 2>/dev/null)"
    for G in 1 2; do
      group_def "$G" || { FAIL=1; continue; }
      VROOT="$G_DIR/视频"
      for M in $METHODS; do
        for AK in A1 A2 A3; do
          eval "AD=\$ARM_DIR_$AK"; eval "AL=\$ARM_LAB_$AK"
          ABS_IT="$(arm_abs "$AK")"
          OURS="$G_DIR/阶段3_RRF重建/$AD/$M"
          case "$AK" in
            A1) REND="$A1_SRC/$M";;
            A2) if [ "$M" = MVDR ] || [ "$M" = CBF ]; then REND="$A2_SRC/$M"; else REND="$OURS"; fi;;
            A3) REND="$OURS";;
          esac
          # ⚠ 复用臂的 640 张**未必在源目录里**：A2 的 MVDR/CBF 在基线 run
          # （run_20260912_多频段标量分支）里**只有 48 张**（abs70000 那档才是 640），
          # 640 张是 RENDER 段在本树软链视图里补渲的 ⇒ 优先用本树臂目录。
          # 本段第一次跑就是踩了这个：A2 MVDR/CBF 只拷到 9/160 帧（REAL_FAIL）。
          has_full_renders "$OURS" "$ABS_IT" && REND="$OURS"
          SRC640="$REND/test/ours_${ABS_IT}"
          [ -d "$SRC640/renders" ] || { echo "  [skip] REAL $G $AK $M 无 $SRC640/renders"; FAIL=1; continue; }
          # 同一批帧：GT 与三臂都拷。GT 四方法各不相同（各方法的伪彩图），故逐 (组,方法) 一份。
          for KIND in renders gt; do
            if [ "$KIND" = gt ]; then
              # GT 只拷一次：四臂共用同一批真值，落在 A1 的迭代里
              [ "$AK" = A1 ] || continue
              DST="$VROOT/$M/漫游/$ARM_LAB_GT/real"
            else
              DST="$VROOT/$M/漫游/$AL/real"
            fi
            if [ "$(pngn "$DST")" -ge "$REAL_FRAMES" ]; then
              echo "  [RE] $G $AK $M $KIND 已有 $(pngn "$DST") 帧"
            else
              mkdir -p "$DST"
              while read -r k i nm; do
                cp -f "$SRC640/$KIND/${i}.png" "$DST/${k}.png" 2>/dev/null
              done < "$IDX"
              echo "  [RE] $G $AK $M $KIND $(pngn "$DST") 帧  ← $SRC640/$KIND"
            fi
            [ "$(pngn "$DST")" -ge "$REAL_FRAMES" ] \
              || { echo "  REAL_FAIL $G $AK $M $KIND 只拷到 $(pngn "$DST")/$REAL_FRAMES 帧"; FAIL=1; }
          done
        done
      done
    done
  fi
fi

# =============================================================================
# 段 COV：阶段1 覆盖度（阶段A 的交付件）
# =============================================================================
if seg COV; then
  echo; echo "[$(date +%F_%T)] ==== 段COV：MVSplat 覆盖度 ===="
  CD="$RUN/阶段1_MVSplat初始化点云"
  if [ -f "$CD/验证数据.json" ]; then
    echo "  已存在，跳过"
  else
    "$PY" tools/mvsplat_coverage.py --cloud "$CD/init.ply" --src "$VIS" --out "$CD" \
        > "$LOG/coverage.log" 2>&1 \
      || { echo "  COV_FAIL"; tail -15 "$LOG/coverage.log"; FAIL=1; }
    [ -f "$CD/验证数据.json" ] && echo "  [C] $(tr -d '\n ' < "$CD/验证数据.json")"
  fi
fi

echo
echo "==== [$(date +%F_%T)] ALL_TWO_BUDGET_THREE_ARM_DONE (FAIL=$FAIL) ===="
echo "  清单目录: $MANI（$(ls "$MANI" 2>/dev/null | wc -l) 个节点，供 pull_three_arm.sh 使用）"
exit "$FAIL"
