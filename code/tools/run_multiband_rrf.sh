#!/bin/bash
# =============================================================================
# run_multiband_rrf.sh —— 形态 A×：多频段标量分支（乘法调制）
# =============================================================================
# 承接形态 A'（标量头 + 固定 jet 查表，先混合再查表）留下的**唯一明确缺口**：
#   CBF 渲染通道 std 37.05 明显低于 GT 51.08（过平滑），MVDR 几乎持平（52.96 vs 54.12）。
#   文档把「CBF 的瓶颈是标量头容量」写为**推断**；本组实验就是去兑现或**证伪**它。
#
# 形态（详见 splat/color_scalar.py 模块头部，含不可塌缩的证明）：
#   s = head(...) → (P,2)
#   mixed = render(override_color=cat[s_0, s_1, s_1])   # 借光栅化逐通道线性，一次出两个场
#   u = ū_0 · (1 + λ·(ū_1 − init_u))                    # 标量域非线性合成，λ∈(0,1)
#   image = apply_jet(u, lut)                           # 只查一次表 ⇒ 输出仍在 jet 上
#   **λ=0 时精确退化成 A'** ⇒ A× 严格包含 A'，对照是**嵌套**的。
#
# 为什么不是"多开几个头再加起来"：渲染器对颜色**严格线性且逐通道独立**
#   （forward.cu:355-373，α/T 在通道循环外算）⇒ 任何"各带混合后线性求和再查表"
#   都等价于把输出层加宽。**只有非线性合成能逃出该塌缩。**
#
# 为什么只做 2 个带：统一输入 + 对称初值下（W_k 全 0、b_k 全 logit(init_u)）s_1≡s_2，
#   且初值时 ∂u/∂ū_k 对各带相同 ⇒ Adam 逐参数确定性 ⇒ W_1≡W_2 **永远**。
#   第 3 个带是纯规范冗余。要真做 3 个必须让各带输入不同（破坏与 A' 的嵌套性，本步不做）。
#
# 方法范围（按"展开域"判据，论文：CIR 在时延域展开、RRF 在 AoA 域展开）：
#   CBF / MVDR = AoA 域角功率谱 ⇒ **本任务内**，且作者基线可比 ⇒ 本次两个方法。
#   Delay（时延域）、MPC（疑为 AoD 编码、实测 2 维色流形非 jet）⇒ **不在本任务**，排除。
#
# 协议：与 A' **逐字相同**的 A* 协议（exp 8e-3→8e-5 / max_steps 40000 / PE 全开），
#   唯一新增 = --n_bands 2 --band_mix mult。故与 A'、V14、SH@A* 都是单变量对比。
#
# 产物：$TOP/<M>/ 下 test/ours_<i>（阶梯档 48 抽帧、末档全量 640）、curve.json、results.json。
#   各段存在即跳过（可续跑）—— 故 **TOP 必须是新目录**，否则会静默复用旧臂产物。
#
# 用法（服务器 code/ 下，占一卡）：
#   CUDA_VISIBLE_DEVICES=2 METHODS="MVDR CBF" nohup bash tools/run_multiband_rrf.sh \
#       > /data/yuexf/3dgs/results/run_20260912_多频段标量分支/日志/驱动.log 2>&1 &
# 冒烟（单方法、极少步数、独立 TOP）：
#   CUDA_VISIBLE_DEVICES=2 METHODS=MVDR TOP=/tmp/smoke_mb \
#     LADDER=30200 FINAL=30400 bash tools/run_multiband_rrf.sh
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "缺 CUDA_VISIBLE_DEVICES"; exit 2; }

METHODS="${METHODS:-MVDR CBF}"

PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
DS=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum
SEED=/data/yuexf/3dgs/results/orig_seed_chkpnt30000.pth
TOP="${TOP:-/data/yuexf/3dgs/results/run_20260912_多频段标量分支}"
LOG="$TOP/日志"
LADDER="${LADDER:-32000 35000 38000 40000 45000 50000 55000 60000 65000}"
FINAL="${FINAL:-70000}"
SUBSET_CURVE=48
A_INIT=8e-3
A_FINAL=8e-5

# —— 形态 A× 开关（本驱动存在的唯一理由）——
NBANDS=2
BANDMIX=mult

[ -d "$DS" ] || { echo "数据缺失: $DS"; exit 1; }
[ -f "$SEED" ] || { echo "seed 缺失: $SEED"; exit 1; }
cd "$CODE" || exit 9
mkdir -p "$LOG"
echo "==== [$(date +%F_%T)] 形态A× 多频段标量分支  卡=$CUDA_VISIBLE_DEVICES  methods='$METHODS'  TOP=$TOP ===="
echo "     n_bands=$NBANDS band_mix=$BANDMIX  ladder='$LADDER' final=$FINAL"

FAIL=0
for M in $METHODS; do
  DATA="$DS/3dgs_${M}_100"
  MD="$TOP/$M"
  mkdir -p "$MD"
  [ -d "$DATA" ] || { echo "  [$M] 数据缺失，跳过: $DATA"; FAIL=1; continue; }
  echo "[$(date +%F_%T)] ===== $M =====  -> $MD"

  if [ ! -f "$MD/point_cloud/iteration_${FINAL}/point_cloud.ply" ]; then
    echo "[$(date +%F_%T)]  训练 -> $FINAL（RRF 本地 $((FINAL-30000)) 步，A* 调度）"
    "$PY" train_rrf_scalar.py -s "$DATA" -m "$MD" \
        --start_checkpoint "$SEED" --iterations "$FINAL" --save_iterations $LADDER \
        --densify_until_iter 0 --data_device cpu --quiet \
        --embed_dim 32 --mlp_hidden 128 --mlp_depth 2 \
        --lr_schedule exp --mlp_lr_init "$A_INIT" --mlp_lr_final "$A_FINAL" \
        --lr_max_steps 40000 --pe_xyz_L 5 --pe_dir_L 3 --use_dist \
        --n_bands "$NBANDS" --band_mix "$BANDMIX" \
        > "$LOG/train_${M}.txt" 2>&1 \
      || { echo "  TRAIN_FAIL_$M"; tail -25 "$LOG/train_${M}.txt"; FAIL=1; continue; }
    # λ 轨迹：初值 0.5；若几乎没动 ⇒ 梯度只来自轮廓像素（内部 ∂u/∂λ = ū_0·d_1 = 0），
    # 属"部分生效"，必须如实写进文档。这里只抽取存档点的读数。
    # 只取存档点那一行的读数（启动行里的"λ 初值"会被误匹配，故先锚定 sidecar 关键字）
    echo "  [λ] $(grep -aho 'sidecar.*' "$LOG/train_${M}.txt" | grep -ao 'λ=[0-9.]*' | tr '\n' ' ')"
  else
    echo "  final 档已存在，跳过训练"
  fi

  for i in $LADDER $FINAL; do
    sub=$SUBSET_CURVE; [ "$i" = "$FINAL" ] && sub=0
    [ -d "$MD/test/ours_$i/renders" ] || "$PY" render_rrf_scalar.py -s "$DATA" -m "$MD" \
        --iteration "$i" --eval --skip_train --subset "$sub" --quiet \
        > "$LOG/render_${M}_$i.txt" 2>&1 \
      || { echo "  RENDER_FAIL_${M}_$i"; tail -12 "$LOG/render_${M}_$i.txt"; FAIL=1; break; }
    echo "  [R] $M ours_$i 已渲染 (subset=$sub)"
  done

  [ -f "$MD/curve.json" ] || "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
      --iterations $LADDER "$FINAL" --subset "$SUBSET_CURVE" --out "$MD/curve.json" \
      > "$LOG/curve_${M}.txt" 2>&1
  [ -f "$MD/results.json" ] || "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
      --iterations "$FINAL" --subset 0 --out "$MD/results.json" \
      > "$LOG/full_${M}.txt" 2>&1
  echo "  [640] $(cat "$MD/results.json" 2>/dev/null)"
  echo "[$(date +%F_%T)]  $M 完成：$MD"
done

echo "==== [$(date +%F_%T)] ALL_MULTIBAND_DONE (FAIL=$FAIL) ===="
