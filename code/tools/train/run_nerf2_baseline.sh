#!/usr/bin/env bash
# run_nerf2_baseline.sh —— NeRF² 第二基线驱动（2 臂 × 2 方法 = 4 个节点）
#
# 范式照 run_two_budget_three_arm.sh（段机制 / 幂等跳过 / 清单 / 收尾哨兵），
# 但**不复用它的 ARMS**：那边把 A1/A2/A3 写死，新图工具自带 ARMS5，
# 不 import 也不改动既有工具（方案 §3「不改」清单）。
#
# 两个臂只差输出参数化：
#     N1 = RGB 头（--mode rgb）      N2 = 标量头 + 固定 jet（--mode scalar）
# 网络 / 渲染器 / 优化器 / 损失 / 数据 / 划分 / 采样**全部相同** ⇒ Δ(N2 − N1) 干净。
#
# 分段（幂等，可中断续跑）：
#   P0      解包上游 RF-3DGS 源码，产出「框架判定」证据（**只出证据，不出结论**）
#   GATE    服务器侧验证闸（G2/G3/G4/G4b/G5b/G9/G7）；任一 FAIL 就地停机
#   DATA    按方法建 npz 缓存（N1/N2 共用；不需 GPU 计算）
#   TRAIN   4 次训练，串行
#   RENDER  4 次渲染 → test/ours_30000/{renders,gt}
#   FIG     拷 4 帧进 定性抽帧/（本机拉 32 张小图而不是 2560 张）
#   METRIC  4 次全量评测（metrics_fast --subset 0）→ results.json
#   SRC     每节点 来源.txt + 组级 来源_三臂复用.txt
#   POST    产物闸 G10/G11（渲染能被评测链读出、交付物可溯源）
#
# 用法（**先跟用户确认卡号**）：
#   CUDA_VISIBLE_DEVICES=3 bash code/tools/train/run_nerf2_baseline.sh
#   CUDA_VISIBLE_DEVICES=3 SEGMENTS="GATE DATA" bash code/tools/train/run_nerf2_baseline.sh
#
# 铁律：
#   * **绝不编辑正在运行的驱动** —— bash 按字节偏移顺序读脚本，改了前部会让后面的段
#     读到错位文本。要改先杀进程。
#   * **行尾必须是纯 LF**。CRLF 会静默打断所有 `\` 续行，`bash -n` 报的错不指向真原因。
#   * 绝不 `rm -rf` 任何目标目录。
#   * `--quiet` 会把 sys.stdout 换成 no-op ⇒ 关键读数一律走 stderr（日志里看得见）。

set -uo pipefail
ulimit -n 65535 2>/dev/null || true
FD_NOW=$(ulimit -n 2>/dev/null || echo 0)
[ "$FD_NOW" -ge 8192 ] 2>/dev/null \
  || echo "[警告] fd 上限只有 $FD_NOW（未提到 8192 以上）：读 3200 张 PNG 可能 EMFILE 假失败"
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "缺 CUDA_VISIBLE_DEVICES（必须只列一张卡）"; exit 2; }
case "$CUDA_VISIBLE_DEVICES" in *,*) echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 暴露多张卡。上游 renderer.py 有 7 处不带设备号的 .cuda()，多卡时它恒取物理 0 号卡（别人满载）。**必须恰好一张**。"; exit 2;; esac

# ---- 服务器路径 -----------------------------------------------------------
PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
DS=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum
SRC_TAR=/data/yuexf/3dgs/datasets/RF3DGS_src/r.tar.gz
A1_SRC=/data/yuexf/3dgs/results/run_rrf40k_sh
RUN="${RUN:-/data/yuexf/3dgs/results/run_20260916_NeRF2第二基线}"

ARMS="${ARMS:-N1 N2}"
METHODS="${METHODS:-MVDR CBF}"
SEGMENTS="${SEGMENTS:-P0 GATE DATA TRAIN RENDER FIG METRIC SRC POST}"
# 定性抽帧的帧号：沿用既有三臂对照的约定，**不许改**，否则跨臂同帧对比失效
FRAMES="${FRAMES:-0 160 320 480}"

# 上游参照配置的超参（UPSTREAM.md §2），逐项固定，不暴露成旋钮
ITER=30000
BATCH=8192; PH=64; PW=128
LR=8e-4; WD=5e-5; TMAX=10000; ETAMIN=1e-6
NSAMP=64; NEAR=0.10
LUTN=256
# 末端映射的**窗口下限**（span/α，即 σ 活性区间在 v 上的宽度）。取值依据与实测见
# `baselines/nerf2/heads.py` 的 calibrate() 长注释，以及 `tools/tune_nerf2_head_window.sh`
# 的扫描结果。**它不是可忽略的常数**，环境变量可覆盖以便复跑扫描。
# 默认 1.0 → **0.04**（2026-09-17 改）。旧值的依据「太小会饱和成全白」是对的，但
# **它把因果搞反了**：实测窗口 1.0 下工作点落到 σ 的平坦尾部（σ'(β) = 0.0061），
# 标量头臂 6000 步只有 **3.174 dB —— 低于单色常数图地板 11.881 dB**，输出是与像素
# 无关的常数（u 恒为 0.006109）。窗口不是拟合之后的微调项，它决定**有没有梯度**。
# 见 `说明.md` §8.7（该节原结论「窗口是二阶的」已被自己的后续实测证伪并整段重写）。
HVW="${HVW:-0.04}"
# 末端映射的形式（2026-09-17 新增，原先靠 train_nerf2.py 的默认 sigmoid）。
# sigmoid 与 clamp 是**同一种病**：带外梯度恒为 0，某通道一旦 100% 落在带外就
# **永久冻结**，没有任何梯度能把它拉回来。实测 RGB 头臂逐通道截断占比
# 14.1% / 88.0% / 100.0%，B 通道整个活在带外 ⇒ 只有 10.6 dB。
# leaky 在带外保留 1e-2 的斜率（逃逸通道，不是工作区），带内 z−c ≡ 0 故与 clamp 等价
# ⇒ N1 从 10.642 抬到 24.495 dB。见 `说明.md` §8.9。
# **两臂必须用同一个末端映射**，否则 Δ(N2−N1) 会混入第二个自变量。
HEND="${HEND:-leaky}"

LOG="$RUN/日志"
MANI="$RUN/清单"
BASE="$RUN/NeRF2基线"
JUDGE="$RUN/基线框架判定"

mkdir -p "$LOG" "$MANI" "$BASE" "$JUDGE"
cd "$CODE" || exit 9
echo "==== [$(date +%F_%T)] NeRF² 第二基线 卡=$CUDA_VISIBLE_DEVICES arms='$ARMS' methods='$METHODS' ===="
echo "     SEGMENTS='$SEGMENTS'  RUN=$RUN"
FAIL=0
seg() { case " $SEGMENTS " in *" $1 "*) return 0;; *) return 1;; esac; }

arm_dir()   { case "$1" in N1) echo "N1_NeRF2_RGB头";; N2) echo "N2_NeRF2_标量头jet";; *) echo "";; esac; }
arm_mode()  { case "$1" in N1) echo rgb;; N2) echo scalar;; *) echo "";; esac; }
# 图内 ASCII 标签（中文标签在某些绘图后端会缺字形，方案沿用既有纪律：图内一律 ASCII）
arm_label() { case "$1" in N1) echo "N1_NeRF2_RGB";; N2) echo "N2_NeRF2_scalar";; *) echo "";; esac; }
node_dir()  { echo "$BASE/$(arm_dir "$1")/$2"; }
data_dir()  { echo "$DS/3dgs_${1}_100"; }
cache_p()   { echo "$BASE/cache_$1.npz"; }
ckpt_p()    { echo "$(node_dir "$1" "$2")/nerf2_iter_${ITER}.pth"; }
pngn()      { find "$1" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l; }
has_full_renders() {  # $1=节点目录；640 张 renders + 640 张 gt 才算全量
  local od="$1/test/ours_$ITER"
  [ "$(pngn "$od/renders")" -ge 640 ] && [ "$(pngn "$od/gt")" -ge 640 ]
}

# ===========================================================================
# 段 P0：解包上游 RF-3DGS 源码 —— 硬前置，未完成不许开 GPU 训练
# ===========================================================================
# 这一段的产物是**证据**，不是结论。「像素 ↔ (方位角, 天顶角)」的映射靠读生成脚本
# 定案（方案 F7），而哪一段代码是生成脚本、里面哪个常数是 Tx 坐标，需要人/模型看过
# 才能判 —— 脚本自己不敢下这个结论，也不该假装下了。
if seg P0; then
  echo; echo "[$(date +%F_%T)] ==== 段P0：上游框架判定 ===="
  if [ -f "$JUDGE/结论.txt" ]; then
    echo "  已有 结论.txt，跳过"
  elif [ ! -f "$SRC_TAR" ]; then
    echo "  P0_FAIL 缺 $SRC_TAR"
    echo "  若该包不在预期位置，先 find /data/yuexf/3dgs -name 'r.tar.gz' 定位" >&2
    FAIL=1
  else
    EX="$JUDGE/_解包"
    mkdir -p "$EX"
    tar -tzf "$SRC_TAR" > "$JUDGE/00_包内清单.txt" 2>"$LOG/p0_tar.log" \
      || { echo "  P0_FAIL tar -t 失败"; tail -5 "$LOG/p0_tar.log"; FAIL=1; }
    if [ -s "$JUDGE/00_包内清单.txt" ]; then
      echo "  包内条目 $(wc -l < "$JUDGE/00_包内清单.txt") 个"
      # 只解 .py/.m/.yml/.yaml/.json/.md/.txt/.cfg —— 包可能有几十 GB 的图，不解
      tar -xzf "$SRC_TAR" -C "$EX" \
        --wildcards '*.py' '*.m' '*.yml' '*.yaml' '*.json' '*.md' '*.txt' '*.cfg' '*.toml' \
        2>>"$LOG/p0_tar.log" || echo "  [note] 部分解包（通配未命中某些成员，非致命）"

      # 把「像生成侧」的文件挑出来。**是宽网**：宁可多列，让人去看，不要漏。
      {
        echo "# P0 上游摘录 —— 由 run_nerf2_baseline.sh 段P0 自动生成"
        echo
        echo "来源：$SRC_TAR"
        echo "解包到：$EX（只解了文本类文件；图像/权重未解）"
        echo
        echo "## 1. 包内清单（前 400 行）"
        echo '```'
        head -400 "$JUDGE/00_包内清单.txt"
        echo '```'
        echo
        echo "## 2. 候选生成脚本（按关键词打分，只看命中行）"
        echo
        echo "关键词：spectrum / azimuth / theta / 方位 / 天顶 / jet / imwrite / png / dBm / dB / tx_pos"
      } > "$JUDGE/P0_上游摘录.md"

      while IFS= read -r f; do
        HIT=$(grep -nEi 'spectrum|azimuth|theta|jet|imwrite|colormap|dBm?|tx_pos|方位|天顶' "$f" 2>/dev/null | head -40)
        [ -z "$HIT" ] && continue
        {
          echo
          echo "### \`${f#$EX/}\`  （$(wc -l < "$f") 行，命中 $(printf '%s\n' "$HIT" | wc -l) 行）"
          echo '```'
          printf '%s\n' "$HIT"
          echo '```'
        } >> "$JUDGE/P0_上游摘录.md"
      done < <(find "$EX" -name '*.py' -o -name '*.m' 2>/dev/null | head -80)

      echo "  摘录写入 $JUDGE/P0_上游摘录.md（$(wc -l < "$JUDGE/P0_上游摘录.md") 行）"
      echo "  ⚠ 结论.txt **不自动写** —— 映射定案要看摘录才能下，见方案 P0 说明"
    fi
  fi
fi

# ===========================================================================
# 段 GATE：服务器侧验证闸（不需要 GPU）
# ===========================================================================
if seg GATE; then
  echo; echo "[$(date +%F_%T)] ==== 段GATE：服务器侧验证闸 ===="
  # 闸② 的参考臂**必须逐方法**（$A1_SRC/$M，只读）。原先写死 $A1_SRC/MPC 并注明
  # 「G2 只比字节，与方法无关」—— **那是错的**：图像内容逐方法不同（实测 MVDR 与 CBF
  # 的 images/00006.png md5 不同），拿 MPC 的 gt 去比 MVDR 的数据必然不等，于是 G2
  # 会对**正确**的数据报警。run_rrf40k_sh 里 CBF/Delay/MPC/MVDR 四个都有，逐方法取即可。
  GFAIL=0
  for M in $METHODS; do
    D="$(data_dir "$M")"
    [ -d "$D" ] || { echo "  [skip] GATE $M：缺 $D"; continue; }
    REF="$A1_SRC/$M"
    [ -d "$REF" ] || { echo "  [skip] GATE $M：缺参考臂 $REF"; continue; }
    echo "  -- $M ($D)"
    "$PY" tools/verify_nerf2_gates.py --stage server --data_dir "$D" \
        --ref_arm_dir "$REF" --out "$JUDGE/闸报告_$M.json" \
        > "$LOG/gate_$M.log" 2>&1
    rc=$?
    grep -E '^\s+\[(PASS|FAIL|SKIP)\]' "$LOG/gate_$M.log" | sed 's/^/    /'
    grep -E '^== stage=' "$LOG/gate_$M.log" | sed 's/^/    /'
    [ $rc -ne 0 ] && { echo "  GATE_FAIL $M"; GFAIL=1; }
  done
  [ $GFAIL -ne 0 ] && { echo "  ⇒ 闸未过，**就地停机**，不进 DATA/TRAIN"; exit 1; }

  # ---- G5 冒烟 + G6 单视角过拟合（**需要 GPU**）-----------------------------
  # 为什么接在这里：这两道闸方案 §7/§9 里是训练的**前置**，但 `verify_nerf2_gates.py`
  # 根本没实现它们，GATE 段也只跑 `--stage server` ⇒ 它们只是「方案里写了、驱动没接」，
  # 于是被**静默跳过**。后果是 2026-09-16 那次：`heads.py` 的窗口取错值让输出饱和全白，
  # **loss 照常缓降（L1 0.47→0.20）看起来完全正常**，只有 G6 能一眼拦住它
  # （全白图绝无可能过「优于单色常数图」这条判据）。所以它必须是 TRAIN 的硬前置。
  #
  # 段 GATE 原先标称「不需要 GPU」—— 加上这一段后它**需要 GPU**（约 25 min）。
  # 不另开段是为了让它无法被单独跳过：只要跑 TRAIN，就一定先跑过 GATE。
  echo "  -- G5 冒烟 + G6 单视角过拟合（GPU）"
  # HEND 必须一起传：只传 HVW 的话，命令行若覆盖 HEND，闸验的（脚本默认 leaky）
  # 就会与 TRAIN 段实际跑的（$HEND）不是同一个配置 ⇒ **闸失去意义**。
  HVW="$HVW" HEND="$HEND" bash tools/run_nerf2_g5g6.sh > "$LOG/g5g6.out" 2>&1
  G56=$?
  grep -E '^\s*\[(PASS|FAIL)\]|G5G6_ALL_PASS|G5G6_FAIL|判据' "$LOG/g5g6.out" | sed 's/^/    /'
  [ $G56 -ne 0 ] && { echo "  GATE_FAIL G5/G6 ⇒ **就地停机**，不进 DATA/TRAIN"; exit 1; }
fi

# ===========================================================================
# 段 DATA：按方法建缓存（N1/N2 共用，不需要 GPU 计算）
# ===========================================================================
if seg DATA; then
  echo; echo "[$(date +%F_%T)] ==== 段DATA：数据缓存 ===="
  for M in $METHODS; do
    D="$(data_dir "$M")"; C="$(cache_p "$M")"
    if [ -s "$C" ]; then echo "  [skip] $M 缓存已存在 $(du -h "$C" | cut -f1)"; continue; fi
    echo "  -- 建缓存 $M"
    "$PY" train_nerf2.py --data_dir "$D" --method "$M" --out_dir "$BASE/_dataonly" \
        --cache "$C" --data_only --quiet > "$LOG/data_$M.log" 2>&1 \
      || { echo "  DATA_FAIL $M"; tail -20 "$LOG/data_$M.log"; FAIL=1; }
    [ -s "$C" ] && echo "     → $(du -h "$C" | cut -f1)" || { echo "  DATA_FAIL $M 缓存未生成"; FAIL=1; }
  done
fi

# ===========================================================================
# 段 TRAIN：4 次训练，串行
# ===========================================================================
if seg TRAIN; then
  echo; echo "[$(date +%F_%T)] ==== 段TRAIN：4 个节点 ===="
  for A in $ARMS; do
    MD="$(arm_mode "$A")"
    for M in $METHODS; do
      ND="$(node_dir "$A" "$M")"; CK="$(ckpt_p "$A" "$M")"; C="$(cache_p "$M")"
      mkdir -p "$ND"
      if [ -s "$CK" ]; then echo "  [skip] $A $M 存档已存在"; continue; fi
      [ -s "$C" ] || { echo "  [skip] $A $M 缺缓存 $C（先跑 DATA 段）"; FAIL=1; continue; }
      echo "  -- 训练 $A($MD) $M  $(date +%F_%T)"
      # **不要加 --rebuild_cache**：那会让每个节点都重读 3200 张 PNG（~10 min × 4），
      # DATA 段就白做了；更要紧的是它会掩盖「缓存 near/far 与本次配置不符」这条报错
      # （train_nerf2.py:174 会直接 raise，那正是我们要它炸的场景）。
      "$PY" train_nerf2.py --data_dir "$(data_dir "$M")" --method "$M" \
          --out_dir "$ND" --cache "$C" \
          --mode "$MD" --lut_n "$LUTN" \
          --iterations "$ITER" --batch_size "$BATCH" --patch_h "$PH" --patch_w "$PW" \
          --lr "$LR" --weight_decay "$WD" --t_max "$TMAX" --eta_min "$ETAMIN" \
          --n_samples "$NSAMP" --near "$NEAR" --far 0 \
          --loss l1ssim --target rgb --save_iterations "$ITER" --quiet \
          --head_end "$HEND" --head_v_window "$HVW" \
          > "$LOG/train_${A}_${M}.log" 2>&1
      rc=$?
      grep -E '^\[nerf2\]' "$LOG/train_${A}_${M}.log" | tail -8 | sed 's/^/    /'
      if [ $rc -ne 0 ] || [ ! -s "$CK" ]; then
        echo "  TRAIN_FAIL $A $M (rc=$rc)"; tail -25 "$LOG/train_${A}_${M}.log"; FAIL=1
      fi
    done
  done
fi

# ===========================================================================
# 段 RENDER：4 次渲染（布局与 render.py 逐字同构）
# ===========================================================================
if seg RENDER; then
  echo; echo "[$(date +%F_%T)] ==== 段RENDER ===="
  for A in $ARMS; do
    for M in $METHODS; do
      ND="$(node_dir "$A" "$M")"; CK="$(ckpt_p "$A" "$M")"; C="$(cache_p "$M")"
      if has_full_renders "$ND"; then echo "  [skip] $A $M 渲染已全量"; continue; fi
      [ -s "$CK" ] || { echo "  [skip] $A $M 缺存档 $CK"; FAIL=1; continue; }
      echo "  -- 渲染 $A $M"
      "$PY" render_nerf2.py --checkpoint "$CK" --out_dir "$ND" \
          --data_dir "$(data_dir "$M")" --cache "$C" --which test \
          > "$LOG/render_${A}_${M}.log" 2>&1
      rc=$?
      grep -E '^\[nerf2-render\]' "$LOG/render_${A}_${M}.log" | tail -3 | sed 's/^/    /'
      has_full_renders "$ND" || { echo "  RENDER_FAIL $A $M (rc=$rc)"; tail -20 "$LOG/render_${A}_${M}.log"; FAIL=1; }
    done
  done
fi

# ===========================================================================
# 段 FIG：把定性图要用的 4 帧从 test/ 拷进本树 定性抽帧/
# ===========================================================================
# 为什么要拷而不是让本机直接从 test/ours_30000/renders/ 取：那样 pull 就得拉
# 640 张 × 4 节点 = 2560 张（~200 MB）。拷 4 帧再拉 = 32 张（~3 MB）。
# 帧号沿用既有三臂对照的约定（0 160 320 480，%05d），保证跨臂同帧可比。
if seg FIG; then
  echo; echo "[$(date +%F_%T)] ==== 段FIG：定性抽帧 ===="
  for A in $ARMS; do
    for M in $METHODS; do
      ND="$(node_dir "$A" "$M")"; SD="$ND/test/ours_$ITER"
      has_full_renders "$ND" || { echo "  [skip] $A $M 渲染不全"; continue; }
      mkdir -p "$ND/定性抽帧/renders" "$ND/定性抽帧/gt"
      miss=0
      for i in $FRAMES; do
        k=$(printf '%05d' "$i")
        for sub in renders gt; do
          if [ -f "$SD/$sub/$k.png" ]; then
            cp -f "$SD/$sub/$k.png" "$ND/定性抽帧/$sub/$k.png"
          else
            echo "  [warn] $A $M 缺 $sub/$k.png"; miss=1
          fi
        done
      done
      [ $miss -eq 0 ] && echo "  [帧] $A $M → $(ls "$ND/定性抽帧/renders" | wc -l) 张"
    done
  done
fi

# ===========================================================================
# 段 METRIC：4 次全量评测（与 metrics.py 逐字同实现，subset 0 = 全部 640 张）
# ===========================================================================
if seg METRIC; then
  echo; echo "[$(date +%F_%T)] ==== 段METRIC ===="
  for A in $ARMS; do
    for M in $METHODS; do
      ND="$(node_dir "$A" "$M")"
      if [ -s "$ND/results.json" ]; then echo "  [skip] $A $M results.json 已存在"; continue; fi
      has_full_renders "$ND" || { echo "  [skip] $A $M 渲染不全"; FAIL=1; continue; }
      echo "  -- 评测 $A $M"
      "$PY" tools/metrics_fast.py -m "$ND" --iterations "$ITER" --subset 0 \
          --out "$ND/results.json" > "$LOG/metric_${A}_${M}.log" 2>&1
      rc=$?
      grep -E 'ours_' "$LOG/metric_${A}_${M}.log" | sed 's/^/    /'
      [ -s "$ND/results.json" ] || { echo "  METRIC_FAIL $A $M (rc=$rc)"; tail -20 "$LOG/metric_${A}_${M}.log"; FAIL=1; }
    done
  done
fi

# ===========================================================================
# 段 SRC：来源与清单（可溯源）
# ===========================================================================
if seg SRC; then
  echo; echo "[$(date +%F_%T)] ==== 段SRC：来源 ===="
  for A in $ARMS; do
    for M in $METHODS; do
      ND="$(node_dir "$A" "$M")"
      [ -d "$ND" ] || { echo "  [skip] $A $M 无目录"; continue; }
      RS="$ND/来源.txt"
      {
        echo "臂：$A（$(arm_dir "$A")，--mode $(arm_mode "$A")）"
        echo "方法：$M"
        echo "数据：$(data_dir "$M")"
        echo "缓存：$(cache_p "$M")"
        echo "存档：$(ckpt_p "$A" "$M")"
        echo "运行目录：$RUN"
        echo "驱动：code/tools/train/run_nerf2_baseline.sh  日期：$(date +%F_%T)"
        echo "上游：XPengZhao/NeRF2（MIT），model.py/renderer.py 逐字 vendor，"
        echo "      sha256 见 code/baselines/nerf2/UPSTREAM.md；训练时已校验"
        echo "评测：tools/metrics_fast.py --subset 0（与 metrics.py 逐字同实现；"
        echo "      PSNR 峰值1.0 / SSIM window11 / LPIPS net_type=vgg）"
        echo "划分：train_index.txt(2560) / test_index.txt(640)；--eval 对频谱集无效"
      } > "$RS"
      echo "  [来源] $A $M"
    done
  done

  SRC3="$RUN/来源_三臂复用.txt"
  {
    echo "本 run 读到的既有三臂读数来源（**只读，未改动**）"
    echo
    echo "A1 原版 RF-3DGS : $A1_SRC   (run_rrf40k_sh，作者发布包口径)"
    echo "A2 原版物理+新色头 / A3 新物理+新色头：见 run_20260912_两组预算_三臂对比"
    echo
    echo "三臂的 run 目录与 md5 由 make_four_arm_bars.py 在本地现算并写入图注，"
    echo "本驱动不复制、不修改任何既有 run。"
  } > "$SRC3"
  echo "  [来源] $SRC3"
fi

# ===========================================================================
# 段 POST：产物可被评测链读出（G10）与可溯源（G11）。放最后，因为 G11 要读
# SRC 段写的 来源.txt，G10 要读 RENDER 段写的 PNG。
# ===========================================================================
if seg POST; then
  echo; echo "[$(date +%F_%T)] ==== 段POST：产物闸 ===="
  for A in $ARMS; do
    for M in $METHODS; do
      ND="$(node_dir "$A" "$M")"
      has_full_renders "$ND" || { echo "  [skip] $A $M 渲染不全"; continue; }
      "$PY" tools/verify_nerf2_gates.py --stage post --node_dir "$ND" \
          --iteration "$ITER" --out "$JUDGE/产物闸_${A}_${M}.json" \
          > "$LOG/post_${A}_${M}.log" 2>&1
      rc=$?
      grep -E '^\s+\[(PASS|FAIL|SKIP)\]' "$LOG/post_${A}_${M}.log" | sed 's/^/    /'
      [ $rc -ne 0 ] && { echo "  POST_FAIL $A $M"; grep -E '^   FAIL' "$LOG/post_${A}_${M}.log"; FAIL=1; }
    done
  done
fi

echo
echo "==== [$(date +%F_%T)] ALL_NERF2_BASELINE_DONE (FAIL=$FAIL) ===="
echo "  节点：$BASE"
echo "  清单：$MANI（$(ls "$MANI" 2>/dev/null | wc -l) 个）"
echo "  下一步（本机）：bash code/tools/archive/pull_nerf2.sh"
exit "$FAIL"
