#!/bin/bash
# =============================================================================
# load_config.sh —— 从仓库根 config.yaml 装载配置到 shell 环境（run/evaluate 共用）
# -----------------------------------------------------------------------------
# 用法：source "$(dirname "${BASH_SOURCE[0]}")/load_config.sh"
#       然后  load_config full      （profile 默认 full；可选 smoke）
#
# 装载后 shell 内可用（均 export）：
#   路径     P_ROOT P_CODE P_TOOLS P_VISUAL P_SPECTRUM P_ORIG_VIS P_ORIG_RRF
#            P_CKPT P_CLOUDS P_OUT PY
#   阶段A    RF_NUM_CLUSTERS RF_VIEWS_PER_CLUSTER RF_GAZES_PER_CLUSTER
#            RF_MAX_GAUSSIANS RF_VOXEL_SIZE RF_OPACITY_THRESHOLD RF_MAX_DEPTH
#            RF_CLUSTERS_LIMIT RF_BACKGROUND RF_NEAR RF_FAR RF_SOURCE_PATH
#            RF_TARGET_FRAMES_PER_CLUSTER
#   阶段B/C  ITR1(短训绝对末迭代) STAGE1_SAVE(阶段B存档迭代档，空格分隔字符串，缺省=ITR1)
#            ITR2(10000RRF 对齐档绝对末迭代) TOTAL2(40000总 档绝对末迭代，可选，缺省空)
#            RES(分辨率) DENSIFY(冻结几何=0)  METHODS("MPC CBF Delay MVDR" 空格分隔)
#   其它     CFG(配置文件) PROFILE EVAL(true/false)
# 注：config.yaml 路径为 WSL 视图；本文件需在 WSL bash 下运行。
# =============================================================================

load_config() {
  local profile="${1:-full}"
  local tools_dir root_dir cfg cfg_py out
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cfg="$root_dir/config.yaml"
  [ -f "$cfg" ] || { echo "[load_config] 找不到 $cfg" >&2; return 1; }

  # 引导解释器：优先用 config 里声明的 mvsplat python（必有 pyyaml），否则退回 python3
  cfg_py="$(sed -n 's/^[[:space:]]*python:[[:space:]]*//p' "$cfg" | head -1 \
            | sed 's/[[:space:]]*#.*$//' | sed 's/[[:space:]]*$//')"
  [ -n "$cfg_py" ] && [ -x "$cfg_py" ] || cfg_py="python3"

  # 内嵌 python：读 config.yaml + 选择 profile，输出 export 行（当前 shell eval）
  out="$("$cfg_py" - "$cfg" "$profile" <<'PYEOF'
import sys
import shlex
import yaml

cfg_path, profile = sys.argv[1], sys.argv[2]
with open(cfg_path, encoding="utf-8") as f:
    d = yaml.safe_load(f) or {}
profiles = d.get("profiles") or {}
if profile not in profiles:
    sys.exit("profile 不存在: %s（可选 %s）" % (profile, ", ".join(sorted(profiles))))

paths = d.get("paths") or {}
prof = profiles.get(profile) or {}
stage_b = d.get("stage_b") or {}
stage_c = d.get("stage_c") or {}
ev = d.get("eval") or {}

lines = []
exported = set()

def sh(v):
    return shlex.quote(str(v))

def emit(name, value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return
    lines.append("export %s=%s" % (name, sh(value)))
    exported.add(name)

# ---- 路径（paths.* → P_* / PY）----
PMAP = {
    "root": "P_ROOT", "code_root": "P_CODE",
    "code_tools": "P_TOOLS", "python": "PY",
    "data_visual": "P_VISUAL", "data_spectrum": "P_SPECTRUM",
    "data_orig_visual": "P_ORIG_VIS", "data_orig_rrf": "P_ORIG_RRF",
    "ckpt_re10k": "P_CKPT", "clouds": "P_CLOUDS", "results": "P_OUT",
}
for k, v in paths.items():
    if k in PMAP:
        emit(PMAP[k], v)
if "PY" not in exported:
    emit("PY", "python3")

# ---- 全局 stage 设置 ----
if stage_b.get("resolution") is not None:
    emit("RES", stage_b["resolution"])
if stage_c.get("densify_until_iter") is not None:
    emit("DENSIFY", stage_c["densify_until_iter"])
emit("EVAL", "true" if ev.get("enabled", True) else "false")

# ---- 分组策略超参：profile 键 → RF_*/ITR env（键名对应 init_cloud.py 读取的 env）----
RFMAP = {
    "num_clusters": "RF_NUM_CLUSTERS",
    "views_per_cluster": "RF_VIEWS_PER_CLUSTER",
    "gazes_per_cluster": "RF_GAZES_PER_CLUSTER",
    "max_gaussians": "RF_MAX_GAUSSIANS",
    "voxel_size": "RF_VOXEL_SIZE",
    "opacity_threshold": "RF_OPACITY_THRESHOLD",
    "max_depth": "RF_MAX_DEPTH",
    "clusters_limit": "RF_CLUSTERS_LIMIT",
    "background": "RF_BACKGROUND",
    "near": "RF_NEAR",
    "far": "RF_FAR",
    "target_frames_per_cluster": "RF_TARGET_FRAMES_PER_CLUSTER",
    "source_path": "RF_SOURCE_PATH",
}
for k, v in prof.items():
    if k == "methods":
        emit("METHODS", " ".join(v))
    elif k == "stage1_iters":
        emit("ITR1", v)
    elif k == "stage1_save_iters":
        emit("STAGE1_SAVE", v)
    elif k == "stage2_iters":
        emit("ITR2", v)
    elif k == "stage2_total_iters":
        emit("TOTAL2", v)
    elif k in RFMAP:
        emit(RFMAP[k], v)
# 缺省：STAGE1_SAVE 未声明时 = ITR1（单档）；TOTAL2 未声明时不 export（=无 40000总 档）
if "STAGE1_SAVE" not in exported:
    it1 = prof.get("stage1_iters")
    if it1 is not None:
        emit("STAGE1_SAVE", it1)
# RF_SOURCE_PATH 未在 profile 声明时缺省 = data_visual（阶段一视觉数据）
if "RF_SOURCE_PATH" not in exported and paths.get("data_visual"):
    emit("RF_SOURCE_PATH", paths["data_visual"])

print("\n".join(lines))
PYEOF
)" || {
    echo "[load_config] 解析 config.yaml 失败 (profile=$profile)" >&2
    return 1
  }
  # 清空本函数管理的历史 export（同一 shell 连续 load_config 不同 profile 时，
  # 前一个 profile 独有且本 profile 不重设的键（如 smoke 无 TOTAL2）不会残留）
  unset P_ROOT P_CODE P_TOOLS P_VISUAL P_SPECTRUM P_ORIG_VIS P_ORIG_RRF \
        P_CKPT P_CLOUDS P_OUT PY RES DENSIFY EVAL METHODS \
        ITR1 STAGE1_SAVE ITR2 TOTAL2 \
        RF_NUM_CLUSTERS RF_VIEWS_PER_CLUSTER RF_GAZES_PER_CLUSTER \
        RF_MAX_GAUSSIANS RF_VOXEL_SIZE RF_OPACITY_THRESHOLD RF_MAX_DEPTH \
        RF_CLUSTERS_LIMIT RF_BACKGROUND RF_NEAR RF_FAR RF_SOURCE_PATH \
        RF_TARGET_FRAMES_PER_CLUSTER 2>/dev/null || true
  eval "$out"
  export CFG="$cfg" PROFILE="$profile" P_ROOT="$root_dir"
  return 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  # 直接执行（bash load_config.sh [profile]）：演示已装载的关键变量
  load_config "${1:-full}"
  echo "PROFILE=$PROFILE"
  echo "P_CODE=$P_CODE  P_TOOLS=$P_TOOLS  P_VISUAL=$P_VISUAL  P_OUT=$P_OUT  P_CLOUDS=$P_CLOUDS"
  echo "ITR1=$ITR1  ITR2=$ITR2  RES=$RES  DENSIFY=$DENSIFY  METHODS='$METHODS'"
  echo "RF_NUM_CLUSTERS=$RF_NUM_CLUSTERS  RF_VIEWS_PER_CLUSTER=$RF_VIEWS_PER_CLUSTER  RF_GAZES_PER_CLUSTER=$RF_GAZES_PER_CLUSTER"
  echo "RF_SOURCE_PATH=$RF_SOURCE_PATH  RF_MAX_DEPTH=$RF_MAX_DEPTH  RF_OPACITY_THRESHOLD=$RF_OPACITY_THRESHOLD"
fi
