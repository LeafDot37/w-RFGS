# code/ —— 合并后的单 import 根（R2）

R2 把两个上游代码树**按功能**并入这一个目录，不再按来源（MVSplat / RF-3DGS）区分。
两引擎内核**整格保留、包更名**，内部 import 晶格未动；跨引擎的同名模块（dataset/loss/utils/…）一律靠包前缀隔离、**永不合并函数体**。

## 布局

```
code/                              # = 单 import 根（cd code; python xxx.py 即命中）
├── config/                        # 阶段A hydra yaml（原 MVSplat config，整树搬入）
├── feedforward/                   # 引擎① 阶段A 前馈内核（原 MVSplat src，包更名）
│   ├── __init__.py
│   ├── config.py  global_cfg.py
│   ├── dataset/  evaluation/  geometry/  loss/  misc/  model/  visualization/
│   └── …                          # 内部相对 import 晶格原样未动（含运行时必需的 geometry/）
├── splat/                         # 引擎② 阶段B/C 3DGS 内核（原 RF-3DGS 平铺模块 → 正规子包）
│   ├── __init__.py
│   ├── arguments/  scene/  gaussian_renderer/  lpipsPyTorch/  configs/
│   └── utils/ (+ __init__.py NEW) #   scene/dataset_readers 读外部 init 云；
│                                   #   configs/pointcloud.yaml = 外部 init 云回退路径
├── tools/                         # 本项目自写：评测聚合 + 检查/诊断脚本
│   ├── make_run_eval.py           # results.json → 阶段一/二 评测 json+png（含原版参考）
│   └── rf_*.py                    # 各阶段检查/诊断（读 code/outputs/rf_init）
└── 入口脚本（sys.path[0]=code/、hydra config_path="config" 须与入口同级）：
    init_cloud.py      # 阶段A 前馈 init 点云（原 rf_init_cloud.py，更名）
    export_ply_scenes.py  rf_memory_sweep.py  rf_render_check.py   # 阶段A 辅助（hydra 入口）
    train.py  render.py  metrics.py                                # 阶段B/C 训练与评测
```

## 入口映射（由哪个上游来）

| 现在 | 原来源 |
|---|---|
| `code/init_cloud.py` | MVSplat `rf_init_cloud.py` |
| `code/export_ply_scenes.py` 等 3 个 | MVSplat 根同名 hydra 脚本 |
| `code/config/` | MVSplat `config/` |
| `code/feedforward/` | MVSplat `src/`（含 geometry 等全部运行时代码） |
| `code/splat/` | RF-3DGS `{arguments,scene,gaussian_renderer,utils,lpipsPyTorch,configs}` |
| `code/train.py` `render.py` `metrics.py` | RF-3DGS 根同名入口 |

两上游的原样件（README/LICENSE/死代码/历史脚本等非运行文件）不再单独存档：它们完整保留在回退点
`git 23c9900` 的 `code/MVSplat/`、`code/RF-3DGS/` 树里，需要时 `git show 23c9900:<路径>` 取回；来源与 commit 见仓库根
README §1「来源与回退」。

## 运行

统一由仓库根驱动脚本驱动（配置文件 = 仓库根 `config.yaml`，唯一事实来源）：

```bash
# WSL bash，仓库根
bash run_end_to_end.sh [full|smoke] [replace|no-replace]   # 只训练（阶段 A→B→C）
bash evaluate_run.sh run_<时间戳> [profile]                  # 对已有 run 补评测
```

手工跑单入口（示例）需 `cd code`：

```bash
cd /path/to/w-RFGS/code
<config.yaml paths.python> init_cloud.py mode=test +experiment=re10k checkpointing.load=<ckpt>
<config.yaml paths.python> train.py -s <data> -m <out> --resolution 2 --iterations <ITR1>
```

> 注意：阶段A 瞬时产物（hydra 日志 + init 点云）落在 `code/outputs/`，已 gitignore。
> 代码里入口脚本 docstring 仍可能残留旧目录（`code/MVSplat`、`code/RF-3DGS`）字样——纯注释，已归档于上游，不在本运行路径内。
