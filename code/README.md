# code/ —— 功能地图（2026-11-23 包化整理）

`code/` 仍是唯一 import 根；所有入口都按 `python code/rfgs/<子包>/<脚本>.py` 运行。
每个脚本顶部有自动插入的 `_ROOT` 前导，保证 `import sim/splat/rfgs.*` 命中 `code/`。

## 1. `code/rfgs/` —— 本项目代码

| 子包 | 内容 |
|---|---|
| `core/` | `angular_data.py`（相机/谱图读取）、`metrics.py`、`init_cloud*.py`、`export_ply_scenes.py`、`rf_memory_sweep.py`、`rf_render_check.py` |
| `csi/` | **主模型**：`train_csi_cbf.py`（含 `render_cir`/`CBFGrid`/`spec_raw`/`spec_from_cir_mvdr`） |
| `rrf/` | 标量/角域/SH 三条 RRF 线：`train_rrf_{scalar,angular,mlp,sh}.py`、`render_rrf_*.py`、`render_sh_rrf.py` |
| `coupled/` | CORF/耦合/蒸馏：`train_corf.py`、`render_corf.py`、`train_student.py`、`train_coupled_student.py`、`render_student.py` 等 |
| `xscene/` | 跨场景 6 合成房间：`train_xscene.py`、`eval_xscene.py`、`render_knn_xscene.py` |
| `geo2rf/` | 前馈几何→RF 线：`train_geo2rf*.py`、`render_geo2rf*.py`、`render_spatial_gt.py` |
| `upstream/` | 上游基线：`train_nerf2.py`、`render_nerf2.py`、`train_cov.py`、`train_resume.py` |
| `engine/` | 原 3DGS 引擎入口：`train.py`、`render.py` |
| `evaluation/` | `eval_csi_cbf.py`、`eval_csi_downstream.py` |

> `hydra` 依赖的 `core/init_cloud*`、`export_ply_scenes`、`rf_memory_sweep`、`rf_render_check`
> 属于 MVSplat 阶段入口；在 `rf` 环境里没有 hydra，导入会报错（既有环境问题，非本次整理引入）。

## 2. 其它内核（保持包名）

`sim/`（Sionna 生成器、`w_codebook`、`spectrum_gen`、`delay_spec`）、`splat/`（3DGS 内核与颜色/标量头）、
`corf/`、`baselines/`、`feedforward/`、`config/`。

## 3. `code/tools/` —— 脚本层（按功能）

```
srv/      服务器驱动、watch、队列、上传
train/    训练驱动、扫描、多臂
eval/     指标聚合、逐实例/逐视角、SE/DBF、w′
figures/  论文图、拼图、导出
data/     生成/打包/上传/jet↔标量/划分
diag/     一次性诊断、探针
archive/  2026-11-15 前的历史脚本（不维护）
```

根目录只保留 3 个文件：`__init__.py`、`graf_prepare.py`、`make_student_features.py`
（后两个被 `rfgs.geo2rf` / 学生模型以 `tools.xxx` 形式 import，故不移动）。
所有驱动/评测/数据/诊断脚本都在上面的子目录里。

## 4. 运行约定

- 入口：`python code/rfgs/csi/train_csi_cbf.py ...`（旧命令 `code/train_csi_cbf.py` 已全部改掉）；
- 路径：服务器根 `/root/autodl-tmp/rf`，本地为仓库根；不要假设 cwd；
- 新脚本：放对应子目录，并在顶部加 `_ROOT=.../../..` 的 sys.path 前导。
