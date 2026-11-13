# MVSplat→RF-3DGS：视觉 3DGS 短训 + RRF 空间频谱重建（端到端）

用**前馈几何初始化**代替传统 SfM/COLMAP 稀疏点云，把 3DGS 训练成本砍到约 1/7.5，
并对同一物理场景做**射频（RF）空间频谱**的多方法重建与评测。

```
① 前馈 init 点云       ② 3DGS 视觉短训      ③ RRF 频谱重建 ×方法      ④ 评测
 位置簇 K×凝视 G×每簇V ─▶ 4000 步(res=2) ───▶ 每频谱方法一模型 ───────▶ 阶段一 161 视觉视角
   → init.ply           → chkpnt4000.pth     (冻结几何+颜色零学)       阶段二 每方法 640 频谱
```

一句话链路：**MVSplat 前馈出的高斯点云** 作为 3DGS 训练起点 → **视觉短训**（几何足够、只需短迭代拟合颜色）
→ 以短训 checkpoint 为起点做 **RRF（Radio Radiance Field）重训**（把“该位置往某方向看的谱”编码进高斯的
视角相关 SH 颜色）→ 按频谱方法独立评测，与**原版 RF-3DGS**（30000 步完整视觉训练）对齐预算对比。

---

## 1. 背景与定位

本仓库是你的**自建流水线封装**，不是以某一个开源项目为主。它借两件“平行依赖”，R2 起已**按功能合并进一个
代码树 `code/`**（不再按来源分目录，内核整格更名保留）：

| 角色 | 上游 | 合并后位置 | 在本流水线里的作用 |
|---|---|---|---|
| 引擎① 前馈几何 | MVSplat（ECCV 2024 Oral，[donydchen/mvsplat](https://github.com/donydchen/mvsplat)） | `code/feedforward/` | 阶段 A：预训练 re10k 权重单次前向产出 **init 点云** |
| 引擎② 3DGS/RRF | RF-3DGS（[SunLab-UGA/RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS)，arXiv:2411.19420） | `code/splat/` | 阶段 B/C：视觉短训 + RRF 频谱重建 |
| hydra 配置 | MVSplat | `code/config/` | 阶段 A 的 yaml 树 |
| 评测聚合/诊断 | 自写 | `code/tools/` | 结果 → 图/表、坐标/渲染自检 |

**来源与回退**：
- 引擎① 采用 MVSplat（根仓 git 历史本身含其上游提交；`code/feedforward/` = 其 `src/` 整格迁入）。
- 引擎② 采用 [SunLab-UGA/RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS) commit
  `332ca7e16525ffc3ac95667b328a64c6bed090e7`（`code/splat/` = 其平铺模块整格迁入）。
- **回退点**：融合前完整可用快照 = git commit `23c9900`，`git checkout 23c9900` 即回到融合前；两上游**原样代码树**
  完整保留在该 commit 的 `code/MVSplat/` 与 `code/RF-3DGS/` 下（含 README/LICENSE/历史脚本等非运行文件），需要时
  `git show 23c9900:<路径>` 即可取回原样件。
- **功能不变证据**：融合后（`results/run_20260905_225308`）与数据迁入 `datasets/` 后（`results/run_20260906_012243`）
  均以真实冒烟 e2e + 评测与融合前锚点 `results/run_20260905_201617` 对齐（init 高斯数 99,933 精确一致、阶段一/二
  指标差 ≤ ~1e-4 浮点噪声级）；逐项数值见各 run 的 `评测/*.json` 与 `results/mvsplat_clouds/provenance.json`。
- 代码结构与入口映射见 `code/README.md`。

**为什么这么做**：3DGS 传统上从 COLMAP 稀疏点云起步并暴力迭代很久。MVSplat 能从少量带位姿图片**前向**给出
像素对齐高斯，几何质量高 → 只需很短的视觉训练即可拟合颜色/透明度，随后把省下的预算花在**每个 RF 频谱方法**
自己的重训上。

**关键结论**（详见 `docs/`）：
- 阶段 B：MVSplat init + **4000 步短训 PSNR ≈32.07 dB**，反超原版 **30000 步完整训练（≈29.04 dB）**近 3 dB。
- 阶段 C：RRF 预算对齐（各 1 万次新迭代）后，与原版频谱重建**基本持平**（ΔPSNR 全部落在 ±0.22 dB 内）。
- 价值：以 **1/7.5 的视觉训练成本**（4000 vs 30000 步）达到与原版相同的第二阶段质量。

---

## 2. 数据（`datasets/`，全部不入 git）

```
datasets/
├── re10k/                          # 引擎① 官方预训练评测数据（*.torch chunk + index.json，~310M）
└── RF-3DGS_dataset/                # RF-3DGS 三套数据（4.4G）
    ├── blender_visual_dataset/     # 阶段 B 视觉数据：Blender 合成场景“光学照片”
    │                               #   transforms_train.json 809 帧 / transforms_test.json 161 帧，1600×900
    ├── training-rf-spectrum/       # 阶段 C RRF 输入：每方法一个 COLMAP 场景 3dgs_{M}_100/
    │   └── 3dgs_{MPC,CBF,Delay,MVDR,TCBF}_100/   #   3200 张谱图(2560 train/640 test)，300×200 PINHOLE 90°FOV
    │                               #   四个方法的 cameras/images.txt 完全相同（同阵列扫描位姿，只谱像素不同）
    ├── blender_visual_trained/     # 原版阶段一(视觉)已训模型：results.json = 原版参考权威源
    └── RF-3DGS_trained_RRF/        # 原版阶段二(RRF)每方法模型：3dgs_{M}_100/results.json = 原版参考权威源
```

- 数据目录名里 `3dgs_{M}_100` 的 `_100` = 谱合成阵列口径档位（本仓库只用 `_100`；`train_test_index_files/`
  里另有 `3DGS_{20,40,60,80}`，代码不解析该数字）。
- AoD 谱**无本地训练数据**、TCBF **无原版基线** → `config.yaml` 的默认 `methods` 只列可复刻的四方法
  `[MPC, CBF, Delay, MVDR]`。
- 阶段 A 另有官方权重 `checkpoints/re10k.ckpt`（48 MB）。数据/权重都已被 `.gitignore` 忽略。

---

## 3. 目录结构

```
mvsplat/                        # 根 = 唯一主 git 仓
├── config.yaml                 # ★ 所有超参数 + 路径（唯一事实来源）
├── load_config.sh              # config.yaml → shell env（run / evaluate 共用）
├── run_end_to_end.sh           # 端到端训练驱动（阶段 A→B→C，只训不评）
├── evaluate_run.sh             # 对已有 run 补评测（render + metrics + 汇总 json/png）
├── code/                       # 单 import 根（R2 起两上游按功能并入，不分来源）
│   ├── init_cloud.py           #   阶段 A 入口（前馈 init 点云）
│   ├── train.py  render.py  metrics.py   # 阶段 B/C 训练、渲染、指标（splat 内核入口）
│   ├── export_ply_scenes.py  rf_memory_sweep.py  rf_render_check.py   # 阶段 A 辅助入口
│   ├── config/                 # 阶段 A hydra yaml（原 MVSplat config）
│   ├── feedforward/            # 引擎① 阶段 A 前馈内核（原 MVSplat src）
│   ├── splat/                  # 引擎② 阶段 B/C 3DGS/RRF 内核（原 RF-3DGS）
│   ├── tools/                  # 自写：评测聚合 make_run_eval.py + rf_*.py 检查/诊断
│   └── README.md               # 合并树说明 + 入口映射
├── datasets/                   # 数据（re10k + RF-3DGS_dataset；不入 git）
├── checkpoints/                # 权重（re10k.ckpt）
├── results/                    # 端到端产物（不入 git）：run_<时间戳>/ + mvsplat_clouds/
└── docs/                       # 工作记录原件（不入 git）
```

---

## 4. code/ 详解：各组成部分的原理与实现

### 4.0 总览：两个“引擎” + 一层薄封装

```
        code/ 入口（sys.path[0]=code/，hydra config_path="config"）
  ┌────────────────────────────────────────────────────────────────┐
  │ init_cloud.py  …stage A   （直接调 feedforward.encoder 推理）  │
  │ train.py       …stage B/C  （直接调 splat 内核训练）           │
  │ render.py / metrics.py     （调 splat 渲染 + 度量）            │
  ├────────────────────────────────────────────────────────────────┤
  │ code/feedforward/ = 引擎①  纯推理库（只加载 re10k.ckpt）       │
  │ code/splat/       = 引擎②  训练/渲染内核                       │
  └────────────────────────────────────────────────────────────────┘
```

两引擎互不 import：阶段 A 与 B/C 之间传的只有**文件**（init.ply / chkpnt*.pth / point_cloud/*.ply），
通过 `RF_POINTCLOUD` 环境变量与 `--start_checkpoint` CLI 衔接。

### 4.1 引擎① `feedforward/` —— 前馈 3D 高斯（阶段 A）

**它是什么**：原 MVSplat `src/` 整格迁入。MVSplat 是**前馈稀疏视角 3DGS**：给 V 张带位姿的图，单次前向
（无优化、无可微渲染反传）就重建场景高斯，可渲染任意新视角。预训练于 re10k（360 视频轨迹），`mode=test`
时只当**推理库**用（仓库没有保留它的 Lightning 训练入口）。

**模块与数据流**（`encoder.forward`）：

```
context( V 张图 + 位姿 + near/far )
   └─ CNNEncoder(UniMatch 预训练, 1/4 分辨率, d=128)
        → 位置编码
        → MultiViewFeatureTransformer(Swin 式分窗自/交叉注意力)   # 每层跨其余 V-1 视角交换信息
        → 4D cost volume(把其它视角特征沿深度候选做 warp 相似度)
        → 沿逆深度的概率分布 softmax
        → 深度细化 UNet → 3 个 head: 视差/密度 · 高斯参数
   └─ GaussianAdapter: 逐像素射线 origins+directions·depth → 世界系 means;
        scale 被像素内参与深度缩放; 协方差 c2w_rot⊗cov⊗c2w_rotᵀ; SH 旋转到世界系
   └─ 输出 Gaussians: means/covariances/harmonics(sh_degree=4→25 系数)/opacities
        （opacity 由深度 PDF 平滑映射，再 /gaussians_per_pixel）
   └─ decoder_splatting_cuda: 光栅化渲染（推理/验证渲染，不反传）
```

- **像素对齐**：每个上下文视图的每个（全分辨率）像素产出 1 个高斯（V×H×W 个/前向），中心落在该视图
  射线方向、深度处 → 高斯**逐像素对齐**在各自 context 视图的射线网格上。
- 关键子目录：`model/encoder/`（costvolume 主模型、backbone/unimatch、common 的 gaussian_adapter、
  costvolume 的 depth_predictor_multiview）、`model/decoder/`（cuda 光栅化）、`dataset/`+`view_sampler/`
  （re10k 专用，见下）、`geometry/projection.py`（射线数学）、`model/ply_export.py`（3DGS .ply 导出）。
- **本项目不走 `Dataset/ViewSampler`**：`feedforward.dataset` 只注册了 re10k。阶段 A 的做法是照
  `init_cloud.py` 手搭一个 `context` dict（图像 + **归一化内参** + **c2w 外参** + near/far）直接喂
  `encoder.forward`。因此 MVSplat 原训练数据流（chunk、ViewSampler 抽帧、代价体……）在阶段 A 基本不参与，
  只有**预训练权重**里的几何能力被复用。

### 4.2 阶段 A 入口 `code/init_cloud.py` —— 多簇前向 union 成 init 云

**动机**：re10k 轨迹是“绕场景一圈”，而 RF 视觉数据是 **809 个地面站 × 8 个罗盘凝视方向**。一次前向
只吃 V 个共享视线的视角，混入不同方向的相机无法被 MVSplat 融合（散射深度片）。于是按
**位置簇 × 凝视扇区**拆成多次前向再 union，容忍冗余（后面 3DGS 自己会增删）。

算法（`@hydra.main`, `mode=test +experiment=re10k checkpointing.load=…`）：

1. 读 `transforms_train.json` 全部相机位姿；Blender→MVSplat c2w **翻轴** `c2w[:,:3,1:3]*=-1`；
   相机中心 `pos` = 平移列，视线 `look` = 第 3 列。
2. 对相机中心做**确定性 kmeans**(seed=0) 分 **K 个位置簇**（`RF_NUM_CLUSTERS`；K≤0 时由
   `RF_TARGET_FRAMES_PER_CLUSTER` 自动取 round(809/per)）。
3. 每簇内把视线量化成 **8 罗盘扇区**，取帧数≥2 的扇区前 **G 个**（`RF_GAZES_PER_CLUSTER`）；
   每扇区用**最远点采样**选 **V 个**上下文视角（`RF_VIEWS_PER_CLUSTER`）。某扇区不足 V 帧则跳过该前向
   （encoder 重排要求偶数 V）。
4. 重建 encoder：**V≠2 时把 `num_context_views=V` 写回配置并重建**（官方 ckpt 把上下文硬编码 2 视角，
   V=8 会静默拆成 4 对、质量骤降——这是文档记录的坑）；`ModelWrapper.load_from_checkpoint` 加载权重，
   `.eval().to(device)`。
5. 每簇前向：读图(RGBA TIF 合成到背景) → **归一化内参** → `rescale_and_crop` 到 256×256 → 手搭 context →
   `encoder.forward`。从 visualization dump 的 `scales/rotations` 与 `means/opacities/harmonics[DC]`
   取高斯参数，并按“view 主序”回算每高斯到所属相机中心距离。
6. **union + 过滤**：全部簇/扇区 concat → opacity ≥ `RF_OPACITY_THRESHOLD`(0.05) → 距自属相机 >
   `RF_MAX_DEPTH`(15) 的高斯丢弃（黑背景让 MVSplat 在 far≈100 吐一圈“黑壳”）。
7. **voxel 降采样**到 ≤ `RF_MAX_GAUSSIANS`：体素边长初始 ≈ 2×median(scales)，循环最多 12 次、超目标
   就 ×1.25，栅格内按单元平均点与属性。
8. 导出三件套（写到 `code/outputs/rf_init/`，再由驱动归档）：
   - `init.ply`：简单 RGB ply（仅 xyz+RGB，3DGS `fetchPly` 直接读）
   - `init_view.ply`：标准 3DGS ply（f_dc/opacity/scale/rot，**世界系不 re-center**，可拖进 supersplat 看）
   - `clusters.json`：K/V/每簇大小、跨距、选中视角与统计

**关键超参（env，经 `config.yaml` profile 注入）**：

| env | 含义 | smoke | full |
|---|---|---|---|
| `RF_NUM_CLUSTERS` | K 位置簇 | 1 | 16 |
| `RF_VIEWS_PER_CLUSTER` | V 每簇上下文视图 | 2 | 4 |
| `RF_GAZES_PER_CLUSTER` | G 凝视扇区/簇 | 1 | 2 |
| `RF_MAX_GAUSSIANS` | voxel 降采样上限 | 500k | 1.5M |
| `RF_OPACITY_THRESHOLD` / `RF_MAX_DEPTH` | 过滤 | 0.05 / 15 | 同 |

其它 `RF_*`（voxel_size、clusters_limit、background、near/far、source_path）见 `config.yaml` 的
`stage_a_defaults`，与 `init_cloud.py` 读的 env 一一对应。

**辅助入口**：
- `rf_render_check.py`：**渲染自检**。用与 init_cloud 完全相同的输入约定选一排朝南相机（CTX `0 6 12 18`→
  TGT `9`，可 env 覆盖），跑 encoder→decoder 渲染目标视角并与真图比 PSNR——PSNR 明显高于“垃圾”即证明
  位姿约定/context 构造正确。
- `export_ply_scenes.py`：把对 **re10k 真数据**的预测高斯导出为可拖 supersplat 的 .ply（查看用途，会
  re-center，坐标系与 init.ply 不同）。
- `rf_memory_sweep.py`：V=2..16 的显存扫描（≈0.66 GiB/视角），定 V 上限用。

### 4.3 引擎② `code/splat/` —— 3DGS 训练 / RRF 内核（阶段 B/C）

**它是什么**：原 RF-3DGS 整格迁入（vanilla 3DGS + 少量 RF 改造）。**相对普通 3DGS 只有三处代码级改动**，
其余“RRF 语义”全部由数据 + CLI 协议承载：

1. `scene/dataset_readers.py`：COLMAP 场景用 `train_index.txt/test_index.txt` 划分（RF-3DGS 自带改造），
   并新增**外部 init 点云读取**（下详）。
2. `train.py:42-50`：`--start_checkpoint` 触发 **RRF 重训块**。
3. 阶段 C 用 `--densify_until_iter 0` 关闭增密/剪枝（CLI 级）。

**模块**：
- `arguments/`：ModelParams / OptimizationParams / PipelineParams 三个参数组；`get_combined_args` 让
  render/metrics 从 `<model_path>/cfg_args` 读回训练参数。
- `scene/`：`Scene`（数据集识别 + 相机装载 + 点云装载/保存）、`cameras.py`（w2c/投影）、
  `colmap_loader.py`、`dataset_readers.py`、`gaussian_model.py`（属性/SH/增密剪枝/存取）。
- `gaussian_renderer/`：包装 `diff_gaussian_rasterization` 的可微光栅化（无任何频谱代码）。
- `utils/`：camera/图形/图像/损失/SH/系统工具；`lpipsPyTorch/`：vendored LPIPS（vgg/alex/squeeze）。

**训练主流程（`train.py`，阶段 B/C 同一条）**：

```
Scene(dataset, gaussians)   # 依 source_path 有无 sparse/ 选 Colmap，有无 transforms_*.json 选 Blender
  └─ GaussianModel(sh_degree=3)
       └─ 无 checkpoint → create_from_pcd(场景点云)   ← 外部 init 云在此进入
       └─ 有 --start_checkpoint → torch.load → gaussians.restore()
            └─ RRF 重训块: _xyz/_scaling/_rotation 冻结(requires_grad=False);
                 _features_dc/_features_rest 清零(颜色从零学); _opacity 可训
训练循环:
  for it in range(first_iter+1, iterations+1):
     随机取一训练相机 → render() → loss = (1-λ)L1 + λ(1-SSIM) → backward
     仅当 it < densify_until_iter 才增密/剪枝/reset_opacity   # 阶段 C 传 0 → 永不
     迭代 ∈ save_iterations  → point_cloud/iteration_<it>/point_cloud.ply
     迭代 ∈ checkpoint_iterations → chkpnt<it>.pth = (capture(), iteration)
```

- **阶段 B（视觉短训）**：`-s blender_visual_dataset --resolution 2 --iterations ITR1`，fresh 训练，外部
  init 云经 `RF_POINTCLOUD` 注入；存 `chkpntITR1.pth`。
- **阶段 C（RRF，每方法）**：`-s …/3dgs_{M}_100 --iterations ITR2 --start_checkpoint …/chkpntITR1.pth
  --densify_until_iter 0`。`first_iter` 记录在 checkpoint 里 → 实际新训 = ITR2−ITR1 次（如 full 的
  5000→15000 = **1 万次 RRF**，与原版“30001→40000 无剪枝”逐条对齐）。

**RRF 到底在做什么**：RF-3DGS 两阶段。阶段一用**光学照片**学几何；阶段二把**射频空间频谱图**（“RF 照片”，
像素 = 波束成形功率谱）当监督，**冻结几何、颜色从零学**，让高斯的 view-dependent SH 颜色把“该位置沿某方向
看的谱”编码进辐射场——这样任意相机位姿都能渲染出谱图。几何共享、频谱各自编码，所以**每个方法各训一个
RRF 模型**。

**四种方法 = 同一信道的不同观测投影**（非不同场景）：

| 方法 | 本质 | 特点 |
|---|---|---|
| CBF | 常规波束成形（Bartlett 角度谱） | 固定导向矢量扫描，主瓣宽、稳健但分辨率低 |
| MVDR | Capon 自适应波束成形（角度谱） | 自适应抑制干扰，零陷尖、分辨率高、对误差敏感 |
| Delay | 时延谱（功率随时延分布） | 时域多径到达时间，类“RF 深度图” |
| MPC | 多径分量谱 | 分辨出的离散多径组件 |

谱图/位姿在仿真（上游用 Sionna）时就定死写进 COLMAP；四个方法 `images.txt/cameras.txt` 逐字相同，
代码里方法名只作为**数据目录名**被消费。

**外部 init 点云怎么进来（`_external_pointcloud_path`, `dataset_readers.py:132-179`）**——二级回退 +
兜底：
1. `RF_POINTCLOUD` env（阶段 B 由驱动 export 为本次 run 的新云）；
2. `code/splat/configs/pointcloud.yaml` 的 `point_cloud` 键（绝对 fallback，当前指
   `results/mvsplat_clouds/init.ply`）；
3. 都没有 → 数据集自带 `points3d.ply`（spectrum 集有），再无 → 随机 100k 点。

外部云经 `fetchPly` 读 `xyz + RGB`，`create_from_pcd` 里：xyz→位置、RGB→`RGB2SH` 填 DC 通道、scale 由
KNN 距离定、rot=单位四元数、opacity=inv_sigmoid(0.1)。**必须与相机世界系对齐**（`init_cloud` 特意不做
re-center）——否则高斯不会出现在训练视图的正确位置、无法拟合。
阶段 C 训练前驱动会 **`unset RF_POINTCLOUD`**，让 spectrum 集用自己的 colmap 点云，防止 init 云泄漏进
频谱数据。

### 4.4 评测：`render.py` / `metrics.py` / `tools/make_run_eval.py`

- `render.py`：`Scene(..., load_iteration=N)` 加载 `point_cloud/iteration_N` → 渲 `test/ours_<N>/renders`。
  `--eval --skip_train` = 只渲测试集（阶段 B：161 视觉测试视角 res=2；阶段 C：每方法 640 测试频谱原生分辨率）。
  已渲染则跳过（幂等，省 GPU）。
- `metrics.py`：逐模型遍历 `test/` 下每张渲染 vs GT 算 **PSNR/SSIM/LPIPS**（psnr 在
  `utils/image_utils`、ssim 在 `utils/loss_utils`（11×11 高斯窗）、LPIPS 用 vendored VGG）→ 写
  `<model>/results.json` 与 `per_view.json`。
- `tools/make_run_eval.py`：聚合一个 run 的 results.json（自动发现阶段一全部 `ours_<i>` 档与
  阶段二每方法全部档）→ 阶段一迭代扫描 `评测/{阶段一评测.json,阶段一_指标图.png}` + 每档
  `评测/快速查看/ours_<i>_4视角对比.png`、阶段二 `评测/{阶段二评测.json,阶段二_方法对比图.png}`、
  `三方对比/{三方对比.png,对比数据.json}`，并按 `config.yaml eval.reference` 嵌“原版 RF-3DGS”
  参考列（`eval.enabled` 可关）。
- `tools/mvsplat_coverage.py`：CPU 覆盖度评测（4 位置 z-buffer 投影 → `覆盖率.png` /
  `覆盖率视角_原图vs投影.png` / `验证数据.json`），口径同 baseline（rf_view_check）。

### 4.5 自写诊断工具 `code/tools/rf_*.py`（均在原版模型/数据上做无 GPU 或轻检查）

- `rf_pose_check.py`：纯 numpy 位姿几何检查（相机基线是否够、主轴是否交于一点 → 判断“点线矩形云”成因）。
- `rf_warp_check.py`：无模型光度 warp 自检（两张 RF 视图按极线重投影 NCC）。
- `rf_gaze_texture.py`：诊断每簇凝视选择是否落在“void”方向（天空/地面）。
- `rf_gaze_poolsize.py`：给定 K 看每个凝视扇区帧池是否 ≥ V（安全 (K,V) 组合）。
- `rf_cloud_check.py` / `rf_view_check.py` / `rf_view_check2.py`：把 init 云投影进真相机做 z-buffer 颜色/
  掩膜对齐自检（读 `code/outputs/rf_init`）。
- `rf_sweep_eval.py`：短训迭代扫描与参考模型的空间外点分析。

---

## 5. 配置（唯一事实来源：根 `config.yaml`）

| 键 | 语义 |
|---|---|
| `paths.*` | 仓库/数据/权重/产物路径（`code_root`、`data_visual/spectrum/orig_*`、`ckpt_re10k`、`clouds`、`results`、`python`）。**数据统一在 `datasets/RF-3DGS_dataset/`** |
| `profiles.<name>` | 每 profile 的分组超参（K/V/G、max_gaussians）+ 阶段步数（`stage1_iters`/`stage2_iters`，及多档 `stage1_save_iters`、双档 `stage2_total_iters`）+ 方法列表 |
| `stage_a_defaults` | 阶段 A 全套 `RF_*` 超参（profile 同名键覆盖） |
| `stage_b.resolution` | 视觉短训渲染分辨率档（2 → 800×450） |
| `stage_c.densify_until_iter` | 阶段 C 冻结几何（0 = 不增密不剪枝） |
| `eval.*` | 评测开关 + 覆盖度评测位置（`mvsplat_coverage.cams`，0/200/400/600）+ 原版参考数值（权威源为原版 `results.json`，勿手改） |

**两个内置 profile**：

| | smoke（链路验证） | full（正式结果） |
|---|---|---|
| K / V / G | 1 / 2 / 1 | 16 / 4 / 2 |
| max_gaussians | 500k | 1.5M |
| stage1_iters（短训末迭代） | 300 | **12000**（视觉 LPIPS 目标 < 原版 30000 步 0.1265） |
| stage1_save_iters（迭代扫描档） | 300 | 500 1000 … 12000（每 1000，13 档） |
| stage2_iters（10000RRF 对齐档） | 500（=300+200） | **22000**（=12000+10000 RRF） |
| stage2_total_iters（40000总 档） | （无） | 40000（对应原版 40000 刻；RRF 单跑冻结几何存两档快照） |
| methods | MPC | MPC CBF Delay MVDR |

`load_config.sh`（须在 WSL bash 下 source）把 yaml 解析成 `export`：`P_CODE/P_TOOLS/P_VISUAL/…`、
`ITR1/STAGE1_SAVE/ITR2/TOTAL2/RES/DENSIFY/METHODS`、全部 `RF_*`、`PY`、`EVAL`，供两个驱动脚本使用。
（同 shell 连续 `load_config` 不同 profile 会先清空上一 profile 独有键，避免 TOTAL2 等残留。）

---

## 6. 完整使用流程

### 6.0 前置
- WSL2（Ubuntu）+ conda env `mvsplat`（python=config.yaml `paths.python`），已装好 torch2.1.2+cu118、
  `diff-gaussian-rasterization`（源码编译）、hydra、pyyaml、plyfile、lpips 依赖。
- 数据在 `datasets/`、权重在 `checkpoints/re10k.ckpt`（见 §2）。
- 全部驱动在 **WSL bash、仓库根** 运行（`.sh` 已保证 LF 行尾；Windows 侧 CRLF 检出不影响运行）。

### 6.1 一条龙：训练 + 评测
```bash
cd /path/to/w-RFGS

# 1) 端到端训练（阶段 A→B→C，只训三段，不评测）
bash run_end_to_end.sh full            # = bash run_end_to_end.sh full replace
#    mode: replace 默认 = 新 init 云覆盖 results/mvsplat_clouds/latest，旧云入 archive
#        | no-replace  = latest 不动，新云仅压入 archive（冒烟/并行对比时保护主云）

# 2) 对刚生成的 run 补评测（渲染 + 指标 + 汇总 json/png）
bash evaluate_run.sh run_<时间戳>      # profile 缺省自动读该 run 的 profile.txt

# 结果：results/run_<时间戳>/
```
冒烟验证（快、不污染主云）：
```bash
bash run_end_to_end.sh smoke no-replace && bash evaluate_run.sh run_<时间戳> smoke
```

### 6.2 评测使用流程（独立于训练，可反复补跑）

评测**只读一个已训完的 run、不改任何权重**：只要 run 里有 `rf-3dgs_1` 与 `rf-3dgs_2/<M>` 的
`point_cloud/iteration_*`，随时可以对它补评测或重跑（幂等：每档渲染已存在会自动跳过，省 GPU）。
评测内容 = baseline（`run_20260905_190255_baseline`）那套评测的通用等价物，自动发现档位，
不写死单档。

```bash
cd /path/to/w-RFGS
bash evaluate_run.sh run_<时间戳>               # profile 缺省 → 自动读 run/profile.txt
bash evaluate_run.sh results/run_<时间戳> full  # 也可给完整路径 + 显式 profile
```

**它做五件事**（`evaluate_run.sh` 自动编排，全部幂等）：

1. **mvsplat 覆盖度**（CPU 秒级）。对 `mvsplat/点云/init.ply` 投影到 4 个位置（cams 0/200/400/600，
   与 baseline 同口径 FX=FY=800 @1600×900、z-buffer nearest-wins）→ `mvsplat/{覆盖率.png,
   覆盖率视角_原图vs投影.png, 验证数据.json}`。
2. **阶段 B 多档迭代扫描**。自动发现 `rf-3dgs_1/point_cloud/iteration_*` 全部档，逐档
   `render.py --iteration <i> --eval --skip_train`（视觉测试集 809 训练之外 161 held-out，
   `--resolution RES`）→ 一次 `metrics.py` 得全部档 PSNR/SSIM/LPIPS（`rf-3dgs_1/results.json`）。
   同时每档生成 **4 视角「原图|渲染」快速查看**（`评测/快速查看/ours_<i>_4视角对比.png`，
   只存 4 个位置防海量图；全量数值仍对所有 161 视角评测）。
3. **阶段 C 双档逐方法**。自动发现每方法存档档：full = 10000RRF 对齐档（绝对 `ITR2`=22000）+
   40000总 档（绝对 `TOTAL2`=40000）；smoke = 单档。在 `3dgs_{M}_100` 上渲染 **640 张测试频谱**
   （原生分辨率，不传 `--resolution`）→ 各 `rf-3dgs_2/<M>/results.json`。
4. **聚合**。`make_run_eval.py` 汇总 → 阶段一扫描图/评测.json、阶段二评测.json + 方法对比图、
   **三方对比**（`rf-3dgs_2/三方对比/{三方对比.png, 对比数据.json}`：原版 / 40000总 / 10000RRF），
   并按 `config.yaml eval.reference` 嵌 **“原版 RF-3DGS”参考列**。
5. **参数量曲线**。`make_params_curve.py`（我方多档高斯球数 vs 原版 `blender_visual_trained`）→
   `三方对比/参数量曲线_我的方法vs原版.png/.json`。

每步日志在 `run/logs/*.log`；某步失败可修后原命令重跑，已渲染的模型直接跳过。

**产物与判读**：

| 看什么 | 位置 | 含义 |
|---|---|---|
| 覆盖度 | `mvsplat/{覆盖率.png, 验证数据.json}` | init 云在 4 位置画面覆盖率（~10–26%，同 baseline）+ 命中精度/深度 |
| 覆盖度快速查看 | `mvsplat/覆盖率视角_原图vs投影.png` | 4 位置 原图(实拍合成) vs 点云投影 对比 |
| 阶段一迭代扫描 | `rf-3dgs_1/评测/阶段一评测.json` + `阶段一_指标图.png` | 本 run 全部 `ours_<i>` 档 vs 原版 `ours_30000`/`ours_7000`（含 高斯球数、`达标判断`: 最优 LPIPS < 原版 0.1265?） |
| 阶段一快速查看 | `rf-3dgs_1/评测/快速查看/ours_<i>_4视角对比.png` | 每档仅 4 个位置 原图(gt) vs 生成图(render) |
| 阶段二双档 | `rf-3dgs_2/<M>/评测/阶段二评测.json` + `阶段二_方法对比图.png` | 每方法 `ours_<ITR2>`(10000RRF 对齐) 与 `ours_<TOTAL2>`(40000总) vs 原版 `ours_40000`（参考来自 `RF-3DGS_trained_RRF`） |
| 三方对比 | `rf-3dgs_2/三方对比/` | 原版 / 40000总 / 10000RRF 三方柱状 + 对比数据.json + 参数量-质量曲线 |
| 逐视角明细 | 各模型 `results.json` / `per_view.json` | 单个视角级指标（找“坏视角”用） |
| 原版参考数值 | `config.yaml eval.reference` | 权威源为原版 `results.json`，已冻结勿手改；`eval.enabled: false` 可去掉参考列 |

**想只评某一段 / 单方法**（不进驱动、手工补，变量来自 `load_config` 的 export）：

```bash
cd code
# 阶段 B 只补一个模型（渲染已存在则 metrics 直接重算）
"$PY" render.py  -s "$P_VISUAL" -m "$RUN/rf-3dgs_1" --iteration "$ITR1" --eval --skip_train --resolution "$RES"
"$PY" metrics.py -m "$RUN/rf-3dgs_1"
# 阶段 C 单方法：-s 指向对应 3dgs_{M}_100，原生分辨率
"$PY" render.py  -s "$P_SPECTRUM/3dgs_MPC_100" -m "$RUN/rf-3dgs_2/MPC" --iteration "$ITR2" --eval --skip_train
"$PY" metrics.py -m "$RUN/rf-3dgs_2/MPC"
# 再跑聚合即可出新 json/png（档位自动发现，无需 --iters1/--iters2）：
"$PY" "$P_TOOLS/make_run_eval.py" --run "$RUN"
```

### 6.3 一个 run 里有什么
```
results/run_<时间戳>/
├── mvsplat/              ① init 点云 + 覆盖度评测（覆盖率.png / 覆盖率视角_原图vs投影.png / 验证数据.json）
│   └── 点云/             init.ply + init_view.ply + clusters.json（本次 run 的新云）
├── rf-3dgs_1/            ② 视觉短训：chkpnt<ITR1>.pth + point_cloud/iteration_<档位…ITR1>/
│                            + test/ours_<档位>/（评测后）+ 评测/（含 快速查看/ 4 视角对比图）
├── rf-3dgs_2/            ③ RRF：<M>/point_cloud/iteration_{ITR2,TOTAL2?}/（评测后 test/ + 评测/）
│   └── 三方对比/         三方对比.png + 对比数据.json + 参数量曲线_我的方法vs原版.png/.json
├── logs/                 三阶段 + 评测日志
└── README.md  profile.txt  cloud_mode.txt
```
共享仓库 `results/mvsplat_clouds/`：`init.ply` 等指向“最近一次 replace 的前向”；`archive/` 存各历史版本
（按 `_K{V}V{V}G{G}` 标签）；`provenance.json` 记录 latest + history（每次 run 都记一条，replace 才改 latest）。
每次 run 自己的阶段 B **总是吃本次 run 的新云**（run 内副本），mode 只决定共享仓库记法。

### 6.4 点云刷新语义（replace vs no-replace）
- `replace`：新云覆盖 `latest/init.ply`，原 latest 先压入 archive → 供下一轮手动跑 splat 复用。
- `no-replace`：latest 不动，新云直接入 archive → 适合并行跑多组冒烟/对比，互不污染。

### 6.5 手工只跑某一段（调试）
```bash
cd code
# 阶段 A 单跑（用 config.yaml paths.python）
/home/…/mvsplat/bin/python init_cloud.py mode=test +experiment=re10k \
    checkpointing.load=/path/to/w-RFGS/checkpoints/re10k.ckpt   # RF_* 走 env

# 阶段 B 单跑（外部云经 RF_POINTCLOUD；--save_iterations 空格分隔多档，末尾=ITR1 保末档云）
export RF_POINTCLOUD=…/init.ply
/home/…/mvsplat/bin/python train.py -s $P_VISUAL -m rf-3dgs_1 --resolution 2 \
    --iterations 12000 --save_iterations "500 … 12000" --checkpoint_iterations 12000

# 阶段 C 单跑（RRF，冻结几何 0；full 双档=单跑到 40000 存 22000 与 40000 两快照）
unset RF_POINTCLOUD
/home/…/mvsplat/bin/python train.py -s …/3dgs_MPC_100 -m out_c --iterations 40000 \
    --start_checkpoint …/chkpnt12000.pth --densify_until_iter 0 --save_iterations "22000 40000"
```

### 6.6 改方法 / 改 profile
- 方法：`config.yaml profiles.<name>.methods`（可复刻：MPC/CBF/Delay/MVDR）。
- 视觉成本：调 `stage1_iters`（短训末迭代）与 `stage1_save_iters`（迭代扫描存档档，供 4 视角/
  扫描曲线，空格分隔；需含 ITR1）。若想让最终视觉 LPIPS 优于原版 30000 步，需把 ITR1 抬到
  实测达标为止（当前 full=12000）。
- RRF 预算：`stage2_iters`（10000RRF 对齐档绝对末迭代 = ITR1+10000）；如需“40000总”对照
  （对应原版 40000 刻）加 `stage2_total_iters`（40000）。冻结几何单跑到后者即同时存两档快照。
- 分辨率/几何冻结：`stage_b.resolution`、`stage_c.densify_until_iter`。

---

## 7. 文档与历史

- 工作记录/复现摘要：`docs/`（`方案一工作记录.md`、`第二步-RRF-多方法评测.md`、`REPRODUCTION_SUMMARY.md`）。
- 每 run 明细：`results/run_<时间戳>/` 与 `results/run_20260905_190255_baseline/`（内 `工作记录.md`）。
- 代码结构/入口映射：`code/README.md`；上游 commit、回退点 `23c9900`、功能不变证据见 §1「来源与回退」。

## 8. 常见坑

- **行尾**：`*.sh` 必须 LF（`.gitattributes` 已强制）；Windows 检出 CRLF 会让 WSL bash 报 `\r` 错。
- **MSYS 传参**：在 Windows git-bash 里调 WSL 用 `MSYS_NO_PATHCONV=1 wsl bash -lc '…'`，防路径被改写。
- **MVSplat 视角数**：re10k ckpt 把上下文硬编码 2 视角；`V≠2` 必须走 `init_cloud.py` 的
  `num_context_views` 重建逻辑，否则 V=8 静默拆 4 对、PSNR 从 ~24-30dB 掉到 ~12-21dB。
- **内参**：喂 encoder 的内参必须是**归一化**（fx/W…），raw 像素内参会把模型打到 ~1700 倍分布外。
- **坐标系**：阶段 A 输出不 re-center，必须与 RF-3DGS 位姿世界系对齐；Blender c2w 需翻轴。
- **阶段 C 勿漏 `unset RF_POINTCLOUD`**：否则 init 云会泄漏进频谱数据（驱动已处理；手工跑见 §6.5）。
- **glm/CUDA 环境**：conda 内 `import glm`、rasterizer 需与系统 nvcc 匹配（torch 2.1.2+cu118），详见
  `docs/REPRODUCTION_SUMMARY.md` 的坑位记录。
