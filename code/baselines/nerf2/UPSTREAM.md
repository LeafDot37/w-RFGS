# UPSTREAM —— NeRF² 来源、校验与偏离声明

## 1. 来源与校验

| 项 | 值 |
|---|---|
| 仓库 | `XPengZhao/NeRF2` |
| 许可 | MIT，`Copyright (c) 2023 ZHAO Xiaopeng`（**注意版权行是真名，不是 GitHub 用户名**） |
| 取件通路 | `https://cdn.jsdelivr.net/gh/XPengZhao/NeRF2@main/<path>` |
| 取件日期 | 2026-09-16 |
| 钉版方式 | **内容 sha256**（该仓无 release，jsdelivr 版本 API 返回 502，commit sha 取不到） |

### 为什么走 jsdelivr

本机（Windows Git Bash）实测：`raw.githubusercontent.com` 与 `api.github.com`
**均不通**（`curl` exit 35，SSL 连接失败）；服务器侧 `raw.githubusercontent.com`
亦不通。**jsdelivr CDN 是唯一可用通路**，`http=200`。

> 遗留风险：`@main` 是浮动引用。sha256 校验把这条风险消掉了 ——
> 内容一旦变化，`verify_nerf2_gates.py` 的 G7 会立刻失败。

### 逐字 vendor 的文件

| 文件 | 字节 | sha256 |
|---|---|---|
| `model.py` | 5870 | `c63fa4246d6255035d03ebc1dbd7b5a1d40391b51e98805ed3d9969bc3546ad1` |
| `renderer.py` | 12582 | `4f5e5b1ab617de27918e8d2bc540182a110d37eb6ac5e743449c34b3a8535c92` |
| `LICENSE` | 1070 | `c308fc9f0bca53656b48b0821a44cb387063a020636e1cff2182aaf83a3e2757` |

三者落盘后**行尾均为纯 LF**（jsdelivr 返回的即是上游字节）。校验命令：

```bash
sha256sum code/baselines/nerf2/{model.py,renderer.py,LICENSE}
```

### 取件但**不 vendor** 的上游文件

| 文件 | 为什么不要 |
|---|---|
| `nerf2_runner.py` | 顶层 `import tensorboardX` / `from utils.logger import Logger`（需 `configargparse`），且 `eval_network_spectrum` 用 `skimage.ssim(..., multichannel=)` —— 该参数在 **skimage ≥ 0.26 已被删除**，直接跑必然 `TypeError`。训练循环我们自己写（800 行不到，且要接我们的数据）。 |
| `dataloader.py` | 原生布局是 `gateway_info.yml + tx_pos.csv + spectrum/*.png`，与我们的 `images/ + sparse/0/ + train_index.txt` 完全不同。**但已通读，用于核对样本布局与标签范围**（见 §3）。 |
| `baseline/mri.py` | 需 `sklearn`。我们不跑 MRI 对比。 |
| `utils/data_painter.py` | 仅可视化用，`matplotlib`。我们出图走本仓 `make_*` 工具链。 |
| `utils/logger.py` | 需 `configargparse`。 |
| `configs/*.yml` | 配置文件，取值已抄录进 §3。 |

**结论：不装任何 Python 包。** `scipy`（`renderer.py` 顶层 `import scipy.constants as sc`）
与 `einops`（顶层 `from einops import rearrange, repeat`）是 `renderer.py` 的**硬 import**，
但两者**只被 `Renderer_RSSI` / `Renderer_CSI` 使用**，而我们的 spectrum 路径只碰
`Renderer_spectrum`。它们仍需存在于环境里（否则 import 失败），但**不引入任何新的
数值依赖** —— gs3d 环境已实测两者皆在。

---

## 2. 上游参照配置（`configs/rfid-spectrum.yml`，逐字抄录）

```yaml
path:   {expname: spectrum-exp1, datadir: data/RFID/s23/, logdir: logs/RFID/s23/}
render: {scale_worldsize: 1, mode: 'spectrum', near: 0, far: 5, n_samples: 64}
train:  {lr: 8e-4, weight_decay: 5e-5, T_max: 10000, eta_min: 1e-6,
         batch_size: 8192, total_iterations: 30000, load_ckpt: True,
         save_freq: 100, val_freq: 2500, report_freq: 100}
networks: {D: 8, W: 256, skips: [4],
           multires:    {'pts': 10, 'view': 10, 'tx': 10},
           input_dims:  {'pts': 3,  'view': 3,  'tx': 3},
           is_embeded:  {'pts': True, 'view': True, 'tx': True}}
```

> ⚠ **`is_embeded.tx` 是 `True`**（`tx` 要做位置编码）。
> `model.py` 的**默认参数**写的是 `is_embeded={'tx': False}`，那是另一回事 ——
> 这份 spectrum 配置覆盖了它。照默认值写会少 60 维输入。
> （`input_tx_dim` = 3 + 3·2·10 = **63**，`input_view_dim` 同为 63，`input_pts_dim` 同为 63。）

我们沿用全部网络超参（D/W/skips/multires/input_dims/is_embeded）、优化器超参
（lr/wd/T_max/eta_min）、`batch_size=8192`、`total_iterations=30000`。
**只改三个 `render` 字段**（`near` / `far` / 语义），理由见 §4。

---

## 3. 上游语义（已通读源码核准，非推测）

### 3.1 样本布局

`dataloader.py:123` —— `nn_inputs[..., :9]` = `ray_o(3) | ray_d(3) | tx(3)`。
`tx` 按**每张谱图**取一个值再 tile 到该图全部射线上（`:122`）：

```python
tx_pos_i = torch.tile(tx_pos[int(idx)-1], (self.rays_per_spectrum,)).reshape(-1, 3)
```

⇒ 本数据集是单 Tx，故 `tx` 是**全局常量**。见 §4.2 的可辨识性论证。

### 3.2 标签

`dataloader.py:119` —— `imageio.imread(png) / 255` ⇒ 标签 ∈ [0, 1]。
**上游 dataloader 不做任何逐图归一化**。原生谱图规格 `90 × 360`
（`nerf2_runner.py:203` 的 `reshape(90, 360)`）＝ (俯仰 90, 方位 360)。

> 这条对我们重要：它说明若我们的靶子呈逐图满量程，那**不是 NeRF² 引入的**，
> 而是来自 RF-3DGS 的生成侧（见 `说明.md` 边界「逐图满量程混杂」）。

### 3.3 损失与优化

`nerf2_runner.py:147` —— **`loss = sig2mse(predict_spectrum, train_label.view(-1))`**，
而 `sig2mse = lambda x, y: torch.mean((x - y) ** 2)`（`model.py:12`）
⇒ **上游 spectrum 用的是纯 MSE，不是 L1+SSIM**。

`:56/:59` —— `Adam(lr, weight_decay) + CosineAnnealingLR(T_max, eta_min)`。

### 3.4 体渲染核心（`renderer.py:95-136` `Renderer_spectrum.raw2outputs`）

```
dists      = 相邻采样点间距，末项置 1e10，再乘 ‖rays_d‖
att_p, s_p = sigmoid(x)·2π                      （相位，∈[0,2π)）
att_a, s_a = |leaky_relu(x)|                    （幅度，非负）
alpha      = 1 − exp(−att_a·dists)
att_i      = cumprod([1, 1−alpha+1e-10])[:-1]   （透射率，沿射线累积）
phase_i    = exp(1j · cumsum([1, phase])[:-1])
path_loss  = 0.025 / path
receive    = |∑_n  s_a · exp(1j·s_p) · att_i · phase_i · path_loss|     ← 标量
```

**关键：返回值是 `abs(复数求和)`，即一个非负、无界的标量**，不是 [0,1] 区间量。
上游靠 `path_loss = 0.025/path` 的尺度让它自然落在 [0,1] 附近，再用 `/255` 的标签
配 MSE 去压。**没有任何 `sigmoid` 在它之后。**

> 这条推翻了本项目方案初稿里「末层 `Linear(W,1)` + `sigmoid`」的描述。
> 正确做法见 §4.3。

### 3.5 输出头是完全参数化的（**上游 `model.py` 无需任何改动**）

`model.py:100` 的构造签名：

```python
def __init__(self, D=8, W=256, skips=[4], input_dims=..., multires=...,
             is_embeded=..., attn_output_dims=2, sig_output_dims=2):
```

`self.attenuation_output = nn.Linear(W, attn_output_dims)`、
`self.signal_output = nn.Linear(W//2, sig_output_dims)`。
⇒ **改通道数只需换构造参数**，两个臂共用同一份 `model.py` 逐字源码。

---

## 4. 我们声明的偏离（全部为**已声明**，且尽量作废于两臂相减）

### 4.1 数据协议不同（不可消解 —— 决定了数字只能内部比）

上游是「单网关 + 90×360 极坐标谱图」；我们这份数据是
**800 个 Rx 位置 × 4 个朝向 × 300×200 针孔图**（HFOV=90°、VFOV=67.38°）。
**本项目的一切读数都不可与 NeRF² 论文表格对比**，只能与自家 A1/A2/A3 对比。

### 4.2 `tx` 固定为常量（不可辨识，故不设为可学习）

单 Tx ⇒ `tx ≡ c`。无论是否位置编码，它在信号网首层只贡献
`W_t · PE(c) + b1`，**被 bias 完全吸收**，函数类一字不变。
⇒ `tx` 不可辨识、不提供任何容量。

保留该输入只为一件事：**忠实上游签名**，并为将来多 Tx 留口。
**不得**把任何「反解出的 Tx 位置」当物理结论报告。

### 4.3 输出参数化：两臂共用同一个末端映射，只差通道数

上游把非负无界的 `abs(...)` 直接当靶子量。我们的 GT 是 [0,1]，且**逐图满量程**
（见 §3.2），不存在一个固定的仿射能把渲染标量映到靶子 —— 所以必须有可学的末端映射。

设计成**两臂共用同一形式**，于是它在 Δ(N2 − N1) 里自动抵消：

```
N1（RGB 头）  : sig_output_dims = 6   ⇒ 渲染器返回 [B, 3]
N2（标量头）  : sig_output_dims = 2   ⇒ 渲染器返回 [B]，与上游逐字同
两臂共用      : v̂ = σ(α·v + β)         （α, β 为可学习逐通道标量）
N2 额外       : rgb = apply_jet(v̂)     （固定的 jet 查表，无参数）
```

`σ` 之后的语义：N1 的 3 个通道就是 RGB；N2 的 1 个通道是色标坐标 u ∈ (0,1)。

这样 N1 与 N2 **唯一**的差别是输出参数化 ⇒ Δ(N2 − N1) 干净地只反映这一件事。
数值等价性（G5b）：**N2 在 `σ` 之前的渲染器输出，必须与上游
`Renderer_spectrum.render_ss` 逐位一致。**

### 4.4 损失用 L1 + 0.2·(1 − SSIM)，不用上游的 MSE

上游是纯 MSE（§3.3）。我们改用 L1 + 0.2·(1−SSIM)，理由：**与 A1/A2/A3 逐字相同**。
基线若换损失函数，对比里就混进了第二个自变量。这是一次**有意识的偏离**，
不是疏漏。两臂同损失 ⇒ 在 Δ 里抵消。

（`--loss mse` 开关保留，可一键回到上游口径，但本轮不跑。）

### 4.5 `near` / `far` 按本数据集实算，不抄上游 `far: 5`

上游 `render: {near: 0, far: 5}` 是给几十米室内场景的。
本数据集场景最大点距 **15.468 m**，`far=5` 会截掉过半场景。
实算 `far = 1.05 × max_v ‖C_v − tx‖`（≈ 8.7 m）、`near = 0.10`（上游取 0，
即首个采样点落在相机位置上；我们留一点余量避开原点退化）。
实算值写进每个节点的 `nerf2_cfg.json`。

### 4.6 不 import `skimage`

上游 `eval_network_spectrum` 的 `skimage.ssim(..., multichannel=)` 在 skimage ≥ 0.26
已删参，直接跑 `TypeError`。我们的评测**只用**本仓
`splat.utils.loss_utils.ssim` / `splat.utils.image_utils.psnr` / `splat.lpipsPyTorch.lpips`
—— 与 A1/A2/A3 同一把尺子。

---

## 5. 硬不变量（违反会**静默**出错）

### 5.1 `CUDA_VISIBLE_DEVICES` 必须只暴露一张卡

`renderer.py` 的 `.cuda()` 出现处：

| 行 | 所在函数 | 是否在我们执行路径上 |
|---|---|---|
| 114, 125, 126, 129 | `Renderer_spectrum.raw2outputs` | **是** |
| 172 | `Renderer_RSSI.render_rssi` | 否 |
| 207 | `Renderer_RSSI.raw2outputs_signal` | 否 |
| 292 | `Renderer_CSI.raw2outputs_signal` | 否 |

全部**不带设备号** ⇒ 恒取 cuda:0。若可见多卡，cuda:0 就是**物理 0 号卡**。
本机 0 号卡被其它用户作业满载，后果是 device mismatch 崩溃、或抢卡把别人的作业 OOM。

**做法**：`CUDA_VISIBLE_DEVICES=<n>` 只列一张卡，则 cuda:0 ≡ 那张卡，`.cuda()` 自动正确，
**vendored 文件保持逐字**。断言入口 `_compat.assert_single_visible_device()`，
由驱动脚本与 G7 双重调用。

### 5.2 `--eval` 对频谱集无效

`splat/scene/dataset_readers.py:271-277` 的 llffhold 分支**被注释掉** ⇒ 划分
100% 由 `train_index.txt`(2560) / `test_index.txt`(640) 决定。
数据适配层**不暴露任何划分开关**，以免误用。

---

## 6. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-16 | 首次 vendor。三文件 sha256 与上游一致；核准 `is_embeded.tx=True`、损失为 MSE、`abs()` 输出无 sigmoid、输出头可参数化 |
