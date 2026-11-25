# -*- coding: utf-8 -*-
"""nerf2 —— NeRF² 基线（vendor 自 XPengZhao/NeRF2, MIT）

模块分工
========
    model.py        上游逐字。`NeRF2` 网络（衰减网 + 信号网 + 两个输出头）
    renderer.py     上游逐字。`Renderer_spectrum` 等三个体渲染器
    LICENSE         上游 MIT 全文（Copyright (c) 2023 ZHAO Xiaopeng）
    UPSTREAM.md     来源、sha256、上游参照配置、以及**我们声明的全部偏离**
    _compat.py      垫片：上游文件的**子类化**扩展 + 设备不变量断言

上游文件一律**只读**。任何需要改上游行为的地方，一律走 `_compat.py` 的子类，
这样 `sha256` 永远对得上，也永远能一句话回答「你改过上游没有」。

两个硬不变量（违反会静默出错，不报错）
======================================
1. **`CUDA_VISIBLE_DEVICES` 必须只暴露一张卡**。上游 `renderer.py` 有 7 处不带
   设备号的 `.cuda()`，其中 `raw2outputs` 里 4 处在我们执行的路径上。若可见多张卡，
   `.cuda()` 恒取 cuda:0，也就是**物理 0 号卡**，而不是我们要用的那张 ——
   轻则 device mismatch 崩，重则抢别人的卡把别人的作业 OOM 掉。
   断言入口：`_compat.assert_single_visible_device()`。
2. **`near`/`far` 不许抄上游配置的 `far: 5`**。上游场景是几十米的室内，
   我们这份数据的场景最大点距 **15.468 m**；`far=5` 会截掉过半场景。
   实算值由 `data_adapter` 给出并写进 `nerf2_cfg.json`。
"""
