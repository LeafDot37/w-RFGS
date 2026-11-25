"""End-to-end render check: MVSplat renders an RF target view; compare to the real image.

If the pose convention / context construction is right, the rendered 256x256
target should match the real (crop-shimmed) image reasonably (PSNR well above
garbage). This tests the full pipeline the same way init_cloud.py uses it.

Usage (WSL, cwd = code/ 单 import 根):
    python rf_render_check.py mode=test +experiment=re10k \
        checkpointing.load=/mnt/d/GS3D/mvsplat/checkpoints/re10k.ckpt
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from jaxtyping import install_import_hook
from PIL import Image

with install_import_hook(("feedforward",), ("beartype", "beartype")):
    from feedforward.config import load_typed_root_config
    from feedforward.dataset.shims.crop_shim import rescale_and_crop
    from feedforward.global_cfg import set_cfg
    from feedforward.misc.wandb_tools import update_checkpoint_path
    from feedforward.model.decoder import get_decoder
    from feedforward.model.encoder import get_encoder
    from feedforward.model.model_wrapper import ModelWrapper

_ROOT = Path(__file__).resolve().parents[2]           # 仓库根（数据在根下 datasets/RF-3DGS_dataset/）
SRC = _ROOT / "datasets/RF-3DGS_dataset/blender_visual_dataset"
OUT = Path("outputs/rf_init")                          # cwd=code/ 时即 code/outputs/rf_init
# South-facing row at y=0.5, all looking south (guaranteed overlap).
# CTX/TGT/NUM_VIEWS can be overridden via env for experiments.
def env_int(name, default):
    import os
    return int(os.environ.get(name, default))
CTX = [int(x) for x in os.environ.get("RC_CTX", "0 6 12 18").split()]
TGT = [int(x) for x in os.environ.get("RC_TGT", "9").split()]
NUM_VIEWS = env_int("RC_NUM_VIEWS", len(CTX))
ALL = CTX + TGT


def load_composite(path):
    im = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0
    rgb, alpha = im[..., :3], im[..., 3:4]
    out = rgb * alpha + np.array([0.0, 0.0, 0.0], np.float32) * (1.0 - alpha)
    return torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1)))


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_dict) -> None:
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    frames = json.load(open(SRC / "transforms_train.json"))["frames"]
    c2w = np.stack([np.asarray(f["transform_matrix"], np.float32) for f in frames])
    c2w[:, :3, 1:3] *= -1

    views = ALL
    imgs = torch.stack([load_composite(SRC / frames[i]["file_path"]) for i in views])
    # Replicate re10k's exact input convention: NORMALIZED intrinsics
    # (fractions of image dimensions) + crop shim to 256x256.
    #   fx=800/1600=0.5, fy=800/900=0.889, cx=800/1600=0.5, cy=450/900=0.5
    intrins = torch.zeros(len(views), 3, 3)
    intrins[:, 0, 0] = 800.0 / 1600.0
    intrins[:, 1, 1] = 800.0 / 900.0
    intrins[:, 0, 2], intrins[:, 1, 2], intrins[:, 2, 2] = 0.5, 0.5, 1.0
    imgs, intrins = rescale_and_crop(imgs, intrins, (256, 256))

    n_ctx = len(CTX)
    context = {
        "image": imgs[:n_ctx].unsqueeze(0).to(device),
        "intrinsics": intrins[:n_ctx].unsqueeze(0).to(device),
        "extrinsics": torch.from_numpy(c2w[CTX]).unsqueeze(0).to(device),
        "near": torch.full((1, n_ctx), 1.0).to(device),
        "far": torch.full((1, n_ctx), 100.0).to(device),
    }
    target = {
        "extrinsics": torch.from_numpy(c2w[ALL]).unsqueeze(0).to(device),
        "intrinsics": intrins.unsqueeze(0).to(device),
        "near": torch.full((1, len(ALL)), 1.0).to(device),
        "far": torch.full((1, len(ALL)), 100.0).to(device),
    }

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    if NUM_VIEWS != 2:
        # Rebuild the encoder with the requested context-view count so the
        # cross-view attention fuses ALL views (the pretrained weights are
        # num_views-independent -- only the reshape pattern changes).
        cfg_dict["dataset"]["view_sampler"]["num_context_views"] = NUM_VIEWS
        cfg = load_typed_root_config(cfg_dict)
        set_cfg(cfg_dict)
        print(f"rebuilt encoder with num_context_views={NUM_VIEWS}")
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model_wrapper = ModelWrapper.load_from_checkpoint(
        checkpoint_path,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=None,
        decoder=decoder,
        losses=[],
        step_tracker=None,
    )
    model_wrapper.eval().to(device)

    with torch.no_grad():
        gaussians = encoder.forward(context, model_wrapper.global_step, True)
        out = decoder.forward(
            gaussians, target["extrinsics"], target["intrinsics"],
            target["near"], target["far"], (256, 256),
        )
    OUT.mkdir(exist_ok=True)
    for j, vi in enumerate(ALL):
        render = out.color[0, j].permute(1, 2, 0).float().cpu().numpy().clip(0, 1)
        real = imgs[j].permute(1, 2, 0).float().cpu().numpy()
        mse = ((render - real) ** 2).mean()
        psnr = 10 * np.log10(1.0 / (mse + 1e-8))
        tag = "CTX" if vi in CTX else "TGT"
        print(f"  [{tag}] frame {vi:3d}: PSNR = {psnr:6.2f} dB  (MSE {mse:.4f})")
        Image.fromarray((render * 255).astype(np.uint8)).save(
            OUT / f"render_f{vi}.png")
        Image.fromarray((real * 255).astype(np.uint8)).save(
            OUT / f"real_f{vi}.png")
    print(f"saved render_f*.png (MVSplat) / real_f*.png (real, 256x256) -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
