"""Export MVSplat predicted Gaussians to standard 3DGS .ply files.

Usage (from the repo root, e.g. inside WSL):
    NUM_PLY_SCENES=5 python export_ply_scenes.py mode=test
    NUM_PLY_SCENES=20 python export_ply_scenes.py mode=test

The .ply files (3DGS format: xyz / f_dc / opacity / scale / rot) can be dragged
into https://supersplat.com or https://antimatter15.com/splat/ to view.
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os
from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from torch import Tensor
from torch.utils.data import default_collate

with install_import_hook(("feedforward",), ("beartype", "beartype")):
    from feedforward.config import load_typed_root_config
    from feedforward.dataset import get_dataset
    from feedforward.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from feedforward.global_cfg import set_cfg
    from feedforward.misc.wandb_tools import update_checkpoint_path
    from feedforward.model.decoder import get_decoder
    from feedforward.model.encoder import get_encoder
    from feedforward.model.model_wrapper import ModelWrapper
    from feedforward.model.ply_export import export_ply


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_dict) -> None:
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    device = torch.device("cuda:0")

    # Load the pretrained weights (encoder is a submodule of ModelWrapper).
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model_wrapper = ModelWrapper.load_from_checkpoint(
        checkpoint_path,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        losses=[],
        step_tracker=None,
    )
    model_wrapper.eval().to(device)

    # Figure out which scenes to export.
    pred_dir = Path("outputs/test/re10k")
    all_scenes = sorted(p.stem for p in pred_dir.iterdir() if p.is_dir())
    limit = int(os.environ.get("NUM_PLY_SCENES", "5"))
    scenes = all_scenes[:limit]
    print(f"exporting {len(scenes)} scenes: {scenes[:5]}{'...' if len(scenes) > 5 else ''}")

    # Number of frames per scene (to pick valid context views).
    scene_frames = {}
    for chunk_path in sorted(Path("datasets/re10k/test").glob("*.torch")):
        for item in torch.load(chunk_path, map_location="cpu"):
            scene_frames[item["key"]] = len(item["images"])

    out_dir = Path("outputs/test/ply")
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for scene in scenes:
        n = scene_frames.get(scene)
        if n is None:
            print(f"skip {scene}: not in downloaded chunks")
            continue
        if n < 2:
            print(f"skip {scene}: only {n} frames")
            continue

        # Two context views with a reasonable baseline.
        context_views = [0, min(40, n - 1)]
        view_sampler_cfg = ViewSamplerArbitraryCfg(
            "arbitrary", 2, 2, context_views=context_views, target_views=None
        )
        cfg.dataset.view_sampler = view_sampler_cfg
        cfg.dataset.overfit_to_scene = scene

        dataset = get_dataset(cfg.dataset, "test", None)
        example = default_collate([next(iter(dataset))])
        example = apply_to_collection(example, Tensor, lambda x: x.to(device))

        # Run the encoder; it also fills in scales/rotations for the PLY export.
        visualization_dump = {}
        gaussians = encoder.forward(
            example["context"],
            model_wrapper.global_step,
            False,
            visualization_dump=visualization_dump,
        )

        path = out_dir / f"{scene}.ply"
        export_ply(
            example["context"]["extrinsics"][0, 0],
            gaussians.means[0],
            visualization_dump["scales"][0],
            visualization_dump["rotations"][0],
            gaussians.harmonics[0],
            gaussians.opacities[0],
            path,
        )
        print(f"  exported {scene}.ply  ({gaussians.means[0].shape[0]} gaussians, context {context_views})")
        exported += 1

    print(f"DONE: {exported} .ply files -> {out_dir}")


if __name__ == "__main__":
    main()
