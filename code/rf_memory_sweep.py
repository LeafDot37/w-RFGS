"""Measure MVSplat encoder GPU memory vs number of context views (inference, no_grad).

Usage (from repo root, e.g. inside WSL):
    python rf_memory_sweep.py mode=test +experiment=re10k \
        checkpointing.load=checkpoints/re10k.ckpt dataset/view_sampler=arbitrary

Memory depends on the number of context views v, the operating resolution (256x256
after the crop shim) and the model config -- NOT on the source image size/content,
so the numbers transfer directly to the RF-3DGS visual data (which is also fed as
256x256). We sweep v = 2..16 on the already-downloaded re10k chunks.
"""
import hydra
import torch
from pathlib import Path
from torch import Tensor
from torch.utils.data import default_collate
from jaxtyping import install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection

with install_import_hook(("feedforward",), ("beartype", "beartype")):
    from feedforward.config import load_typed_root_config
    from feedforward.dataset import get_dataset
    from feedforward.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from feedforward.global_cfg import set_cfg
    from feedforward.misc.wandb_tools import update_checkpoint_path
    from feedforward.model.decoder import get_decoder
    from feedforward.model.encoder import get_encoder
    from feedforward.model.model_wrapper import ModelWrapper


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_dict) -> None:
    device = torch.device("cuda:0")

    # Pick the scene with the most frames among the downloaded chunks.
    scene_frames = {}
    for chunk_path in sorted(Path("datasets/re10k/test").glob("*.torch")):
        for item in torch.load(chunk_path, map_location="cpu"):
            scene_frames[item["key"]] = len(item["images"])
    scene = max(scene_frames, key=scene_frames.get)
    nframes = scene_frames[scene]
    print(f"scene={scene}  frames={nframes}  (memory is content-independent)")

    sweeps = [2, 3, 4, 6, 8, 10, 12, 16]
    for N in sweeps:
        if N > nframes:
            print(f"N={N:2d}: skipped (scene only has {nframes} frames)")
            continue
        try:
            # num_context_views sets the encoder UNets' n_frames (reshape-only, no params).
            cfg_dict["dataset"]["view_sampler"]["num_context_views"] = N
            cfg = load_typed_root_config(cfg_dict)
            set_cfg(cfg_dict)

            encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
            decoder = get_decoder(cfg.model.decoder, cfg.dataset)
            checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
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

            # Spread the N context views over the trajectory.
            ctx = sorted({round(i * (nframes - 1) / (N - 1)) for i in range(N)})
            cfg.dataset.view_sampler = ViewSamplerArbitraryCfg(
                "arbitrary", N, 1, context_views=ctx, target_views=[0]
            )
            cfg.dataset.overfit_to_scene = scene
            dataset = get_dataset(cfg.dataset, "test", None)
            example = default_collate([next(iter(dataset))])
            example = apply_to_collection(example, Tensor, lambda x: x.to(device))

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            with torch.no_grad():
                gaussians = encoder.forward(
                    example["context"], model_wrapper.global_step, True
                )
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            print(f"N={N:2d}: gaussians={gaussians.means.shape[1]:7d}  "
                  f"peak_mem={peak:.2f} GiB")

            del gaussians, example, model_wrapper, encoder, decoder, dataset
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"N={N:2d}: OOM (out of memory)")
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"N={N:2d}: ERROR {type(e).__name__}: {str(e)[:140]}")


if __name__ == "__main__":
    main()
