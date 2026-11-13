"""MVSplat -> 3DGS init cloud for the RF-3DGS visual dataset (方案一).

Runs the pretrained MVSplat encoder on K spatial clusters of the 809-camera
trajectory, each cluster using V evenly-spread views, then unions the per-cluster
Gaussian clouds, filters by opacity, voxel-downsamples to a target size, and
writes a standard 3DGS .ply in the *world frame of the RF visual poses* (no
re-centering -- the cloud must stay aligned with the poses RF-3DGS stage-1 will
optimise).

The two key numbers are hyperparameters (see "usage" below):
  RF_NUM_CLUSTERS        K (default 8). Larger scene / longer trajectory -> more.
  RF_VIEWS_PER_CLUSTER   V (default 8; RECOMMEND 4 for large scenes). Number of
                         MVSplat context views per forward. Peak VRAM ~0.66 GiB
                         per view (RTX 4060 8GB measured), so V <= ~12 on 8GB.
                         WARNING: V=8 degrades reconstruction vs V=4 (the re10k
                         attention is trained at num_views=2; measured context
                         PSNR drops ~24-30 dB at V=4 -> ~12-21 dB at V=8).
                         Large scene -> raise K and keep V in 2..4.
  RF_GAZES_PER_CLUSTER   G (default 1). Gaze directions run per cluster. Each
                         position cluster only reconstructs the direction(s) it
                         looks along; G=2 fills holes where some cameras view a
                         region along a direction no single-gaze cluster covered.

Rule of thumb: the cloud must *cover* the whole scene, and every camera belongs
to exactly one cluster. Raising K adds forwards (more coverage of a big space);
raising G adds directions per region (fills direction holes); raising V adds
baseline diversity within one forward but hurts quality past ~4 (see above).
The union is redundant by design -- the 3DGS fit that follows prunes/densifies
it anyway.

Other knobs:
  RF_MAX_GAUSSIANS       target cloud size after voxel downsampling (def 500_000)
  RF_VOXEL_SIZE          override voxel size (0 = auto from max_gaussians)
  RF_OPACITY_THRESHOLD   drop Gaussians with opacity < this (def 0.05)
  RF_MAX_DEPTH           drop Gaussians farther than this from their own camera
                         (def 15; 0 = off). The black void makes MVSplat emit a
                         far-plane shell around depth~far=100 -- cut it here.
  RF_CLUSTERS_LIMIT      process only the first N clusters (0 = all; debug)
  RF_BACKGROUND          black|white composite bg (def black: matches RF-3DGS
                         white_background=False, so init == training images)
  RF_NEAR / RF_FAR       depth bounds (def 1 / 100, same as re10k training)
  RF_SOURCE_PATH         path to blender_visual_dataset
  RF_NUM_CLUSTERS=0      auto: K = round(809 / RF_TARGET_FRAMES_PER_CLUSTER)

This dataset is ground stations x 8 outward gazes, so each position cluster
keeps only the V views sharing its dominant gaze (a forward-facing rig) --
mixed-direction views cannot be fused by MVSplat (scattered depth sheets).

Correctness notes (all three were bugs, now fixed & verified by render check):
  * intrinsics are NORMALIZED (fx/W, fy/H, cx/W, cy/H) like re10k -- raw-pixel
    K puts the model ~1700x out of distribution (garbage + tiny scales).
  * the encoder is rebuilt with num_context_views=V so all V views fuse (the
    re10k checkpoint is hard-wired to 2, silently splitting V=8 into 4 pairs).
  * views are clustered by position + gaze so a cluster overlaps.

Usage (cwd = code/ 单 import 根；例如 WSL 内；K=8/V=8 峰值 ~4.5 GiB):
    cd /path/to/w-RFGS/code
    RF_SOURCE_PATH=/path/to/w-RFGS/datasets/RF-3DGS_dataset/blender_visual_dataset \
    python init_cloud.py mode=test +experiment=re10k \
        checkpointing.load=/path/to/w-RFGS/checkpoints/re10k.ckpt

    # 端到端（仓库根 run_end_to_end.sh 阶段 A）即按此运行：cwd=code/ 使
    # hydra 日志写 code/outputs/<时间>/，init 点云写 code/outputs/rf_init/，
    # 随后由驱动归档到 results/mvsplat_clouds/（latest + archive/），并以
    # RF_POINTCLOUD 指向本次 run 的新云副本供 RF-3DGS 阶段一读取。

Output: outputs/rf_init/init.ply       (simple RGB ply -- RF-3DGS fetchPly input)
        outputs/rf_init/init_view.ply  (standard 3DGS -- drag into supersplat)
        outputs/rf_init/clusters.json  (cluster/view selection + stats)
        # 路径相对 cwd；从 code/ 下运行即 code/outputs/rf_init/
"""
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from jaxtyping import install_import_hook
from PIL import Image
from plyfile import PlyData, PlyElement

with install_import_hook(("feedforward",), ("beartype", "beartype")):
    from feedforward.config import load_typed_root_config
    from feedforward.dataset.shims.crop_shim import rescale_and_crop
    from feedforward.global_cfg import set_cfg
    from feedforward.misc.wandb_tools import update_checkpoint_path
    from feedforward.model.decoder import get_decoder
    from feedforward.model.encoder import get_encoder
    from feedforward.model.model_wrapper import ModelWrapper
    from feedforward.model.ply_export import construct_list_of_attributes


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def kmeans(points: np.ndarray, k: int, seed: int = 0, iters: int = 60):
    """Deterministic k-means on 3-D positions (k-means++ init). Returns (labels, centers)."""
    rng = np.random.default_rng(seed)
    centers = [points[rng.integers(len(points))]]
    for _ in range(k - 1):
        d2 = np.stack([((points - c) ** 2).sum(1) for c in centers]).min(0)
        probs = d2 / d2.sum()
        centers.append(points[rng.choice(len(points), p=probs)])
    centers = np.stack(centers)
    for _ in range(iters):
        labels = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
        for i in range(k):
            m = labels == i
            if m.any():
                centers[i] = points[m].mean(0)
    return labels, centers


def fps(points: np.ndarray, n: int) -> np.ndarray:
    """Farthest-point sampling of n indices from points (spreads views out)."""
    idx = [0]
    d = np.full(len(points), np.inf)
    while len(idx) < n and len(idx) < len(points):
        d = np.minimum(d, ((points - points[idx[-1]]) ** 2).sum(1))
        idx.append(int(d.argmax()))
    return np.array(idx)


def quantize_gaze(look: np.ndarray) -> np.ndarray:
    """Map horizontal gaze vectors to 8 compass-sector indices (0..7)."""
    ang = np.arctan2(look[:, 1], look[:, 0])      # -pi..pi, atan2(y, x)
    return np.round(ang / (np.pi / 4)).astype(int) % 8


def load_composite(path: Path, bg: np.ndarray) -> torch.Tensor:
    """Load an RGBA TIF and composite onto the background -> float [0,1] RGB [3,H,W].

    Same math as RF-3DGS scene/dataset_readers.readCamerasFromTransforms.
    """
    im = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0
    rgb, alpha = im[..., :3], im[..., 3:4]
    out = rgb * alpha + bg * (1.0 - alpha)
    return torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1)))


def voxel_downsample(points: np.ndarray, voxel: float, attrs: list):
    """Average points (and each attribute) per occupied voxel cell. 1-D attrs OK."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    cells = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=cells)
    sums = np.zeros((cells, points.shape[1]))
    np.add.at(sums, inv, points)
    new_pts = sums / counts[:, None]
    new_attrs = []
    for a in attrs:
        is_1d = a.ndim == 1
        a = a[:, None] if is_1d else a
        s = np.zeros((cells, a.shape[1]))
        np.add.at(s, inv, a)
        out = s / counts[:, None]
        new_attrs.append(out[:, 0] if is_1d else out)
    return new_pts, new_attrs, counts


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    """SH DC coefficients -> displayable uint8 RGB (same C0 supersplat uses)."""
    c0 = 0.28209479177387814
    return np.clip(f_dc * c0 + 0.5, 0.0, 1.0)


def export_ply_simple(means: np.ndarray, rgb_u8: np.ndarray, path: Path):
    """Simple x/y/z/nx/ny/nz/red/green/blue ply -- what RF-3DGS fetchPly reads.

    This is the init-cloud format: only positions + colors are used by
    create_from_pcd (scales/rots/opacity are recomputed there).
    """
    n = means.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(n, dtype=dtype)
    attrs = np.concatenate([means, np.zeros_like(means), rgb_u8], axis=1)
    elements[:] = list(map(tuple, attrs))
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def export_ply_3dgs(means, f_dc, opacities, scales, rotations, path: Path):
    """Standard 3DGS .ply in world coords (no recentering) -- for supersplat view."""
    n = means.shape[0]
    dtype_full = [(a, "f4") for a in construct_list_of_attributes(0)]
    elements = np.empty(n, dtype=dtype_full)
    rot_wxyz = np.roll(rotations, 1, axis=1)  # model [x,y,z,w] -> ply [w,x,y,z]
    attrs = np.concatenate(
        [
            means,
            np.zeros_like(means),  # normals
            f_dc,                  # SH DC band
            opacities[:, None],
            np.log(np.clip(scales, 1e-6, None)),
            rot_wxyz,
        ],
        axis=1,
    )
    elements[:] = list(map(tuple, attrs))
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def env_float(name, default):
    return float(os.environ.get(name, default))


def env_int(name, default):
    return int(os.environ.get(name, default))


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_dict) -> None:
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    # --- hyperparameters --------------------------------------------------
    K = env_int("RF_NUM_CLUSTERS", 8)               # number of spatial clusters
    V = env_int("RF_VIEWS_PER_CLUSTER", 8)          # MVSplat context views per cluster
    G = env_int("RF_GAZES_PER_CLUSTER", 1)          # gaze directions run per cluster
    target = env_int("RF_MAX_GAUSSIANS", 500_000)
    voxel_override = env_float("RF_VOXEL_SIZE", 0.0)
    opac_th = env_float("RF_OPACITY_THRESHOLD", 0.05)
    max_depth = env_float("RF_MAX_DEPTH", 15.0)  # 0 = keep all depths
    limit = env_int("RF_CLUSTERS_LIMIT", 0)      # 0 = all clusters (debug)
    bg_name = os.environ.get("RF_BACKGROUND", "black").lower()
    near, far = env_float("RF_NEAR", 1.0), env_float("RF_FAR", 100.0)
    src = Path(os.environ.get(
        "RF_SOURCE_PATH", "datasets/RF-3DGS_dataset/blender_visual_dataset"
    ))
    out_dir = Path("outputs/rf_init")

    bg = (np.array([1.0, 1.0, 1.0], np.float32) if bg_name == "white"
          else np.array([0.0, 0.0, 0.0], np.float32))
    assert src.exists(), f"source path not found: {src.resolve()}"

    if V > 12:
        print(f"[warn] V={V} context views -> peak VRAM ~{0.66 * V:.1f} GiB; "
              f"8 GB cards cap out around V=12 (OOM beyond).")

    # --- poses -------------------------------------------------------------
    with (src / "transforms_train.json").open() as f:
        frames = json.load(f)["frames"]
    n = len(frames)
    c2w = np.stack([np.asarray(fr["transform_matrix"], np.float32) for fr in frames])
    c2w[:, :3, 1:3] *= -1                       # Blender -> MVSplat c2w convention
    paths = [src / fr["file_path"] for fr in frames]
    pos = c2w[:, :3, 3]                         # camera centres
    look = c2w[:, :3, 2]                        # camera +Z (gaze) in world
    print(f"loaded {n} visual poses from {src.name}")

    # sanity: camera-to-content depth should lie within [near, far]
    step = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    print(f"median step {step:.3f} units/frame, total path "
          f"{np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.1f}")

    if K <= 0:                                   # auto cluster count
        per = env_int("RF_TARGET_FRAMES_PER_CLUSTER", 100)
        K = max(1, int(round(n / per)))
        print(f"auto K = round({n}/{per}) = {K}")

    # This dataset is 105 ground stations x 8 outward gazes (N/NE/../NW):
    # co-located frames point in every direction. So views must be grouped by
    # gaze OR they don't overlap and MVSplat emits per-view depth sheets
    # ("lines of dots"). Strategy: cluster stations by POSITION, then inside a
    # cluster select the V views that all share the cluster's dominant gaze
    # (a forward-facing rig). RF_GAZE_WEIGHT no longer used for clustering.
    labels, centers = kmeans(pos, K, seed=0)
    sizes, spans = [], []
    for i in range(K):
        m = labels == i
        sizes.append(int(m.sum()))
        spans.append(float(np.ptp(pos[m], axis=0).max()))
        print(f"  cluster {i:2d}: {sizes[-1]:3d} frames, max span {spans[-1]:.2f}")

    # --- model --------------------------------------------------------------
    print(f"loading checkpoint: {cfg.checkpointing.load}")
    if V != 2:
        # Rebuild the encoder with V context views so the cross-view attention
        # actually fuses ALL V views. The pretrained re10k model is hard-wired
        # to num_views=2 at construction; with V=8 it silently ran 4 independent
        # 2-view pairs (8 % 2 == 0), and odd V crashed. The attention weights
        # are num_views-independent (only the rearrange pattern changes), so the
        # same checkpoint loads fine -- verified: num_views=4 reproduces context
        # views at ~25-30 dB PSNR.
        cfg_dict["dataset"]["view_sampler"]["num_context_views"] = V
        cfg = load_typed_root_config(cfg_dict)
        set_cfg(cfg_dict)
        print(f"rebuilt encoder with num_context_views={V}")
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
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
    step = model_wrapper.global_step
    print(f"peak VRAM estimate for this run: ~{0.66 * V:.1f} GiB (V={V})")

    fx, fy, cx, cy = 800.0, 800.0, 800.0, 450.0   # camera_angle_x = pi/2 @1600x900
    W, H = 1600.0, 900.0

    # --- per-cluster MVSplat forward -----------------------------------------
    # Each cluster runs forwards on its top-G gaze directions (G=1 -> the old
    # single dominant gaze; G=2 fills holes where a camera looks along a
    # direction no cluster reconstructs). The union is redundant by design.
    means, opac, colors, scales, rots, depths = [], [], [], [], [], []
    sel_info = {}
    n_forward = 0
    for i in range(K):
        if limit and i >= limit:
            break
        idx = np.where(labels == i)[0]
        # gaze sectors in this position cluster, most populous first
        k = quantize_gaze(look[idx])
        counts = np.bincount(k, minlength=8)
        order = np.argsort(-counts)
        sectors = [int(s) for s in order if counts[s] >= 2][:G]
        if not sectors:
            print(f"  [skip] cluster {i}: no gaze sector with >= 2 frames")
            continue
        for g in sectors:
            pool = idx[k == g]
            views = pool[fps(pos[pool], min(V, len(pool)))]
            if len(views) < V:
                # a num_views=V forward requires len(views) % 2 == 0 (the
                # encoder rearrange splits the batch into 2-view pairs); skip
                # rather than crash. Coverage loss is fine -- the union is
                # redundant and K is chosen so pools comfortably exceed V.
                print(f"  [skip] cluster {i} gaze {g}: pool {len(pool)} < V={V}")
                continue
            n_forward += 1
            sel_info[f"{i}:g{g}"] = [int(v) for v in views]

            imgs = torch.stack([load_composite(paths[v], bg) for v in views])  # [v,3,900,1600]
            # NORMALIZED intrinsics (fractions of image dims), exactly like re10k:
            # the model was trained on fx~0.84, cx=cy=0.5. Feeding raw pixels
            # (fx=800) put K ~1700x out of distribution -> garbage cost volume and
            # ~1e-7 gaussian scales (the "tiny scales" symptom). After the crop shim
            # the normalized K becomes the true K of the 256x256 crop, so output
            # gaussian positions stay in the true world frame.
            intrins = torch.zeros(len(views), 3, 3)
            intrins[:, 0, 0] = fx / W          # 0.5
            intrins[:, 1, 1] = fy / H          # 0.889
            intrins[:, 0, 2] = cx / W          # 0.5
            intrins[:, 1, 2] = cy / H          # 0.5
            intrins[:, 2, 2] = 1.0
            imgs, intrins = rescale_and_crop(imgs, intrins, (256, 256))

            context = {
                "image": imgs.unsqueeze(0).to(device),           # [1,v,3,256,256]
                "intrinsics": intrins.unsqueeze(0).to(device),   # [1,v,3,3]
                "extrinsics": torch.from_numpy(c2w[views]).unsqueeze(0).to(device),
                "near": torch.full((1, len(views)), near).to(device),
                "far": torch.full((1, len(views)), far).to(device),
            }
            with torch.no_grad():
                dump = {}
                gs = encoder.forward(context, step, False, visualization_dump=dump)

            gmeans = gs.means[0].detach().cpu().numpy()
            n_per_view = gmeans.shape[0] // len(views)
            # each gaussian was emitted by one of the V views, in view-major order
            view_idx = np.repeat(np.arange(len(views)), n_per_view)
            cam_centers = c2w[views][:, :3, 3]
            gdepth = np.linalg.norm(gmeans - cam_centers[view_idx], axis=1)

            means.append(gmeans)
            opac.append(gs.opacities[0].detach().cpu().numpy())
            colors.append(gs.harmonics[0, :, :, 0].detach().cpu().numpy())  # SH DC band
            scales.append(dump["scales"][0].detach().cpu().numpy())
            rots.append(dump["rotations"][0].detach().cpu().numpy())
            depths.append(gdepth)
            print(f"  cluster {i:2d} gaze {g}: {means[-1].shape[0]:7d} gaussians  "
                  f"(V={len(views)}, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB)")

            del gs, dump, context, imgs
            torch.cuda.empty_cache()

    means = np.concatenate(means)
    opac = np.concatenate(opac)
    colors = np.concatenate(colors)
    scales = np.concatenate(scales)
    rots = np.concatenate(rots)
    depths = np.concatenate(depths)
    print(f"\nunion: {means.shape[0]} gaussians from {K} clusters x {G} gazes "
          f"= {n_forward} forwards (each 256x256 pixel -> {V*256*256} max/forward)")

    # --- opacity filtering ----------------------------------------------------
    q = np.quantile(opac, [0.5, 0.9, 0.99, 1.0])
    print(f"opacity quantiles p50/p90/p99/max: {np.round(q, 3)}")
    mask = opac >= opac_th
    print(f"opacity filter (>= {opac_th}): keep {mask.mean()*100:.1f}% "
          f"({mask.sum()}/{means.shape[0]})")
    means, opac, colors, scales, rots, depths = \
        [a[mask] for a in (means, opac, colors, scales, rots, depths)]

    # --- depth filtering (cuts far-plane shells in the void) --------------------
    dq = np.quantile(depths, [0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(f"depth-from-own-view quantiles p1/p5/p50/p95/p99/max: "
          f"{np.round(dq, 2)}")
    if max_depth > 0:
        m = depths <= max_depth
        print(f"depth filter (<= {max_depth}): keep {m.mean()*100:.1f}% "
              f"({m.sum()}/{means.shape[0]})")
        means, opac, colors, scales, rots, depths = \
            [a[m] for a in (means, opac, colors, scales, rots, depths)]

    # --- voxel downsampling to target size -------------------------------------
    # Size the voxel from the gaussian scales (local density), not the bbox -- the
    # far-plane halo inflates the bbox but not the actual content volume.
    if voxel_override > 0:
        vox = voxel_override
    else:
        vox = max(2.0 * float(np.median(scales)), 1e-3)
        print(f"auto voxel size {vox:.4f} (median gaussian scale "
              f"{np.median(scales):.4f}, target cap {target})")
    for _ in range(12):
        pts, (o2, c2, s2, r2), counts = voxel_downsample(
            means, vox, [opac, colors, scales, rots]
        )
        if len(pts) <= target:
            break
        vox *= 1.25
    means, opac, colors, scales, rots = pts, o2, c2, s2, r2
    print(f"after voxel downsample (vox={vox:.4f}): {means.shape[0]} gaussians "
          f"(bbox {np.round(np.ptp(means, axis=0), 2)})")

    # --- export ---------------------------------------------------------------
    rgb_u8 = (sh_dc_to_rgb(colors) * 255.0 + 0.5).astype(np.uint8)
    print(f"colour stats: mean {rgb_u8.mean(0).round(1)}, "
          f"p5 {np.percentile(rgb_u8, 5, 0).round(0)}, "
          f"p95 {np.percentile(rgb_u8, 95, 0).round(0)}")

    out_ply = out_dir / "init.ply"          # RF-3DGS fetchPly input format
    export_ply_simple(means, rgb_u8, out_ply)
    view_ply = out_dir / "init_view.ply"    # standard 3DGS, viewable in supersplat
    export_ply_3dgs(means, colors, opac, scales, rots, view_ply)
    with (out_dir / "clusters.json").open("w") as f:
        json.dump({"num_clusters": K, "views_per_cluster": V,
                   "sizes": sizes, "spans": spans,
                   "selected_views": sel_info,
                   "gaussians": {"union": int(len(opac)), "final": int(means.shape[0])},
                   "voxel_size": float(vox), "max_depth": max_depth,
                   "opacity_threshold": opac_th,
                   "source": str(src), "bg": bg_name}, f, indent=2)
    print(f"\nDONE -> {out_ply.resolve()} ({means.shape[0]} gaussians, "
          f"world-frame, no recentering)")
    print(f"  view -> {view_ply.resolve()}  (drag into https://supersplat.com)")
    print(f"  clusters.json -> {out_dir.resolve()}")
    print(f"\nRF-3DGS integration: copy init.ply over the dataset's points3d.ply "
          f"(back it up first), e.g. cp {out_ply} "
          f"{src / 'points3d.ply'}")


if __name__ == "__main__":
    main()
