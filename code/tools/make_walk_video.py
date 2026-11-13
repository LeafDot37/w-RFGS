# -*- coding: utf-8 -*-
"""
make_walk_video.py —— 把漫游帧编成 mp4（每方法一条 + 组级 MVDR|CBF 并排对照）
================================================================================
输入：pull_rrf_walk.sh 拉回的帧
      <VIS>/<M>/漫游/<traj>/00000.png ...  （300×200，A× 的评测同口径）
输出（遵循「图按每方法文件夹；说明.md 只放组目录一份」）：
      <VIS>/<M>/漫游_环绕.mp4        每方法一条
      <VIS>/<M>/漫游_推轨.mp4
      <VIS>/对照_环绕.mp4            组级并排（左 MVDR / 右 CBF）
      <VIS>/对照_推轨.mp4

两条编码路线，分工明确
--------------------------------------------------------------------------------
- **每方法一条**：ffmpeg 直接读 PNG 序列（`-i %05d.png`），不逐帧解到 Python。
- **并排对照**：PIL 先 hstack 并加帧标签条，落临时目录，再交给 ffmpeg。
  之所以不用 ffmpeg 的 hstack/drawtext 滤镜：drawtext 依赖 fontconfig 与字体路径，
  在 Windows/WSL 混搭环境下最容易炸；PIL 侧用 matplotlib 自带的 DejaVuSans
  （一定存在）画 ASCII 标签，零系统依赖。
  标签**只用 ASCII 方法名**，故不需要 CJK 字体。

ffmpeg 路径解析（本地是「WSL 里跑 python、ffmpeg 是 Windows exe」，必须转路径）
--------------------------------------------------------------------------------
按序找：`--ffmpeg` → `$FFMPEG` → PATH 里的 ffmpeg → 仓库外挂的那个。
若命中的是 Windows exe 而当前在 posix（WSL），用 `wslpath -w` 把参数路径
转成 `D:\\...` 再传 —— Windows 程序不认 `/mnt/d/...`。

用法（本地仓库根）：
    python code/tools/make_walk_video.py --vis results/run_20260912_多频段标量分支/阶段2_多频段_三维可视化
    ... --fps 24 --scale 3 --traj orbit dolly --methods MVDR CBF
    ... --dry-run        # 只打印将要执行的 ffmpeg 命令

三臂模式（两组预算三臂对比用）
--------------------------------------------------------------------------------
臂不再等同于「方法」：同一方法下有 A1/A2/A3 三条臂的帧，要并成一条视频。
用 `--arms "标签=目录[,标签=目录...]"` 显式给臂表，目录里可用 `{M}` 占位符
（对 `--methods` 逐个替换），输出落到 `--arms-out` 模式（同样支持 `{M}`/`{traj}`/`{cn}`）：

    python code/tools/make_walk_video.py \
      --vis 'results/run_20260912_两组预算_三臂对比/组1_物理30000_RRF30000/视频' \
      --traj orbit dolly --methods MVDR CBF \
      --arms 'A1=.../A1_原版物理_原版RRF/{M}/漫游-orbit形式, A2=..., A3=...' \
      --arms-out '{vis}/{M}/对照_{cn}.mp4'

给了 `--arms` 就**不再**出「每方法一条」的单臂视频（那是 `--methods` 模式的产物，
帧路径模板不同，不混用）。缺任一臂即整条跳过并打印缺谁 —— 三臂对比里静默少一条臂
比报错更危险。
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    # stderr 也要 —— 否则报错里的中文路径在 GBK 控制台上变成乱码，看不懂是哪个目录缺了
    sys.stderr.reconfigure(encoding="utf-8")

# 轨迹名 → 交付物用的中文标签（中文命名固定，见 results 命名约定）
TRAJ_CN = {"orbit": "环绕", "dolly": "推轨", "orbit_pitch": "俯仰外推",
           # `real` **不是合成轨迹**：每一帧都是数据集真实机位（见 pick_real_views.py），
           # 故它那条视频的第 1 栏 GT 与三臂严格同机位 —— 与 orbit/dolly 的诚实性口径不同。
           "real": "真实机位"}
_FF = "gaussian-splatting/gaussian-splatting/external/ffmpeg/bin/ffmpeg"
DEFAULT_FFMPEG = ["ffmpeg"]


def find_ffmpeg(explicit: str) -> str:
    for cand in [explicit, os.environ.get("FFMPEG", ""), shutil.which("ffmpeg") or ""] + \
            DEFAULT_FFMPEG:
        if cand and Path(cand).is_file():
            return cand
    raise SystemExit("找不到 ffmpeg：用 --ffmpeg <路径> 或设 $FFMPEG")


class Runner:
    """统一处理「Windows exe 从 WSL 调用」的路径转换。"""

    def __init__(self, ffmpeg: str, dry: bool):
        self.ffmpeg = ffmpeg
        self.dry = dry
        self.is_win_exe = ffmpeg.lower().endswith(".exe") or ffmpeg.startswith("/mnt/")
        self.need_wslpath = self.is_win_exe and os.name == "posix"

    def native(self, p) -> str:
        p = str(p)
        if not self.need_wslpath:
            return p
        out = subprocess.run(["wslpath", "-w", p], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"wslpath 失败: {p} :: {out.stderr.strip()}")
        return out.stdout.strip()

    def run(self, args: list):
        # argv[0] **不能**转成 Windows 路径：从 WSL 调 Windows exe 靠的是 binfmt 互操作，
        # 内核要看到的是 Linux 形式（/mnt/d/.../ffmpeg.exe）才找得到文件；写成
        # `D:\...\ffmpeg.exe` 会直接 FileNotFoundError。而**参数**恰恰相反 ——
        # exe 是 Windows 进程，它打不开 /mnt/d 这种路径，故参数必须转 Windows 形式。
        cmd = [self.ffmpeg] + [self.native(a) if a.startswith("/") else a
                               for a in args]
        if self.dry:
            print("    [dry] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            return
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stderr[-3000:], file=sys.stderr)
            raise RuntimeError(f"ffmpeg 退出码 {r.returncode}")


def encode_seq(rn: Runner, frames: Path, out: Path, fps: int, scale: int):
    """PNG 序列 → mp4。yuv420p + 偶数边长，保证各播放器都能开。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = (f"scale=iw*{scale}:ih*{scale}:flags=lanczos" if scale != 1 else "null")
    rn.run(["-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frames / "%05d.png"),
            "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-movflags", "+faststart", str(out)])
    if not rn.dry:
        print(f"    → {out}  ({out.stat().st_size/2**20:.2f} MB)")


def make_side_by_side(rn: Runner, dirs: list, labels: list, tmp: Path,
                      out: Path, fps: int, scale: int):
    """逐帧 hstack + 顶部标签条，再编 mp4。

    dirs 顺序即左右顺序；轴 0 竖直拼接（标签在上、画面在下）。
    标签若非纯 ASCII（如 `A×` 的 `×`），必须走 CJK 字体，否则 `load_default()`
    会渲染成豆腐块 —— 图注读不出来等于没标。
    """
    from PIL import Image, ImageDraw, ImageFont
    font = _find_font(need_cjk=any(not l.isascii() for l in labels))
    pairs = []
    for d in dirs:
        fs = sorted(Path(d).glob("*.png"))
        if not fs:
            raise FileNotFoundError(f"无帧: {d}")
        pairs.append(fs)
    n = min(len(p) for p in pairs)
    if n < 2:
        raise ValueError(f"帧数太少: {[len(p) for p in pairs]}")
    bar = 30
    for k in range(n):
        ims = [Image.open(p[k]).convert("RGB") for p in pairs]
        w, h = ims[0].size
        canvas = Image.new("RGB", (w * len(ims), h + bar), (16, 16, 16))
        dr = ImageDraw.Draw(canvas)
        for i, im in enumerate(ims):
            canvas.paste(im, (i * w, bar))
            dr.text((i * w + 8, 6), f"{labels[i]}   {k:03d}/{n-1}", fill=(235, 235, 235),
                    font=font)
            if i:
                dr.line([(i * w, 0), (i * w, h + bar)], fill=(90, 90, 90), width=2)
        canvas.save(tmp / f"{k:05d}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale=iw*{scale}:ih*{scale}:flags=lanczos" if scale != 1 else "null"
    rn.run(["-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(tmp / "%05d.png"), "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-movflags", "+faststart", str(out)])
    if not rn.dry:
        print(f"    → {out}  ({out.stat().st_size/2**20:.2f} MB, {n} 帧)")


def _find_font(need_cjk: bool = False):
    """need_cjk=False 时只要 ASCII（DejaVuSans，matplotlib 自带，一定存在）；
    为 True 时优先中文字体（Windows 的 msyh/simhei），找不到就**如实告警**，
    不假装标签能读。"""
    from PIL import ImageFont
    cjk = [Path("/mnt/c/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyh.ttc"),
           Path("/mnt/c/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/simhei.ttf"),
           Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
           Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")]
    ascii_ = [Path("/mnt/c/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arial.ttf")]
    try:
        import matplotlib
        ascii_.insert(0, Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
    except Exception:
        pass
    for c in (cjk + ascii_ if need_cjk else ascii_ + cjk):
        if c.is_file():
            try:
                return ImageFont.truetype(str(c), 18)
            except Exception:
                pass
    if need_cjk:
        print("  [warn] 标签含非 ASCII 但未找到中文字体 —— 标签会显示为豆腐块。"
              "改用纯 ASCII 臂名（如 A1_yuanban_RRF）重跑。")
    else:
        print("  [warn] 未找到 TrueType 字体，改用 PIL 默认位图字体（标签会很小）")
    return ImageFont.load_default()


def parse_arms(spec: str):
    """'标签=目录[,标签=目录...]' → [(label, dir)]；标签留空则退回目录名。"""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--arms 项缺 '=': {item!r}（格式 标签=目录）")
        lab, d = item.split("=", 1)
        lab, d = lab.strip(), d.strip()
        if not d:
            raise SystemExit(f"--arms 项目录为空: {item!r}")
        out.append((lab or Path(d).name, d))
    if not out:
        raise SystemExit("--arms 解析为空")
    return out


def resolve_arm_dir(d: str, vis: Path) -> Path:
    """把 --arms 的目录解析成 Path。

    历史契约是**相对当前工作目录**的完整路径（头注释的示例就是这么写的），所以先按原样认。
    但三臂调用方更自然的写法是给 **vis 相对路径**（如 'MPC/漫游/A1_yuanban_RRF/orbit'），
    而原来只 `Path(d)` 会让它相对 CWD 解析 —— 结果是每个臂都判成"目录缺失"、四条视频
    全被静默跳过（**只打印一行"缺臂目录"**，rc 仍是 0）。故这里加一层回退：
    原样不存在时，再相对 `vis`（即 `--vis` 指的那个目录）找。
    已有的调用方传的路径都能原样命中，回退不触发 ⇒ 对它们是行为保持的。
    """
    p = Path(d)
    if p.is_absolute() or p.is_dir():
        return p
    return vis / p


def main():
    ap = argparse.ArgumentParser(description="漫游帧 → mp4（每方法 + 组级并排对照）")
    ap.add_argument("--vis", required=True, help="阶段2_多频段_三维可视化 目录")
    ap.add_argument("--methods", nargs="+", default=["MVDR", "CBF"])
    ap.add_argument("--traj", nargs="+", default=["orbit", "dolly"])
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--scale", type=int, default=3, help="整数倍放大（帧原生 300×200）")
    ap.add_argument("--ffmpeg", default="")
    ap.add_argument("--only-method", action="store_true", help="只出每方法视频，不出并排对照")
    ap.add_argument("--arms", default="",
                    help="三臂模式：'标签=目录[,标签=目录...]'；目录里可用 {M} 占位符")
    ap.add_argument("--arms-out", default="",
                    help="三臂模式的输出路径模式（支持 {M} {traj} {cn}）；"
                         "空则 <vis>/<M>/对照_<中文轨迹>.mp4")
    ap.add_argument("--tmp", default="", help="并排帧的临时根目录；默认 <vis>/../_walk_video_tmp")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    vis = Path(args.vis)
    if not vis.is_dir():
        raise SystemExit(f"目录不存在: {vis}")
    arms = parse_arms(args.arms) if args.arms else []
    rn = Runner(find_ffmpeg(args.ffmpeg), args.dry_run)
    n_skip = n_ok = 0        # 跳过的条数 / 成功的条数 —— 全跳过时必须回非零（见文件尾）
    print(f"ffmpeg: {rn.ffmpeg}   (WSL→Windows 路径转换: {rn.need_wslpath})")
    print(f"vis: {vis}   methods={args.methods}  traj={args.traj}  fps={args.fps}  scale={args.scale}")
    if arms:
        print(f"三臂模式，臂表: {[l for l, _ in arms]}")

    for T in args.traj:
        cn = TRAJ_CN.get(T, T)
        print(f"[{T} / {cn}]")

        if not arms:
            for M in args.methods:
                d = vis / M / "漫游" / T
                if not d.is_dir():
                    print(f"  [{M}] 缺帧目录，跳过: {d}")
                    n_skip += 1
                    continue
                encode_seq(rn, d, vis / M / f"漫游_{cn}.mp4", args.fps, args.scale)
                n_ok += 1

        if args.only_method:
            continue
        # 临时帧必须落在 **Windows 可见的盘**上：若在 WSL 里跑，/tmp 传给 Windows ffmpeg
        # 会变成 \\wsl.localhost\... 的 UNC 路径，慢且易失败。故默认放 vis 同级。
        tmp_root = Path(args.tmp) if args.tmp else vis.parent / "_walk_video_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)

        if arms:
            # 每方法一条三臂并排：{M} 在臂目录与输出模式里替换
            for M in args.methods:
                pairs = [(lab, resolve_arm_dir(d.format(M=M), vis)) for lab, d in arms]
                miss = [f"{lab}:{d}" for lab, d in pairs if not d.is_dir()]
                if miss:                       # 缺任一臂 ⇒ 整条跳过（三臂对比少一条臂不能静默）
                    print(f"  [并排 {M}] 缺臂目录，跳过: {miss}")
                    n_skip += 1
                    continue
                out = (Path(args.arms_out.format(M=M, traj=T, cn=cn, vis=vis))
                       if args.arms_out else vis / M / f"对照_{cn}.mp4")
                td = Path(tempfile.mkdtemp(dir=str(tmp_root), prefix=f"{T}_{M}_"))
                try:
                    make_side_by_side(rn, [d for _, d in pairs], [lab for lab, _ in pairs],
                                      td, out, args.fps, args.scale)
                    n_ok += 1
                finally:
                    if not args.dry_run:
                        shutil.rmtree(td, ignore_errors=True)
            continue

        dirs = [vis / M / "漫游" / T for M in args.methods]
        if not all(d.is_dir() for d in dirs):
            print(f"  [并排] 缺臂目录，跳过: {[str(d) for d in dirs if not d.is_dir()]}")
            n_skip += 1
            continue
        td = Path(tempfile.mkdtemp(dir=str(tmp_root), prefix=f"{T}_"))
        try:
            make_side_by_side(rn, dirs, args.methods, td,
                              vis / f"对照_{cn}.mp4", args.fps, args.scale)
            n_ok += 1
        finally:
            # 只删自己刚 mkdtemp 出来的那一个目录（名字唯一，绝不动已有内容）
            if not args.dry_run:
                shutil.rmtree(td, ignore_errors=True)

    if not args.dry_run:
        try:
            # 只在**空**的时候删掉这个壳；非空（自己或别人留的东西）就不动
            (Path(args.tmp) if args.tmp else vis.parent / "_walk_video_tmp").rmdir()
        except OSError:
            pass

    # 两条轨道的诚实性口径**不同**，不能合并成一句：
    #   real      —— 每帧都是数据集真实机位 ⇒ 有真值，四栏严格同机位，可算指标；
    #   orbit 等  —— 合成新机位 ⇒ 没有真值，只能看、不能算指标。
    if "real" in args.traj:
        print("完成。`real`（真实机位）那条的每一帧**都是数据集真实机位**（位置/朝向/图像均取自"
              "数据集，只做了排序）⇒ 第 1 栏 GT 与三臂渲染**逐帧严格同机位**，可以算指标。")
    if any(T != "real" for T in args.traj):
        print("完成。注意：新视角（orbit/dolly 等合成机位）**没有 GT**，那些视频只能看，"
              "**不能算指标**。")
    # 一条都没出成 ⇒ 必须回非零：否则调用方看到 rc=0 会把它当成功，
    # 而实际产物是**零个视频**（本脚本踩过：臂目录写错时每条都只打印一行"缺臂目录"）。
    if n_skip and not n_ok:
        raise SystemExit(f"成片失败：{n_skip} 条全被跳过，产出 0 个视频")
    if n_skip:
        print(f"  [注意] 有 {n_skip} 条被跳过、{n_ok} 条成功 —— 交付物不完整")


if __name__ == "__main__":
    main()
