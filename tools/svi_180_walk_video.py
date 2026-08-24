"""Each walk as a short film: the comparison frames at a fixed dwell.

One video per street per direction, not per street. The two walks are
opposite traversals of the same block faces, so joining them would run to the
far end and teleport back to the start halfway through -- the export splits
them for the same reason, and the split is the comparison the study is about.

Frames come from results/svi_180_segmented_comparison, already photo-over-mask
and already in walking order, so this only sets the dwell and encodes.

Timing is done with the concat demuxer rather than -framerate, because the
zero-padding width varies with how many nodes a street has and ffmpeg's
Windows builds usually lack glob support for -pattern_type. An explicit list
also fixes the order rather than trusting a directory scan.

H.264 needs even dimensions and the comparison frames are 1440x1853, an odd
height, so the scale filter rounds to even (-2) instead of cropping a row off
the mask.

Street View pixels, so the output is gitignored like the frames it is built
from.

    .venv/Scripts/python tools/svi_180_walk_video.py
    .venv/Scripts/python tools/svi_180_walk_video.py --dwell 0.4 --width 1080
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import RES, banner


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # winget edits PATH only for shells started after the install, so a
    # session older than the install will not see it. Gyan.FFmpeg unpacks to a
    # versioned directory under WinGet\Packages, so the version is globbed
    # rather than pinned.
    pkgs = (Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet"
            / "Packages")
    for cand in sorted(pkgs.glob("*FFmpeg*/**/bin/ffmpeg.exe"), reverse=True):
        return str(cand)
    sys.exit("ffmpeg not found; winget install --id Gyan.FFmpeg -e, "
             "then start a new shell")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=RES / "svi_180_segmented_comparison")
    ap.add_argument("--out", type=Path, default=RES / "svi_180_walk_videos")
    ap.add_argument("--dwell", type=float, default=0.2,
                    help="seconds per image (default 0.2)")
    ap.add_argument("--width", type=int, default=1440,
                    help="output width; height follows, rounded to even")
    ap.add_argument("--crf", type=int, default=20,
                    help="x264 quality, lower is better and larger")
    ap.add_argument("--street", default=None, help="only this street folder")
    args = ap.parse_args()
    banner("one video per walk")

    if not args.src.exists():
        sys.exit(f"no frames at {args.src} -- run tools/svi_180_comparison.py")
    ffmpeg = _ffmpeg()
    print(f"ffmpeg {ffmpeg}")
    print(f"{args.dwell}s per image ({1 / args.dwell:.1f} fps), width {args.width}")

    walks = sorted(d for d in args.src.glob("*/*") if d.is_dir())
    if args.street:
        walks = [w for w in walks if w.parent.name == args.street]
    if not walks:
        sys.exit(f"no walk folders under {args.src}")

    made, failed = [], []
    for walk in walks:
        frames = sorted(walk.glob("*.jpg"))
        if not frames:
            continue
        street, direction = walk.parent.name, walk.name
        dest = args.out / street / f"{direction}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)

        # The concat demuxer holds the last entry for its duration only if the
        # file is repeated at the end; without it the final frame flashes by.
        lines = []
        for f in frames:
            lines.append(f"file '{f.resolve().as_posix()}'")
            lines.append(f"duration {args.dwell}")
        lines.append(f"file '{frames[-1].resolve().as_posix()}'")

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            listfile = fh.name
        try:
            r = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe",
                 "0", "-i", listfile, "-vf", f"scale={args.width}:-2",
                 "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
                capture_output=True, text=True)
        finally:
            Path(listfile).unlink(missing_ok=True)

        if r.returncode != 0 or not dest.exists():
            failed.append((street, direction, r.stderr.strip()[:200]))
            continue
        mb = dest.stat().st_size / 1024 / 1024
        made.append((street, direction, len(frames),
                     len(frames) * args.dwell, mb))
        print(f"  {street:<22}{direction:<16}{len(frames):>4} frames  "
              f"{len(frames) * args.dwell:>5.1f}s  {mb:>5.1f} MB")

    total = sum(m[4] for m in made)
    print(f"\n{len(made)} videos -> {args.out}  ({total:.0f} MB)")
    if failed:
        print(f"\n{len(failed)} failed:")
        for s, d, err in failed:
            print(f"  {s}/{d}: {err}")


if __name__ == "__main__":
    main()
