"""Run the segmentation-taxonomy pipeline over every svi_90 half-view.

The source is a Colab notebook that segments ONE uploaded image per run:
tools/vmst_build.py converts it to a script, and this drives that script once
per image, mirroring svi_90's street/walk folders into svi_90_seg.

WHY A SUBPROCESS PER IMAGE, AND NOT AN IMPORT LOOP. The notebook is 29 cells
of linear code sharing module-level state -- `final_test`, `final_v06`,
`TEST_ID` and about forty other names, rebound as the pipeline progresses.
Importing it once and looping would carry state between images, and the first
one to leak would be silent: a mask from the previous frame surviving into the
next produces a plausible label map, not a crash. A process per image costs
the model loads again but cannot leak, and correctness matters more here than
the wall clock. `--warm` reports what that trade actually costs.

ORIGINAL.png is dropped after each run: it is a lossless re-encode of the
input JPEG, verified pixel-identical across all 2,639,520 pixels, at 2.35 MB
against the source's 100 KB. Everything else is kept.

    .venv-gpu/Scripts/python tools/vmst_batch.py --warm 6      # time it first
    .venv-gpu/Scripts/python tools/vmst_batch.py               # the full run
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import banner

RUNNER = HERE / "vmst_run.py"
DROP = ["_ORIGINAL.png"]
# written once per run into <out>/config; identical every time
CONFIG_DIR = "config"


def outputs_for(root, rel):
    """Where one image's files land: svi_90_seg/<street>/<walk>/<stem>/."""
    return root / rel.parent / rel.stem


def already_done(dest):
    return (dest / f"{dest.name.upper()}_STATISTICS_STABLE.csv").exists() or \
           any(dest.glob("*_STATISTICS_STABLE.csv"))


def run_one(img, dest, python, env_extra=None):
    tmp = dest.parent / f".__tmp_{dest.name}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, VMST_IMAGE=str(img.resolve()),
               VMST_OUT=str(tmp.resolve()), PYTHONIOENCODING="utf-8")
    env.update(env_extra or {})
    r = subprocess.run([python, str(RUNNER)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return False, (r.stderr or r.stdout or "")[-600:]

    # the runner writes <tmp>/outputs/<version>/<TEST_ID>/*
    made = list(tmp.glob("outputs/*/*"))
    if not made:
        shutil.rmtree(tmp, ignore_errors=True)
        return False, "no output directory produced"
    src = made[0]
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if any(f.name.endswith(d) for d in DROP):
            continue
        shutil.move(str(f), str(dest / f.name))
    shutil.rmtree(tmp, ignore_errors=True)
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/svi_90_seg"))
    ap.add_argument("--python", default=str(
        Path(".venv-gpu/Scripts/python.exe").resolve()))
    ap.add_argument("--warm", type=int, default=None,
                    help="time this many images and stop, without writing an ETA")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="images in flight at once. Peak VRAM is 3.9 GB "
                         "allocated / 6.5 GB reserved per image on a 12.9 GB "
                         "card, so 2 fits once the allocator is capped; the "
                         "cap is set below and applies per worker.")
    ap.add_argument("--restart", action="store_true",
                    help="redo images that already have output")
    args = ap.parse_args()
    banner("segmentation taxonomy over svi_90")

    if not RUNNER.exists():
        sys.exit(f"{RUNNER} missing -- run tools/vmst_build.py first")

    imgs = sorted(args.src.rglob("*.jpg"))
    rel = [p.relative_to(args.src) for p in imgs]
    todo = [(p, r) for p, r in zip(imgs, rel)
            if args.restart or not already_done(outputs_for(args.out, r))]
    print(f"{len(imgs)} images, {len(imgs) - len(todo)} already done, "
          f"{len(todo)} to do")
    if args.warm:
        todo = todo[:args.warm]
    elif args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do")
        return

    ok = fail = 0
    times = []
    t0 = time.time()
    # Cap the caching allocator so N workers cannot each reserve the card.
    # Reserved is 6.5 GB against 3.9 GB actually allocated, so without this a
    # second worker meets an out-of-memory error the first one is not using.
    extra = {}
    if args.workers > 1:
        frac = 0.92 / args.workers
        extra["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"expandable_segments:True,max_split_size_mb:256")
        extra["VMST_MEM_FRACTION"] = f"{frac:.3f}"

    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs = [(img, r) for img, r in todo]
    i = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, img, outputs_for(args.out, r), args.python,
                          extra): (img, r, time.time()) for img, r in jobs}
        for fut in as_completed(futs):
            img, r, started = futs[fut]
            i += 1
            good, err = fut.result()
            times.append(time.time() - started)
            if good:
                ok += 1
            else:
                fail += 1
                print(f"  FAILED {r}: {err.splitlines()[-1] if err else '?'}")
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                left = (len(todo) - i) / max(rate, 1e-9)
                print(f"  {i}/{len(todo)}  ok {ok} fail {fail}  "
                      f"{rate*3600:.0f} img/h  eta {left/3600:.1f} h", flush=True)

    # THROUGHPUT, not per-image elapsed. With W workers each image's timer
    # starts at submission, so anything queued behind another job counts the
    # wait as compute -- at 2 workers that reads 111 s/image for work that
    # actually finishes 181 an hour. Wall clock over count is the only figure
    # that means anything once more than one image is in flight.
    #
    # Measured on this card (12.9 GB, peak 3.9 GB allocated per image):
    #   1 worker   129 img/h   23.8 h
    #   2 workers  181 img/h   16.9 h   <- best
    #   3 workers  134 img/h   22.9 h   thrashes, no better than serial
    elapsed = time.time() - t0
    rate = ok / max(elapsed, 1e-9)
    print(f"\n{ok} ok, {fail} failed in {elapsed/60:.1f} min")
    print(f"throughput {rate*3600:.0f} images/h "
          f"({elapsed/max(ok,1):.1f} s/image wall clock, "
          f"{args.workers} worker(s))")
    if args.warm:
        n = len(imgs)
        print(f"\nprojection for all {n} images:")
        print(f"  {n / max(rate*3600, 1e-9):.1f} h")
        one = outputs_for(args.out, todo[0][1])
        if one.exists():
            mb = sum(f.stat().st_size for f in one.iterdir()) / 1e6
            print(f"  {mb:.2f} MB/image  ->  {n * mb / 1000:.1f} GB")


if __name__ == "__main__":
    main()
