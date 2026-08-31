"""Batch driver: fresh namespace per image, models loaded once.

The obvious speedup -- import the notebook once and loop -- risks state
leaking between images through the ~40 module-level names the pipeline
rebinds as it runs, and that failure is silent. Measured instead of assumed:
running image B alone and B-after-A in one process gave byte-identical output
across all 11 files, because runpy.run_path builds a FRESH namespace per call.

So the leak risk is handled by construction. What is left is the cost: each
run re-executes every from_pretrained, and four model loads are essentially
the whole 27.9 s/image. Memoising the loaders keeps the fresh namespace and
pays for the weights once.

    .venv-gpu/Scripts/python tools/vmst_fast.py --warm 8
    .venv-gpu/Scripts/python tools/vmst_fast.py
"""
import argparse
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import banner

RUNNER = (HERE / "vmst_run.py").resolve()
DROP = ["_ORIGINAL.png"]


def install_model_cache():
    """Memoise every from_pretrained so weights load once per process."""
    import transformers
    from transformers.modeling_utils import PreTrainedModel

    cache = {}

    def memo(orig, tag):
        def wrapper(cls, *a, **kw):
            key = (tag, getattr(cls, "__name__", str(cls)),
                   a[0] if a else kw.get("pretrained_model_name_or_path"))
            if key not in cache:
                kw.setdefault("use_safetensors", True) if tag == "model" else None
                cache[key] = orig(cls, *a, **kw)
            return cache[key]
        return classmethod(wrapper)

    PreTrainedModel.from_pretrained = memo(
        PreTrainedModel.from_pretrained.__func__, "model")
    for name in ("AutoProcessor", "AutoImageProcessor", "AutoTokenizer",
                 "AutoConfig"):
        c = getattr(transformers, name, None)
        if c is not None:
            c.from_pretrained = memo(c.from_pretrained.__func__, name)
    return cache


def outputs_for(root, rel):
    return root / rel.parent / rel.stem


def done(dest):
    return dest.exists() and any(dest.glob("*_STATISTICS_STABLE.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/svi_90_seg"))
    ap.add_argument("--warm", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    banner("segmentation taxonomy over svi_90 (cached models)")

    imgs = sorted(args.src.rglob("*.jpg"))
    todo = [(p, p.relative_to(args.src)) for p in imgs
            if args.restart or not done(outputs_for(args.out, p.relative_to(args.src)))]
    print(f"{len(imgs)} images, {len(imgs)-len(todo)} done, {len(todo)} to do")
    todo = todo[:args.warm] if args.warm else (todo[:args.limit] if args.limit else todo)
    if not todo:
        print("nothing to do")
        return

    cache = install_model_cache()
    tmp_root = args.out / "__tmp"
    ok = fail = 0
    times = []
    t0 = time.time()
    for i, (img, rel) in enumerate(todo, 1):
        dest = outputs_for(args.out, rel)
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        os.environ["VMST_IMAGE"] = str(img.resolve())
        os.environ["VMST_OUT"] = str(tmp_root.resolve())
        s = time.time()
        try:
            runpy.run_path(str(RUNNER), run_name="__main__")
            made = list(tmp_root.glob("outputs/*/*"))
            if not made:
                raise RuntimeError("no output produced")
            dest.mkdir(parents=True, exist_ok=True)
            for f in made[0].iterdir():
                if any(f.name.endswith(d) for d in DROP):
                    continue
                shutil.move(str(f), str(dest / f.name))
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAILED {rel}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        times.append(time.time() - s)
        shutil.rmtree(tmp_root, ignore_errors=True)
        if i % 25 == 0 or i == len(todo):
            med = sorted(times)[len(times)//2]
            print(f"  {i}/{len(todo)}  ok {ok} fail {fail}  {med:.1f}s/img  "
                  f"eta {(len(todo)-i)*med/3600:.1f} h  "
                  f"[{len(cache)} models cached]", flush=True)

    med = sorted(times)[len(times)//2]
    print(f"\n{ok} ok, {fail} failed in {(time.time()-t0)/60:.1f} min")
    print(f"median {med:.1f} s/image")
    if args.warm:
        print(f"\nprojection for all {len(imgs)}: {len(imgs)*med/3600:.1f} h")


if __name__ == "__main__":
    main()
