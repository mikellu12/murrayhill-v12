"""Recompute the sequence prefix on exported half-views from the full street.

The exporter used to enumerate the FILTERED batch, so every partial pass
restarted at 1. A --missing-only run over four nodes wrote seq 1-4 into a
folder that already held 1-4, duplicating them and interleaving two orderings;
the zero-pad width came from the batch size too, so the same folder ended up
with both 1_ and 01_. Sorting by name or by seq then jumps around the street.

The exporter now numbers the whole street before filtering. This brings
already-exported files into line without re-rendering them: the pixels are
correct, only the names are wrong.

The filename is a join key -- `file` links observations to vlm_calculations
and tells sim_vlm_run what is already rated -- so files and tables are
rewritten in one pass, matched on (street, walk, node_id, side).

    .venv/Scripts/python tools/renumber_svi.py --dry-run
    .venv/Scripts/python tools/renumber_svi.py --apply
"""
import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner
from export_svi_180 import _cardinal, _street_axis, _walks
from export_svi_90 import SEQ_WIDTH

UTM = 32618
NAME = re.compile(r"^(\d+)_(n\d+)_([NESW])(?:_([LR]))?\.jpg$")
TABLES = ["vlm_observations.csv", "vlm_calculations.csv", "sim_vlm_v3.csv",
          "sim_vlm_v4_clean.csv", "sim_vlm_180_holdout.csv"]


def correct_seq(nodes):
    """(folder, walk, node_id) -> seq, numbering each street in full."""
    out = {}
    u = nodes.to_crs(UTM)
    nodes = nodes.assign(_e=u.geometry.x.values, _n=u.geometry.y.values)
    for street, g in nodes.groupby("folder"):
        axis = _street_axis(g._e.to_numpy(), g._n.to_numpy())
        for bearing, walk in _walks(axis):
            e, n_ = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
            ordered = g.assign(
                _proj=np.round(g._e * e + g._n * n_),
                _perp=g._e * n_ - g._n * e,
            ).sort_values(["_proj", "_perp"])
            for seq, row in enumerate(ordered.itertuples(), start=1):
                out[(street, walk, row.node_id)] = seq
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    banner(f"renumber {args.root} from the full street ordering")

    nodes = gpd.read_file(PROC / "nodes.gpkg")
    col = ("street_segment" if "street_segment" in nodes.columns
           and nodes.street_segment.notna().any() else "cleaned_street")
    nodes = nodes[nodes[col].notna()].copy()
    nodes["folder"] = nodes[col]
    seq = correct_seq(nodes)
    print(f"numbered {len(seq)} (street, walk, node) positions from {col}")

    moves, unknown = {}, 0
    for p in sorted(args.root.rglob("*.jpg")):
        m = NAME.match(p.name)
        if not m:
            unknown += 1
            continue
        street, walk = p.parts[-3], p.parts[-2]
        k = (street, walk, m.group(2))
        if k not in seq:
            unknown += 1
            continue
        side = f"_{m.group(4)}" if m.group(4) else ""
        new = f"{seq[k]:0{SEQ_WIDTH}d}_{m.group(2)}_{m.group(3)}{side}.jpg"
        if new != p.name:
            moves[str(p.relative_to(args.root)).replace("\\", "/")] = \
                str((p.parent / new).relative_to(args.root)).replace("\\", "/")

    print(f"{len(moves)} file(s) to rename, {unknown} unrecognised\n")
    for a, b in list(moves.items())[:4]:
        print(f"  {a}\n    -> {b}")
    if len(moves) > 4:
        print(f"  ... and {len(moves) - 4} more")

    clash = [b for b in moves.values()
             if (args.root / b).exists() and b not in moves]
    if clash:
        sys.exit(f"{len(clash)} target name(s) already taken, e.g. {clash[0]}")

    hits = {}
    for t in TABLES:
        f = RES / "tables" / t
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if "file" not in d.columns:
            continue
        n = int(d.file.isin(moves).sum())
        hits[t] = (f, d, n)
        print(f"  {t}: {n} of {len(d)} rows affected")

    if not args.apply:
        print("\ndry run -- nothing written. re-run with --apply")
        return

    # two phases, so a name being freed and taken in the same pass cannot clash
    tmp = {}
    for a in moves:
        src = args.root / a
        t = src.with_name("__tmp__" + src.name)
        src.rename(t)
        tmp[a] = t
    for a, b in moves.items():
        tmp[a].rename(args.root / b)
    print(f"\nrenamed {len(moves)} file(s)")

    for t, (f, d, n) in hits.items():
        if not n:
            continue
        d["file"] = d.file.map(lambda x: moves.get(x, x))
        if "seq" in d.columns:
            d["seq"] = d.file.str.split("/").str[-1].str.split("_").str[0].astype(int)
        d.to_csv(f, index=False)
        print(f"  rewrote {t}")


if __name__ == "__main__":
    main()
