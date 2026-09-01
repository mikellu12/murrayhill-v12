"""Who took each panorama: Google, or a user.

Google captures streets with a car or a trekker. Everything else in Street View
is a photosphere somebody uploaded -- shop interiors, and, as it turns out,
photographs taken from a tour boat and the top deck of a bus. Those are not
street-level public space and have no business in a street measure, but they
are not distinguishable from a good frame by anything in the pixels alone.

The copyright field in the metadata says who. Metadata requests are free, so
this costs nothing but time.

    .venv/Scripts/python tools/probe_provenance.py
"""
import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, banner

META = "https://maps.googleapis.com/maps/api/streetview/metadata"


def key():
    env = Path(".env").read_text(encoding="utf-8") if Path(".env").exists() else ""
    m = re.search(r"GMAPS_KEY\s*=\s*(\S+)", env)
    return (m.group(1).strip('"\'') if m else os.environ.get("GMAPS_KEY"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    banner("panorama provenance")

    n = pd.read_csv(PROC / "nodes.csv")[["node_id", "lat", "lon"]]
    k = key()
    rows = []
    for _, r in tqdm(list(n.iterrows()), desc="metadata", mininterval=2.0):
        try:
            js = requests.get(META, params={"location": f"{r.lat},{r.lon}",
                                            "source": "outdoor", "key": k},
                              timeout=15).json()
        except Exception:
            js = {}
        rows.append({"node_id": r.node_id,
                     "copyright": str(js.get("copyright", "")).strip()})
    d = pd.DataFrame(rows)
    d["google"] = d.copyright.str.contains("Google", case=False, na=False)
    out = args.out or PROC / "provenance.csv"
    d.to_csv(out, index=False)
    ng = int((~d.google).sum())
    print(f"\n{len(d)} nodes: {int(d.google.sum())} Google, {ng} user "
          f"({ng/len(d)*100:.1f}%)")
    print("\n  most frequent uploaders:")
    print("   " + d[~d.google].copyright.value_counts().head(12)
          .to_string().replace("\n", "\n   "))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
