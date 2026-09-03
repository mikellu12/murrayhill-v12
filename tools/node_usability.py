"""usable: True/False on nodes.csv, instead of a node quietly not being there.

Nine of 766 Murray Hill nodes never got a 180 strip: five read as road tunnel
interiors and four sit on the Park Avenue viaduct deck, no sidewalk at all.
export_svi_180.py drops them at render time and prints why, but the drop was
only ever visible in that log -- a table with 757 rows looks complete unless
you already know to expect 766, and a reader with only vlm_observations.csv
has no way to tell "excluded" from "never existed".

This computes the SAME two tests export_svi_180.py renders by -- imported from
it, not reimplemented, so the two cannot drift apart -- and writes usable and
exclude_reason onto nodes.csv itself, the one table every node already
appears in regardless of whether it was ever rendered.

    .venv/Scripts/python tools/node_usability.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RAW, banner
from common import CFG
from export_svi_180 import _tunnel_nodes, TUNNEL_MAX_SKY, TUNNEL_MAX_MASS

# each named list in config keeps its own reason, so the table says WHY
EXCLUDE_LISTS = {
    "viaduct": "Park Avenue viaduct deck, no sidewalk",
    "viaduct_approach": "viaduct approach ramp, leaves the straight avenue",
}


def main():
    banner("usable: True/False on every node")
    path = PROC / "nodes.csv"
    nodes = pd.read_csv(path)

    tunnels = _tunnel_nodes(nodes.node_id)
    listed = {}
    for key, why in EXCLUDE_LISTS.items():
        for nid in CFG.get("excluded_nodes", {}).get(key, []):
            listed[nid] = why
    viaduct = set(listed) & set(nodes.node_id)

    # USER-CONTRIBUTED PANORAMAS ARE OUT. Google's own captures carry 22-char
    # pano ids; user photospheres carry long CAoS... ids, and in the City of
    # London they are what filled fifteen years of side streets -- from
    # tourist buses, hotel lobbies and shop interiors wearing a street's
    # coordinates. The id format is the provenance record.
    user_pano = {}
    meta_path = RAW / "metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "pano_id" in meta.columns:
            ids = meta.pano_id.astype(str)
            mask = ids.str.len() > 22
            user_pano = dict(zip(meta.loc[mask, "node_id"],
                                 ids[mask].str[:6]))

    reason = pd.Series("", index=nodes.index, dtype=object)
    for i, nid in enumerate(nodes.node_id):
        if nid in tunnels:
            sky, mass = tunnels[nid]
            reason.iat[i] = (f"tunnel interior (sky {sky:.1%} < "
                             f"{TUNNEL_MAX_SKY:.0%}, classified {mass:.1%} < "
                             f"{TUNNEL_MAX_MASS:.0%})")
        elif nid in viaduct:
            reason.iat[i] = listed[nid]
        elif nid in user_pano:
            reason.iat[i] = ("user-contributed panorama, not a Street View "
                             "capture")

    nodes["usable"] = reason == ""
    nodes["exclude_reason"] = reason
    nodes.to_csv(path, index=False)

    n_bad = int((~nodes.usable).sum())
    n_user = int(nodes.exclude_reason.str.startswith("user-contributed").sum())
    print(f"{len(nodes)} nodes, {n_bad} not usable "
          f"({len(tunnels)} tunnel, {len(viaduct)} viaduct, {n_user} "
          f"user-contributed)")
    for nid in sorted(tunnels) + sorted(viaduct):
        r = nodes.loc[nodes.node_id == nid, "exclude_reason"].iloc[0]
        print(f"  {nid}  {r}")


if __name__ == "__main__":
    main()
