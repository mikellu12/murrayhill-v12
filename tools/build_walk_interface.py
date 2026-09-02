"""A walk-through page: the view, the scores, and what the model said.

Writes a self-contained HTML file that steps along one street node by node --
the rendered view on the left, the ten field ratings and M on the right, and
the model's own description beneath them.

LOCAL ONLY, AND DELIBERATELY. The page references the rendered JPEGs by
relative path rather than embedding them. Street View imagery is not
redistributable -- Google caps caching at 30 days, and derived measurements are
ours to publish while the photographs are not -- so this must never be uploaded
or handed out as a link. Opening it from the repository is fine; that is a
local view of local files.

It degrades. Ratings, descriptions and M each appear if their table exists and
are quietly absent if not, so the page can be built and looked at before the
qualitative pass has finished rather than only afterwards.

ORDER COMES FROM THE FILENAME SEQUENCE within a corridor, which is the one
ordering that follows the walk on every street: chain_pos_m runs in an
arbitrary direction per chain, and the frame's own seq restarts inside each
corridor of a split street.

    .venv/Scripts/python tools/build_walk_interface.py --street east_38th_street
    SIM_CONFIG=config_london.yaml .venv/Scripts/python \
        tools/build_walk_interface.py --street london_wall
"""
import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RAW, RES, banner

NAME = re.compile(r"^(\d+)_(n\d+)_([NESW])(?:_([LRF]))?\.jpg$")
FIELDS = ["vertical_greenery", "green_eye_level", "green_softening",
          "vertical_hardscape", "sky_openness", "signage_detail",
          "walkable_ground", "resting_affordance", "ground_floor_activity",
          "facade_variation"]
DIM = {"I_raw": "imageability", "Y": "identity", "D_raw": "dependence"}
QCOLS = ["scene", "greenery", "ground", "frontage"]


def load(path, cols=None):
    if not path.exists():
        print(f"  (no {path.name})")
        return None
    d = pd.read_csv(path)
    return d[[c for c in cols if c in d.columns]] if cols else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street", required=True)
    ap.add_argument("--walk", default=None)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--ratings", type=Path, default=None)
    ap.add_argument("--descriptions", type=Path, default=None)
    ap.add_argument("--calc", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    area = CFG.get("study_area_name", "study area")
    banner(f"walk-through page: {args.street}")

    src = args.src or (RAW / "svi_180")
    base = src / args.street
    if not base.exists():
        sys.exit(f"no such street folder: {base}")
    walks = ([base / args.walk] if args.walk
             else sorted(p for p in base.iterdir() if p.is_dir()))
    w = walks[0]

    files, seqs = {}, {}
    for p in sorted(w.glob("*.jpg")):
        m = NAME.match(p.name)
        if m:
            files[m.group(2)] = p
            seqs[m.group(2)] = int(m.group(1))
    if not files:
        sys.exit(f"no frames in {w}")

    nf = pd.read_csv(PROC / "nodes.csv")
    nf = nf[nf.node_id.isin(files)].copy()
    nf["_seg"] = (nf.source_id.astype(str).str.rsplit("_", n=1).str[0]
                  if "source_id" in nf.columns and nf.source_id.notna().any()
                  else nf.get("chain", "all"))
    nf["_seq"] = nf.node_id.map(seqs)
    seg = nf._seg.value_counts().idxmax()
    nf = nf[nf._seg == seg].sort_values("_seq")
    print(f"{w.name}: corridor {seg}, {len(nf)} nodes")

    rt = load(args.ratings or (RES / "tables" / "sim_vlm_180_placeless.csv"))
    if rt is None:
        rt = load(RES / "tables" / "sim_vlm_v3.csv")
    de = load(args.descriptions or (RES / "tables" / "vlm_descriptions_180.csv"))
    ca = load(args.calc or (RES / "tables" / "vlm_calculations.csv"))

    def per_node(d, cols):
        if d is None:
            return {}
        have = [c for c in cols if c in d.columns]
        if not have or "node_id" not in d.columns:
            return {}
        g = d.groupby("node_id")[have]
        num = [c for c in have if pd.api.types.is_numeric_dtype(d[c])]
        out = {}
        for nid, sub in g:
            rec = {}
            for c in have:
                v = sub[c]
                rec[c] = (float(v.mean()) if c in num
                          else str(v.dropna().iloc[0]) if v.notna().any() else None)
            out[nid] = rec
        return out

    R = per_node(rt, FIELDS)
    D = per_node(de, QCOLS)
    C = per_node(ca, list(DIM) + ["M", "M_noA"])

    out = args.out or (RES / "figures" / f"walk_{args.street}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = []
    for r in nf.itertuples():
        p = files[r.node_id]
        rel = os.path.relpath(p, out.parent).replace("\\", "/")
        steps.append({"node": r.node_id, "img": rel,
                      "street": str(getattr(r, "osm_name", None)
                                    or getattr(r, "street_name", "")),
                      "r": R.get(r.node_id, {}), "d": D.get(r.node_id, {}),
                      "c": C.get(r.node_id, {})})
    have_q = sum(1 for s in steps if s["d"])
    print(f"  ratings on {sum(1 for s in steps if s['r'])} nodes, "
          f"descriptions on {have_q}, M on {sum(1 for s in steps if s['c'])}")

    page = TEMPLATE.replace("__DATA__", json.dumps(steps))
    page = page.replace("__TITLE__", html.escape(f"{args.street.replace('_',' ')}"))
    page = page.replace("__AREA__", html.escape(area))
    page = page.replace("__FIELDS__", json.dumps(FIELDS))
    page = page.replace("__DIMS__", json.dumps(DIM))
    page = page.replace("__QCOLS__", json.dumps(QCOLS))
    out.write_text(page, encoding="utf-8")
    print(f"\nwrote {out}")
    print("  open it from the repository -- it points at the JPEGs on disk and")
    print("  must not be uploaded: Street View imagery is not redistributable.")


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
:root{--bg:#0e0f12;--fg:#e8e6e1;--mut:#9a9aa2;--line:#23262b;--acc:#5fbf6a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 "Segoe UI",system-ui,sans-serif}
header{display:flex;align-items:baseline;gap:14px;padding:12px 18px;
       border-bottom:1px solid var(--line)}
header h1{font-size:17px;margin:0;font-weight:600}
header .a{color:var(--mut);font-size:13px}
main{display:grid;grid-template-columns:1fr 380px;gap:18px;padding:18px;
     align-items:start}
#view{width:100%;border:1px solid var(--line);border-radius:3px;display:block}
.bar{height:26px;background:#171a1f;border-radius:2px;position:relative;
     overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;background:var(--acc);opacity:.75}
.row{display:grid;grid-template-columns:150px 1fr 34px;gap:8px;
     align-items:center;margin:5px 0}
.row span{color:var(--mut);font-size:12px}
.row b{font-weight:600;text-align:right;font-variant-numeric:tabular-nums}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
   color:var(--mut);margin:18px 0 8px;font-weight:600}
.q{margin:0 0 12px}
.q dt{color:var(--mut);font-size:11px;text-transform:uppercase;
      letter-spacing:.07em;margin-bottom:2px}
.q dd{margin:0 0 10px;font-size:13px}
#ctl{display:flex;gap:8px;align-items:center;margin-top:10px}
button{background:#171a1f;color:var(--fg);border:1px solid var(--line);
       border-radius:3px;padding:6px 12px;font-size:13px;cursor:pointer}
button:hover{border-color:#3a3f46}
#pos{color:var(--mut);font-variant-numeric:tabular-nums}
input[type=range]{flex:1}
.M{font-size:30px;font-weight:600;font-variant-numeric:tabular-nums}
.none{color:#6b7078;font-style:italic}
</style>
<header>
  <h1>__TITLE__</h1><span class="a">__AREA__</span>
  <span class="a" id="node"></span>
</header>
<main>
  <div>
    <img id="view" alt="street view">
    <div id="ctl">
      <button id="prev">&larr;</button>
      <button id="play">play</button>
      <button id="next">&rarr;</button>
      <input type="range" id="slider" min="0" value="0">
      <span id="pos"></span>
    </div>
  </div>
  <aside>
    <h2>composite</h2>
    <div class="M" id="M">--</div>
    <div id="dims"></div>
    <h2>ratings, 1 to 7</h2>
    <div id="fields"></div>
    <h2>what the model says</h2>
    <dl class="q" id="qual"></dl>
  </aside>
</main>
<script>
const STEPS=__DATA__, FIELDS=__FIELDS__, DIMS=__DIMS__, QCOLS=__QCOLS__;
let i=0, timer=null;
const $=id=>document.getElementById(id);
$("slider").max=STEPS.length-1;

function draw(){
  const s=STEPS[i];
  $("view").src=s.img;
  $("node").textContent=s.node+(s.street?"  ·  "+s.street:"");
  $("pos").textContent=(i+1)+" / "+STEPS.length;
  $("slider").value=i;

  const m = s.c.M_noA ?? s.c.M;
  $("M").textContent = m==null ? "--" : m.toFixed(3);
  $("dims").innerHTML = Object.entries(DIMS).map(([k,label])=>{
    const v=s.c[k];
    return v==null ? "" :
      `<div class="row"><span>${label}</span>
        <div class="bar"><i style="width:${Math.max(0,Math.min(1,v))*100}%"></i></div>
        <b>${v.toFixed(2)}</b></div>`;
  }).join("");

  $("fields").innerHTML = FIELDS.map(f=>{
    const v=s.r[f];
    if(v==null) return "";
    const pct=Math.max(0,Math.min(1,(v-1)/6))*100;
    return `<div class="row"><span>${f.replace(/_/g," ")}</span>
      <div class="bar"><i style="width:${pct}%"></i></div>
      <b>${v.toFixed(1)}</b></div>`;
  }).join("") || '<div class="none">no ratings for this node</div>';

  const q=QCOLS.filter(c=>s.d[c]).map(c=>
    `<dt>${c}</dt><dd>${s.d[c]}</dd>`).join("");
  $("qual").innerHTML = q || '<div class="none">not described yet</div>';
}
function go(n){ i=(n+STEPS.length)%STEPS.length; draw(); }
$("prev").onclick=()=>go(i-1);
$("next").onclick=()=>go(i+1);
$("slider").oninput=e=>go(+e.target.value);
$("play").onclick=function(){
  if(timer){clearInterval(timer);timer=null;this.textContent="play";}
  else{timer=setInterval(()=>go(i+1),900);this.textContent="pause";}
};
addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft")go(i-1);
  if(e.key==="ArrowRight")go(i+1);
});
draw();
</script>
"""

if __name__ == "__main__":
    main()
