import base64, pathlib, numpy as np, pandas as pd
R = pathlib.Path(r"C:\Users\lumic\Documents\murrayhill\results")
D = pathlib.Path(r"C:\Users\lumic\Documents\murrayhill\data\processed")
def b64(p): return base64.b64encode(pathlib.Path(p).read_bytes()).decode()

FRAME = b64(R/"figures"/"frame_audit.png")
MAP   = b64(R/"figures"/"sim_dwell_map.png")
AXON  = b64(R/"figures"/"figure_axonometric_sim_layers.jpg")
SAMP  = {f.name.replace(".jpg", ".png"): b64(f)
         for f in sorted((R/"figures"/"sim_samples"/"web").glob("sample_*.jpg"))}
meta  = pd.read_csv(R/"tables"/"sim_samples.csv")
idx   = pd.read_csv(D/"sim_index.csv")
_st   = pd.read_csv(R/"tables"/"nodes_per_street.csv")
_bl   = pd.read_csv(R/"tables"/"nodes_per_block.csv")

STREETROWS = "\n".join(
  f'<tr><td>{r.osm_name}</td><td class="ty">{r.typology.replace("_"," ")}</td>'
  f'<td class="num">{r.nodes}</td><td class="num">{r.analytic}</td>'
  f'<td class="num">{"&mdash;" if r.excluded == 0 else r.excluded}</td>'
  f'<td class="num">{_bl[_bl.street == r.osm_name].shape[0]}</td></tr>'
  for _, r in _st.iterrows())

BLOCKROWS = "\n".join(
  f'<tr><td><code>{r.block}</code></td><td>{r.between}</td>'
  f'<td class="num">{r.nodes}</td><td class="num">{r.analytic}</td>'
  f'<td class="num">{"&mdash;" if r.excluded == 0 else r.excluded}</td></tr>'
  for _, r in _bl.iterrows())
_NBLOCK, _MEDBLOCK = len(_bl), int(_bl.nodes.median())

# ---- the 9 x 5 cross-street matrix, laid out as the map
_mc = pd.read_csv(R/"tables"/"block_matrix_counts.csv", index_col=0)
_ms = pd.read_csv(R/"tables"/"block_matrix_sim.csv", index_col=0)

def _ramp(v, stops):
    """Interpolate a hex ramp at v in [0,1]."""
    if v != v:
        return "transparent"
    v = min(max(v, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(v), len(stops) - 2)
    f = v - i
    a, b = stops[i].lstrip("#"), stops[i + 1].lstrip("#")
    rgb = [round(int(a[k:k+2], 16) + f * (int(b[k:k+2], 16) - int(a[k:k+2], 16)))
           for k in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(rgb)

def _matrix(df, stops, fmt, lo=None, hi=None):
    vals = df.values.astype(float)
    lo = np.nanmin(vals) if lo is None else lo
    hi = np.nanmax(vals) if hi is None else hi
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []
    for r, row in df.iterrows():
        cells = []
        for v in row.values.astype(float):
            t = (v - lo) / (hi - lo) if hi > lo else 0.5
            bg = _ramp(t, stops)
            fg = "#10161C" if t > 0.55 else "#E8EDF2"
            cells.append(f'<td style="background:{bg};color:{fg}">{fmt.format(v)}</td>')
        body.append(f'<tr><th class="rh">{r}</th>{"".join(cells)}</tr>')
    return (f'<table class="mx"><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')

MX_COUNTS = _matrix(_mc, ["#151C24", "#2b4a5c", "#4f8fa6", "#9fd8e0", "#e8f7fa"], "{:.0f}")
MX_SIM    = _matrix(_ms, ["#1a0b28", "#6b1f6e", "#c33f6b", "#f2864f", "#fce6a8"], "{:.3f}")

# Captions are generated, not written. The sampled nodes change whenever the
# frame or the study-area filter changes, so a hand-written caption silently
# starts describing a different photograph.
_TYPO = {"mid_block": "rowhouse mid-block", "avenue_canyon": "avenue canyon",
         "avenue_secondary": "secondary avenue", "other": "outside the three typologies"}
_ixm = idx.set_index("node_id")
rows = []
for _, r in meta.iterrows():
    v = _ixm.loc[r.node_id] if r.node_id in _ixm.index else None
    bits = [f"GVI {r.GVI:.2f}", f"VEI {r.VEI:.3f}"]
    if v is not None:
        bits.append(f"<strong>SIM {v.SIM:.3f}</strong>")
        bits += [f"G {v.G:.3f}", f"M {v.M:.3f}", f"P {v.P:.3f}"]
    rows.append(f"""
    <figure class="sample">
      <img src="data:image/jpeg;base64,{SAMP[r.file]}" alt="{r.street} 180-degree segmentation sample">
      <figcaption><span class="sname">{r.street}</span> &middot; <code>{r.node_id}</code> &middot;
      {_TYPO.get(r.typology, r.typology)} &middot; street axis {r.axis_deg}&deg; &middot;
      left&nbsp;to&nbsp;right: original, overlay, mask<br>{" &middot; ".join(bits)}</figcaption>
    </figure>""")
SAMPLES = "\n".join(rows)

_ix = idx.set_index("node_id")
WORKED = [(r.street, r.node_id,
           float(_ix.loc[r.node_id, "G"]), float(_ix.loc[r.node_id, "M"]),
           float(_ix.loc[r.node_id, "P"]), float(_ix.loc[r.node_id, "SIM"]))
          for _, r in meta.iterrows() if r.node_id in _ix.index]
WORKEDROWS = "\n".join(
  f'<tr><td>{s}</td><td><code>{n}</code></td><td class="num g">{g:.3f}</td>'
  f'<td class="num m">{mm:.3f}</td><td class="num p">{p:.3f}</td>'
  f'<td class="num"><strong>{v:.3f}</strong></td></tr>'
  for s, n, g, mm, p, v in WORKED)

typ = idx.merge(pd.read_csv(D/"metrics.csv")[["node_id","typology","osm_name"]], on="node_id")
tt = typ.groupby("typology")[["G","M","P","SIM"]].median().round(3)
TYPROWS = "\n".join(
  f"<tr><td>{i.replace('_',' ')}</td><td>{r.G:.3f}</td><td>{r.M:.3f}</td>"
  f"<td><strong>{r.P:.3f}</strong></td><td>{r.SIM:.3f}</td></tr>"
  for i, r in tt.iterrows())

# Curated from the checkpoint's 150 labels: 39 are already in sim.groups,
# and most of the remaining 111 are indoor (bed, sink, chandelier). These are
# the ones that could serve a construct the manuscript already names.
CANDIDATES = [
 ("high", "signage &amp; commerce", "poster (100), trade name (123), bulletin board (144), booth (88)",
  "Extends articulation beyond <code>signboard</code>. Li et al. tie commercial signage to place identity, and <code>booth</code> catches newsstands and kiosks &mdash; active ground-floor frontage the current term misses entirely."),
 ("high", "visual anchors", "sculpture (132), fountain (104), flag (149)",
  "Lynch's imageability rests on distinctive landmarks. These are the only ADE20K classes that isolate one, and none is currently read."),
 ("high", "street furniture", "streetlight (87), pole (93), ashcan (138), traffic light (136)",
  "Furniture density is the standard proxy for pedestrian-realm investment. Splits two ways: streetlight and ashcan read as amenity, pole and traffic light as clutter and traffic exposure."),
 ("high", "unpaved ground", "path (52), earth (13), land (94)",
  "<code>green_ground</code> currently divides by road + sidewalk only, so an unpaved verge or planted strip is invisible in the denominator."),
 ("medium", "multi-level networks", "escalator (96), bridge (61), stairway (59)*",
  "Yoos &amp; James's parallel-cities framework is cited in the introduction but nothing measures grade separation. <code>bridge</code> would also flag the East 36th case rather than requiring an OSM tag."),
 ("medium", "the waterfront", "water (21), river (60), pier (140), boat (76)",
  "Section 4.1 is about the East River esplanade. No current term can see water, so the greenification argument has no pixel support."),
 ("medium", "topography", "hill (68), rock (34), mountain (16)",
  "The Inclenberg drumlin opens the site description. Unlikely to fire in a built canyon, but cheap to record."),
 ("low", "temporary occupation", "tent (114), stage (101), grandstand (51), awning (86)*",
  "Markets, scaffolding sheds and event structures. Worth watching precisely because the scaffolding detector benchmarks at AUC 0.51."),
 ("low", "activity traces", "animal (126), plaything (108), bag (115), bicycle (127)*",
  "Dogs and play equipment are weak but real dwell signals. People are already counted; what they are doing is not."),
]
CANDROWS = "\n".join(
  f'<tr><td><span class="chip {"none" if pr=="high" else "part" if pr=="medium" else "have"}">{pr}</span></td>'
  f'<td><strong>{grp}</strong></td><td class="mono">{cls}</td><td>{why}</td></tr>'
  for pr, grp, cls, why in CANDIDATES)

import sys
sys.path.insert(0, r"C:\Users\lumic\Documents\murrayhill\src")
from common import bin_mask
_z = np.load(D/"sim_profiles.npz", allow_pickle=True)
_rows = [str(r) for r in _z["__rows__"]]; _R = {r: i for i, r in enumerate(_rows)}
_m = pd.read_csv(D/"metrics.csv")
_m = _m[_m.node_id.isin(set(idx.node_id))]   # study area only
_ax = dict(zip(_m.node_id, _m.street_axis_deg))
_tot = {r: 0.0 for r in _rows if r != "weight"}; _k = 0
for _nid in (x for x in _z.files if x != "__rows__"):
    _c = _ax.get(_nid)
    if _c is None or np.isnan(_c): continue
    _a = _z[_nid]; _mk = bin_mask(_c, 180.0); _W = _a[_R["weight"]][_mk].sum()
    if _W <= 0: continue
    _k += 1
    for _r in _tot: _tot[_r] += _a[_R[_r]][_mk].sum()/_W
SHARES = sorted(((r.replace("_", " "), 100*v/_k) for r, v in _tot.items()),
                key=lambda kv: -kv[1])
_MX = SHARES[0][1]
SHAREROWS = "\n".join(
  f'<tr><td>{n}</td><td class="num">{v:.2f}%</td>'
  f'<td><div class="bar" style="width:{max(v/_MX*100,0.4):.1f}%"></div></td></tr>'
  for n, v in SHARES)

HTML = f"""<title>Street Interface Matrix</title>
<!-- Shareable but not searchable: this is unpublished work, and a page that
     turns up in results is hard to take back once cached. -->
<meta name="robots" content="noindex, nofollow">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600;7..72,700&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#F4F5F2; --ink:#10161C; --body:#2C333B; --slate:#6B7480;
  --rule:#D8DCD6; --panel:#EBEEE9; --spruce:#2F5D50; --amber:#A85F27;
  --brick:#9E4038; --ok:#2F5D50; --pcol:#3F6CB2;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0D1117; --ink:#E8EDF2; --body:#BFC8D2; --slate:#8B949E;
    --rule:#242C36; --panel:#151C24; --spruce:#7FBFA6; --amber:#E0975A;
    --brick:#E0796E; --ok:#7FBFA6; --pcol:#5296C1;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0D1117; --ink:#E8EDF2; --body:#BFC8D2; --slate:#8B949E;
  --rule:#242C36; --panel:#151C24; --spruce:#7FBFA6; --amber:#E0975A;
  --brick:#E0796E; --ok:#7FBFA6; --pcol:#5296C1;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--paper); color:var(--body); margin:0;
  font-family:"Public Sans",system-ui,sans-serif; font-size:16.5px; line-height:1.65;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1120px; margin:0 auto; padding:0 28px 96px; }}
.measure {{ max-width:68ch; }}
header {{ border-bottom:2px solid var(--ink); padding:64px 0 26px; margin-bottom:40px; }}
.eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--slate); margin:0 0 14px; }}
h1 {{ font-family:Literata,Georgia,serif; font-weight:700; font-size:clamp(30px,4.6vw,46px);
  line-height:1.12; color:var(--ink); margin:0 0 16px; text-wrap:balance; letter-spacing:-.015em; }}
.standfirst {{ font-size:18.5px; max-width:60ch; margin:0; }}
h2 {{ font-family:Literata,Georgia,serif; font-weight:600; font-size:25px; color:var(--ink);
  margin:64px 0 6px; text-wrap:balance; letter-spacing:-.01em; }}
h2 .num {{ font-family:"IBM Plex Mono",monospace; font-size:13px; color:var(--amber);
  font-weight:500; display:block; margin-bottom:8px; letter-spacing:.08em; }}
h3 {{ font-family:Literata,Georgia,serif; font-weight:600; font-size:18px; color:var(--ink); margin:34px 0 4px; }}
p {{ margin:14px 0; }}
code {{ font-family:"IBM Plex Mono",monospace; font-size:.875em; background:var(--panel);
  padding:.12em .38em; border-radius:3px; }}
figure {{ margin:34px 0; }}
figure img {{ width:100%; display:block; border:1px solid var(--rule); border-radius:2px; }}
figcaption {{ font-size:13.5px; color:var(--slate); margin-top:10px; max-width:76ch; }}
.sample {{ margin:14px 0 30px; }}
.sample img {{ image-rendering:auto; }}
.sample figcaption {{ margin-top:8px; }}
.striphead {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px;
  margin:22px 0 2px; font-family:"IBM Plex Mono",monospace; font-size:10.5px;
  letter-spacing:.11em; text-transform:uppercase; color:var(--slate); }}
.sname {{ color:var(--ink); font-weight:600; }}
.striphead {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:26px 0 6px;
  font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--slate); }}
.tablewrap {{ overflow-x:auto; margin:26px 0; border:1px solid var(--rule); border-radius:2px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; min-width:620px; }}
th {{ text-align:left; font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--slate); padding:12px 14px; border-bottom:1px solid var(--rule);
  font-weight:500; background:var(--panel); }}
td {{ padding:10px 14px; border-bottom:1px solid var(--rule); vertical-align:middle; }}
td.num {{ font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }}
tr:last-child td {{ border-bottom:none; }}
.scroll {{ max-height:460px; overflow-y:auto; }}
/* Stacked, not side by side: two five-column matrices in one row shrink
   each below a readable size on a phone. Full width each is legible. */
.mxwrap {{ display:grid; grid-template-columns:1fr; gap:34px; margin:26px 0; }}
.mxwrap > div {{ overflow-x:auto; }}
/* Fixed layout so both matrices share a column grid. Left to itself the
   count table sizes its columns to two digits and the SIM table to five,
   and the two stop lining up when stacked. */
table.mx {{ border-collapse:separate; border-spacing:3px; font-size:15px;
  font-variant-numeric:tabular-nums; width:100%; min-width:420px;
  table-layout:fixed; }}
table.mx th:first-child {{ width:76px; }}
table.mx th {{ background:none; border:none; padding:7px 6px; font-size:11px;
  letter-spacing:.04em; color:var(--slate); text-transform:none; text-align:center;
  font-weight:500; }}
table.mx th.rh {{ text-align:right; padding-right:11px; font-size:13px; color:var(--ink);
  font-family:"IBM Plex Mono",monospace; white-space:nowrap; }}
table.mx td {{ padding:13px 6px; text-align:center; border:none; border-radius:3px;
  font-family:"IBM Plex Mono",monospace; }}
td.ty {{ color:var(--slate); font-size:13px; }}
.bar {{ height:9px; background:var(--spruce); border-radius:1px; min-width:2px; }}
.chip {{ display:inline-block; font-family:"IBM Plex Mono",monospace; font-size:10.5px;
  letter-spacing:.06em; text-transform:uppercase; padding:3px 8px; border-radius:2px;
  border:1px solid currentColor; white-space:nowrap; }}
.chip.have {{ color:var(--ok); }} .chip.part {{ color:var(--amber); }} .chip.none {{ color:var(--brick); }}
.master {{ font-family:Literata,Georgia,serif; font-size:clamp(22px,3.2vw,32px); color:var(--ink);
  text-align:center; padding:34px 20px; margin:30px 0 0; background:var(--panel);
  border:1px solid var(--rule); border-bottom:none; overflow-x:auto; }}
.master .v {{ font-style:italic; }}
.master .op {{ color:var(--slate); margin:0 .28em; }}
.master .w {{ font-family:"IBM Plex Mono",monospace; font-size:.62em; color:var(--slate);
  vertical-align:.12em; margin-right:.12em; }}
.dims {{ display:grid; grid-template-columns:1fr; border:1px solid var(--rule); }}
.dim {{ padding:20px 24px; border-top:1px solid var(--rule); border-left:4px solid transparent; }}
.dim.gd {{ border-left-color:var(--spruce); }}
.dim.md {{ border-left-color:var(--amber); }}
.dim.pd {{ border-left-color:var(--pcol); }}
.dl {{ font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--slate); margin:0 0 10px; }}
.de {{ font-family:Literata,Georgia,serif; font-size:19px; color:var(--ink); margin:0 0 8px;
  overflow-x:auto; }}
.de .w {{ font-family:"IBM Plex Mono",monospace; font-size:.68em; color:var(--slate); }}
.dn {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--slate); margin:0; }}
.hat {{ border-top:1.5px solid currentColor; padding-top:1px; }}
.satbox {{ border:1px solid var(--rule); border-top:none; padding:20px 24px; background:var(--panel);
  margin-bottom:26px; }}
.g {{ color:var(--spruce); font-weight:600; font-style:italic; }}
.m {{ color:var(--amber); font-weight:600; font-style:italic; }}
.p {{ color:var(--pcol); font-weight:600; font-style:italic; }}
td.g {{ color:var(--spruce); }} td.m {{ color:var(--amber); }} td.p {{ color:var(--pcol); }}
.lbl {{ color:var(--slate); font-size:12px; font-family:"IBM Plex Mono",monospace; }}
.mono {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px; line-height:1.5; }}
.fn {{ font-size:12.5px; color:var(--slate); margin:-14px 0 0; }}
.note {{ border-left:3px solid var(--amber); background:var(--panel); padding:16px 20px;
  margin:26px 0; font-size:15px; }}
.note strong {{ color:var(--ink); }}
.note.stop {{ border-left-color:var(--brick); }}
.note.good {{ border-left-color:var(--spruce); }}
ul {{ padding-left:20px; }} li {{ margin:7px 0; }}
.kv {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); margin:26px 0; }}
.kv div {{ background:var(--paper); padding:15px 17px; }}
.kv dt {{ font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--slate); }}
.kv dd {{ margin:6px 0 0; font-size:22px; font-family:Literata,serif; color:var(--ink);
  font-variant-numeric:tabular-nums; }}
footer {{ margin-top:80px; padding-top:22px; border-top:1px solid var(--rule);
  font-size:13px; color:var(--slate); }}
</style>

<div class="wrap">
<header class="measure">
  <p class="eyebrow">Murray Hill v12 &middot; 20 August 2026 &middot; run on MIKE_PC</p>
  <h1>Street Interface Matrix</h1>
  <p class="standfirst">Twelve segmented classes across 2,940 frames, a dwell index built only
  from terms the segmenter can actually deliver, and what the pixel shares say about the
  argument.</p>
</header>

<div class="kv">
  <div><dt>Frames segmented</dt><dd>2,940</dd></div>
  <div><dt>Nodes indexed</dt><dd>{len(idx)}</dd></div>
  <div><dt>Classes</dt><dd>12</dd></div>
  <div><dt>GPU time</dt><dd>10m</dd></div>
  <div><dt>Streets</dt><dd>17</dd></div>
</div>

<div class="measure">
<div class="note good">
<strong>The pipeline is platform-independent.</strong> Before the frame changed, re-segmenting
the v12 imagery on this PC reproduced the Mac's profiles to <strong>0.0032 GVI points</strong>,
with VEI identical to four decimals and the weight row matching at
5.2&times;10<sup>&minus;18</sup> &mdash; across a different OS, a different GPU architecture
(CUDA against Apple MPS) and a different transformers version. Worth a line in the methods.
Runtime fell from 35&nbsp;min&ndash;6&nbsp;hr to under ten minutes, which is what made replacing
the frame and re-fetching 2,940 images a same-day operation rather than a week's work.
</div>

<h2><span class="num">01</span>The sampling frame</h2>
<p>The <strong>v13 frame</strong> holds 766 nodes at 20&nbsp;m along 17 streets, of which
<strong>657 fall inside the defined study area</strong> and 634 carry an along-street window.
It replaces the OSM-derived v12 frame: the East 36th bridge block is covered and Tudor City
Place is included.</p>
</div>

<figure>
  <img src="data:image/png;base64,{FRAME}" alt="Frame map of Murray Hill">
  <figcaption><strong>Left:</strong> the v13 frame. Analytic sample in green, the 78
  tunnel-flagged nodes in blue, and the 31 date-filtered exclusions as red crosses. The
  exclusions are scattered rather than contiguous, which is what separates a capture-date filter
  from a coverage failure &mdash; the East 36th bridge block that was a 257&nbsp;m hole in v12 is
  now covered. <strong>Right:</strong> GVI, where Park Avenue's planted median is the one bright
  line.</figcaption>
</figure>

<div class="measure">
<ul>
  <li><strong>31 nodes &mdash; capture date.</strong> <code>capture.target: "2026-04"</code>
  demands an exact month; all 735 usable nodes come from that single capture, so season is held
  constant across the whole frame.</li>
  <li><strong>2 nodes &mdash; empty profile.</strong> <code>n00024</code> and <code>n00450</code>
  returned zero pixels in every class. Dropped by rule rather than by id, so the check catches
  future cases automatically.</li>
  <li><strong>78 nodes &mdash; tunnel, flagged not removed.</strong> Park Avenue's tunnel segment
  (57) and Tunnel Exit Street (21) carry <code>is_tunnel</code>. The v12 frame excluded these by
  OSM tag; v13 names them, so whether they count as pedestrian streetscape is now a one-line
  filter rather than a re-fetch.</li>
  <li><strong>14 nodes &mdash; uncertain street axis.</strong> 1st and Park Avenue carry parallel
  service roadways ~24&nbsp;m apart, close enough that a local fit can straddle both. Their
  180&deg; window may be rotated.</li>
</ul>

<h3>Nodes per street</h3>
</div>
<div class="tablewrap">
<table>
<thead><tr><th>street</th><th>typology</th><th style="text-align:right">nodes</th>
<th style="text-align:right">analytic</th><th style="text-align:right">excluded</th>
<th style="text-align:right">blocks</th></tr></thead>
<tbody>
{STREETROWS}
</tbody>
</table>
</div>

<div class="measure">
<h3>Nodes per block</h3>
<p>A block is the stretch between two consecutive crossings &mdash; East 41st between Madison and
Park is one built condition, between Park and Lexington another. The frame's chains do not encode
that, since a chain runs a street's whole length and crosses five avenues on the way. Crossings
are derived from the node geometry rather than from OSM, because a crossing with a street outside
the study area is not a boundary for this purpose.</p>
<p>Cross-streets read west to east, avenues north to south. Numbering runs consecutively over
blocks that hold nodes, so a missing stretch leaves no gap in the sequence.
<strong>{_NBLOCK} blocks, median {_MEDBLOCK} nodes each.</strong></p>
</div>
<div class="tablewrap scroll">
<table>
<thead><tr><th>block</th><th>between</th><th style="text-align:right">nodes</th>
<th style="text-align:right">analytic</th><th style="text-align:right">excluded</th></tr></thead>
<tbody>
{BLOCKROWS}
</tbody>
</table>
</div>

<div class="measure">
<h3>The cross-streets as a matrix</h3>
<p>Nine cross-streets by five blocks, oriented as the neighbourhood is: <strong>north at the top,
west at the left</strong>, so a cell sits where its block sits. Columns are the block number, so
the bottom-right cell is <code>34th_5</code> &mdash; reading
<strong>1 = Madison&ndash;Park, 2 = Park&ndash;Lex, 3 = Lex&ndash;3rd, 4 = 3rd&ndash;2nd,
5 = 2nd&ndash;1st</strong>. Only the six avenues bound a block
here &mdash; Tunnel Exit Street and Tudor City Place are partial north-south runs that cross some
cross-streets and not others, which made the block set ragged and no rectangular matrix possible.
They stay in the frame and in every other count; they are just not treated as edges.</p>
</div>

<div class="mxwrap">
  <div>
    <p class="dl">node count</p>
    {MX_COUNTS}
  </div>
  <div>
    <p class="dl">mean SIM</p>
    {MX_SIM}
  </div>
</div>

<div class="measure">
<div class="note good">
<strong>The gradient runs north to south.</strong> Read the right-hand matrix down any column:
East 42nd sits at 0.16&ndash;0.27 while East 34th to 39th run 0.28&ndash;0.49. The northern
cross-streets are the Grand Central commercial district; the southern ones are the 1847 covenant
rowhouse blocks. The highest single cell is <strong>East 36th between Park and Lexington at
0.493</strong>, the lowest East 42nd between Lexington and 3rd at 0.164.
<br><br>
Node counts are near-uniform by contrast &mdash; 7 to 12 per block, driven by block length rather
than anything about the streetscape &mdash; which is what makes the SIM matrix readable as
signal.
</div>
</div>

<div class="measure">
<div class="note">
<strong>109 nodes sit outside the study area and are excluded, on two rules.</strong>
<br><br>
<strong>One sampled roadway per street.</strong> The v13 frame samples Park Avenue three times
&mdash; <code>Park_Ave_East</code>, <code>Park_Ave_West</code> and
<code>Park_Ave_Tunnel_Segment</code> &mdash; putting 132 nodes on one avenue against 34 on
Madison, entering the same street once per carriageway. Only <code>Park_Ave_East</code> is kept,
which makes Park comparable with every other avenue. Note this is a rule about duplication, not
about tunnels: <strong>Tunnel Exit Street stays in</strong>, because it is its own street with a
single roadway and duplicates nothing.
<br><br>
<strong>Inside the boundary streets.</strong> A further 21 nodes sit past the outermost crossing
on their street &mdash; East 42nd east of 1st Avenue, the avenues north of 42nd &mdash; so they
belong to no block and have no matching condition opposite.
<br><br>
Nothing is deleted. <code>in_study</code> is a column on the frame, so both decisions stay
inspectable and reversible.
</div>

<h2><span class="num">02</span>What the segmenter actually finds</h2>
<p>Mean class share across the 180&deg; along-street window, all {len(idx)} indexed nodes. This is the number that
should decide which terms the index can carry.</p>
</div>

<div class="tablewrap">
<table>
<thead><tr><th>class</th><th style="text-align:right">share</th><th style="width:52%"></th></tr></thead>
<tbody>
{SHAREROWS}
</tbody>
</table>
</div>

<div class="measure">
<div class="note stop">
<strong>Three findings that bear directly on the argument.</strong>
<ul>
<li><strong>Eye-level greenery is 0.30%; canopy is 4.06%.</strong> Eye level is about a
fourteenth of all vegetation here. The manuscript's central claim is that eye-level greenery
drives dwell more than overhead canopy &mdash; in Murray Hill there is barely any eye-level
greenery to do the driving. That reframes the argument rather than defeating it.</li>
<li><strong>The affordance classes are near-empty.</strong> rest 0.07%, shelter 0.03%, soft buffer
0.00%. Individually they are zero at 68%, 53% and 78% of nodes. No term weighted on one of those
alone is defensible.</li>
<li><strong>Road is 40%, sidewalk 2.6%.</strong> Nodes sit on the street centreline and Street View
is captured from a vehicle, so the camera stands in the carriageway, not on the pavement. For a
study of pedestrian dwell this should be stated plainly.</li>
</ul>
</div>

<h2><span class="num">03</span>Categories the segmenter can deliver</h2>
<p>Each manuscript term against the labels this checkpoint returns.</p>
</div>

<div class="tablewrap">
<table>
<thead><tr><th>manuscript term</th><th>ADE20K classes</th><th>status</th><th>note</th></tr></thead>
<tbody>
<tr><td><code>GVI_eye</code></td><td>tree, grass, plant, flower, palm &mdash; below horizon</td><td><span class="chip have">measured</span></td><td>Split before the azimuthal collapse; exact to 0.000e+00.</td></tr>
<tr><td><code>P_natural / P_asphalt</code></td><td>veg &divide; (veg + road + sidewalk)</td><td><span class="chip part">rewritten</span></td><td>Bounded form of the same question.</td></tr>
<tr><td><code>SVF</code></td><td>sky</td><td><span class="chip part">band only</span></td><td>&plusmn;45&deg; only; must stay <code>SVF_band</code>.</td></tr>
<tr><td><code>P_signboard + P_detail</code></td><td>signboard, windowpane, door</td><td><span class="chip part">proxy</span></td><td>No architectural-detail class; openings stand in.</td></tr>
<tr><td><code>P_sidewalk / P_paver</code></td><td>sidewalk &divide; (sidewalk + road)</td><td><span class="chip none">was broken</span></td><td>ADE20K's class is &ldquo;sidewalk, pavement&rdquo; &mdash; it divided a class by itself.</td></tr>
<tr><td><code>GFAPI</code></td><td>windowpane, door</td><td><span class="chip part">proxy</span></td><td>Standard active-frontage proxy.</td></tr>
<tr><td>rest + shelter + soft buffer</td><td>bench, seat, stairs, awning, canopy, column, pot</td><td><span class="chip part">pooled</span></td><td>Pooled into one sparse term; see below.</td></tr>
<tr><td>hard barrier</td><td>fence, railing, bannister</td><td><span class="chip have">measured</span></td><td>ADE20K <code>wall</code> is the facade &mdash; excluded.</td></tr>
<tr><td>soft buffer alone</td><td>pot</td><td><span class="chip none">inadequate</span></td><td>0.00% share. Planters and hedges have no class.</td></tr>
</tbody>
</table>
</div>

<div class="measure">
<h3>Classes available but unused</h3>
<p>The checkpoint returns <strong>150 labels; 39 are in <code>sim.groups</code></strong>. Most of
the remaining 111 are indoor &mdash; bed, sink, chandelier &mdash; and will never fire on a street
frame. These are the ones that could serve a construct the manuscript already names, ranked by how
directly they do so. Adding any of them is a config edit plus eight minutes of GPU.</p>
</div>

<div class="tablewrap">
<table>
<thead><tr><th>priority</th><th>would serve</th><th>ADE20K classes</th><th>why it matters here</th></tr></thead>
<tbody>
{CANDROWS}
</tbody>
</table>
</div>

<div class="measure">
<p class="fn">* already in use; listed because the group it would join is not.</p>
<div class="note">
<strong>The three worth adding first.</strong> <em>Signage and commerce</em> would roughly double
what articulation can see &mdash; at present it reads windows and doors but not a single shopfront
sign beyond <code>signboard</code>. <em>Visual anchors</em> is the only route to Lynch's
imageability, which the introduction leans on and nothing currently measures. <em>Unpaved ground</em>
fixes a denominator: <code>green_ground</code> divides by road and sidewalk alone, so a planted
verge is invisible to it.
<br><br>
The waterfront group is a different case. Section 4.1 argues the East River esplanade raised the
peripheral GVI, and no current class can see water at all &mdash; but the study area's eastern
boundary means few nodes would look at it. Worth adding for completeness, not for a result.
</div>
</div>

<div class="measure">
<h2><span class="num">04</span>The index</h2>
<p>Three dimensions, every term a bounded share of the along-street view. Weights are declared in
<code>config.yaml</code>, not fitted &mdash; with no measured dwell outcome there is nothing to fit
them against, and claiming otherwise is the one thing a reviewer could not let pass.</p>
</div>

<div class="master">
  <span class="v">SIM</span><sub>i</sub> <span class="op">=</span>
  <span class="w">0.34</span> <span class="g">G</span><sub>i</sub> <span class="op">+</span>
  <span class="w">0.33</span> <span class="m">M</span><sub>i</sub> <span class="op">+</span>
  <span class="w">0.33</span> <span class="p">P</span><sub>i</sub>
</div>

<div class="dims">
  <div class="dim gd">
    <p class="dl">Green / habitat</p>
    <p class="de"><span class="g">G</span><sub>i</sub> = <span class="w">0.60</span>&#8202;<span class="hat">green_eye</span>
      + <span class="w">0.40</span>&#8202;green_ground</p>
    <p class="dn">green_ground = veg &divide; (veg + road + sidewalk)</p>
  </div>
  <div class="dim md">
    <p class="dl">Morphological</p>
    <p class="de"><span class="m">M</span><sub>i</sub> = <span class="w">0.35</span>&#8202;<span class="hat">articulation</span>
      + <span class="w">0.40</span>&#8202;&Phi;(VEI) + <span class="w">0.25</span>&#8202;(1 &minus; <span class="hat">barrier</span>)</p>
    <p class="dn">&Phi;(VEI) = 4&#8202;VEI&#8202;(1 &minus; VEI) &mdash; inverted-U, peaks at VEI = 0.5</p>
  </div>
  <div class="dim pd">
    <p class="dl">Permeability</p>
    <p class="de"><span class="p">P</span><sub>i</sub> = <span class="w">0.50</span>&#8202;walkable
      + <span class="w">0.50</span>&#8202;<span class="hat">affordance</span></p>
    <p class="dn">walkable = sidewalk &divide; (sidewalk + road)<br>
      affordance = rest + shelter + soft_buffer &mdash; pooled, zero at 33% of nodes</p>
  </div>
</div>

<div class="satbox">
  <p class="dl">The hat: saturating transform on sparse terms</p>
  <p class="de"><span class="hat">x</span> = 1 &minus; e<sup>&minus;x / s<sub>0</sub></sup>
    <span class="lbl">&nbsp;&nbsp;s<sub>0</sub> = that term's own 75th percentile</span></p>
</div>

<div class="measure">
<h3>The same equation, evaluated</h3>
<p>The five sampled nodes above, carried through term by term.</p>
</div>
<div class="tablewrap">
<table>
<thead><tr><th>street</th><th>node</th><th style="text-align:right">G</th>
<th style="text-align:right">M</th><th style="text-align:right">P</th>
<th style="text-align:right">SIM</th></tr></thead>
<tbody>
{WORKEDROWS}
</tbody>
</table>
</div>

<div class="measure">
<div class="note">
<strong>Why the sparse terms saturate.</strong> Eye-level greenery has a median share of 0.001 and
the pooled affordance 0.0008. Against building at 0.43, a raw share would contribute nothing at
all. The transform is not a rescaling trick &mdash; it states a claim: <em>one bench makes a spot
dwellable; ten do not make it ten times more so.</em> Diminishing returns on affordance, which is
what ecological psychology predicts. It does mean the index is relative to this study area, and
that belongs in the methods.
</div>

<h3>Two departures from the manuscript's LaTeX</h3>
<ul>
  <li><strong>Enclosure enters once, through VEI, and non-monotonically.</strong> H/W runs
  0.05&ndash;10.07 here while SVF is [0,&nbsp;1]; summing them under weights totalling 1 lets the
  larger decide the answer. &Phi; encodes the manuscript's own claim &mdash; some enclosure defines
  the room, too much oppresses.</li>
  <li><strong>No denominator can vanish.</strong> Each is either total pixels or a pair that cannot
  both be zero in a street view, and nodes with an empty profile are dropped upstream.</li>
</ul>

<h2><span class="num">05</span>What the classes look like</h2>
<p>Five nodes stratified across the GVI range, each shown as the <strong>180&deg; along-street
view</strong> &mdash; the same window every metric is integrated over, not a single 90&deg; frame.
Four rectilinear frames are reprojected into one cylindrical strip centred on the street axis, and
the class map is carried through the identical mapping, so the mask corresponds pixel for pixel to
the photograph.</p>
<p>The shares alone cannot tell you whether &ldquo;rest 0.07%&rdquo; is a real scarcity of benches
or a detector that never fires. The pictures can.</p>
<div class="striphead"><span>original</span><span>overlay</span><span>mask</span></div>
</div>

{SAMPLES}

<div class="measure">
<h2><span class="num">06</span>The surface</h2>
<p>SIM at every node, interpolated to 1&nbsp;m along each street and never across one. A holdout
test puts linear interpolation at R&sup2;&nbsp;=&nbsp;0.84 for predicting an unseen midpoint,
against 0.76 for nearest-node and 0.54 for the street mean.</p>
</div>

<figure>
  <img src="data:image/png;base64,{MAP}" alt="SIM and permeability maps of Murray Hill">
  <figcaption>In the order of the equation: the composite, then each dimension carrying the
  colour it has in the formula. <strong>G</strong> is essentially Park Avenue &mdash; its planted
  median is the one bright line in an otherwise dark neighbourhood. <strong>M</strong> is close to
  uniform, which is what you would expect of a regular grid of similar canyons. <strong>P</strong>
  and the composite both light up the rowhouse mid-blocks and go dark on the avenues, so
  permeability is doing most of the work in separating the typologies. Ramps are clipped to the
  2nd&ndash;98th percentile; the full range is printed on each panel.</figcaption>
</figure>

<figure>
  <img src="data:image/jpeg;base64,{AXON}" alt="Exploded axonometric of the SIM layers over Murray Hill">
  <figcaption>The same quantities as strata rather than panels, over 1,285 extruded footprints.
  Bottom to top: built fabric, eye-level greenery, then <strong>G</strong>, <strong>M</strong>,
  <strong>P</strong> and the composite, bar height being the value at each 20&nbsp;m node and red
  the upper tail. The <strong>green eye</strong> layer is the one to look at &mdash; it is nearly
  flat where every other stratum has relief, which is the sparsity the saturating transform
  exists to handle: a median share of 0.0009 against building at 0.43.</figcaption>
</figure>

<div class="measure">
<h3>By typology</h3>
</div>
<div class="tablewrap">
<table>
<thead><tr><th>typology</th><th>green</th><th>morphological</th><th>permeability</th><th>SIM</th></tr></thead>
<tbody>
{TYPROWS}
</tbody>
</table>
</div>

<div class="measure">
<div class="note good">
<strong>The rowhouse mid-blocks score highest.</strong> East 36th (0.369), East 38th (0.363) and
East 37th (0.360) top the list; 3rd Avenue (0.215) and East 42nd (0.222) sit at the bottom. The
separation comes almost entirely from <strong>permeability</strong> &mdash; 0.197 on mid-blocks
against 0.056 on secondary avenues.
<br><br>
The mechanism is visible in sample 04: <em>stoops register as</em> <code>rest</code>, because
ADE20K reads them as stairs. The 1847 covenant blocks are built to the property line with stoops
at the façade, so the morphology that the manuscript argues produces dwell is the morphology that
puts seatable steps in the frame. That is the study's thesis appearing in the pixel shares
without being asked to.
</div>

<h2><span class="num">07</span>What is still open</h2>
<ul>
  <li><strong>No outcome variable.</strong> The index is a weighted restatement of pixel shares.
  Nothing here validates it against behaviour, and the weights cannot be fitted until something
  does.</li>
  <li><strong>Soft buffer is dead.</strong> 0.00% share. Planters and hedges have no ADE20K class
  and the flowerpot proxy finds nothing.</li>
  <li><strong>Scaffolding is unmeasured.</strong> The detector benchmarks at AUC 0.51 &mdash; a
  coin flip &mdash; and sidewalk sheds dominate the shelter term in Manhattan.</li>
  <li><strong>31 date-filtered nodes</strong> remain outside the sample, all with usable panoramas
  from other months. Admitting them would mix capture seasons into the variable being measured,
  so it is the wrong fix rather than a pending one.</li>
  <li><strong>The tunnel question is undecided.</strong> 78 nodes carry <code>is_tunnel</code> and
  are currently included. They score lower than the rest (SIM 0.262 against 0.302), mostly through
  permeability, so keeping them shifts the averages.</li>
  <li><strong>Moran's I for GVI reads +1.002</strong>, above its theoretical bound. Park Avenue's
  carriageways sit ~19&nbsp;m apart with near-identical values, so a 60&nbsp;m Euclidean
  neighbourhood treats each as its own neighbour. The weighting needs to follow the street
  network, not straight-line distance.</li>
  <li><strong>14 nodes have an uncertain street axis</strong> on 1st and Park Avenue, where
  parallel service roadways sit inside the local fit.</li>
</ul>
</div>

<footer class="measure">
Generated on MIKE_PC. 2,940 frames segmented with
<code>facebook/mask2former-swin-large-ade-semantic</code> on an RTX 3080 Ti.
Reproducible via <code>tools/s03_sim_profiles.py</code> then <code>tools/sim_dwell.py</code>.
</footer>
</div>
"""
# docs/index.html IS the GitHub Pages site. This previously wrote to a
# scratch directory from a session long gone, so regenerating the report
# left the published page untouched.
out = pathlib.Path(__file__).resolve().parent.parent / "docs" / "index.html"
out.write_text(HTML, encoding="utf-8")
print("wrote", out, f"{out.stat().st_size/1024/1024:.2f} MB")
