"""
Build a self-contained HTML dashboard from the pipeline outputs.

    python make_dashboard.py
    open results/dashboard.html          # macOS
    start results\\dashboard.html         # Windows

Everything -- data, figures, styling -- is inlined into one file. No
server, no internet, no Python needed to view it.

Reads whatever exists in data/processed and results/, and skips the rest,
so it works part-way through a run.

New in v12: the zone section is gone (replaced by the continuous
north-south gradient), the enclosure envelope section reports the
inverted-U test against the framework's pre-specified bands, and the
pedestrian-realm segmentation is broken out by street and avenue.
"""
import base64, sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
PROC = HERE / "data" / "processed"
RES = HERE / "results"


def read(p, **kw):
    p = Path(p)
    return pd.read_csv(p, **kw) if p.exists() else None


def read_multi(p):
    """pedestrian_by_street.csv is written from a MultiIndex column frame.

    Pandas writes that as two header rows, so a plain read_csv returns a
    frame whose first data row is the words mean/count. Read both header
    rows and flatten, then keep only the means -- the counts are identical
    across columns and belong in one place, not thirteen.
    """
    p = Path(p)
    if not p.exists():
        return None
    flat = pd.read_csv(p)
    if "n_nodes" in flat.columns or any(c.endswith("_mean") for c in flat.columns):
        # written by v12 pedestrian.py -- already flat
        for c in flat.columns:
            if c.endswith("_mean"):
                flat[c] = (flat[c] * 100).round(1)
        flat.columns = [c[:-5] if c.endswith("_mean") else c for c in flat.columns]
        return flat
    try:
        d = pd.read_csv(p, header=[0, 1], index_col=0)
    except Exception:
        return None
    if not isinstance(d.columns, pd.MultiIndex):
        return None
    n = None
    for a, b in d.columns:
        if b == "count":
            n = d[(a, b)]
            break
    keep = {a: d[(a, "mean")] for a, b in d.columns if b == "mean"}
    out = pd.DataFrame(keep)
    out = (out * 100).round(1)
    if n is not None:
        out.insert(0, "nodes", n.astype(int))
    return out.reset_index().rename(columns={"index": "osm_name",
                                             out.index.name or "index": "osm_name"})


MAX_EMBED_PX = 2400
JPEG_ABOVE_BYTES = 1_200_000


def _img_src(p):
    """Inline a figure, downscaled and recompressed if it is a photograph.

    Every figure is embedded as base64, which costs a third again on top of
    the file. The per-face contact sheets are 3,900 px of Street View
    imagery, and PNG is the wrong container for that -- five of them put
    30 MB into a page nobody can open on a phone. Photographs go out as
    JPEG; plots stay PNG, where the flat fills compress well and the text
    stays crisp. Full-resolution files remain on disk in results/figures.
    """
    import io
    raw = p.read_bytes()
    try:
        from PIL import Image
    except ImportError:
        return "image/png", base64.b64encode(raw).decode()
    im = Image.open(io.BytesIO(raw))
    if im.width > MAX_EMBED_PX:
        h = round(im.height * MAX_EMBED_PX / im.width)
        im = im.convert("RGB").resize((MAX_EMBED_PX, h), Image.LANCZOS)
    if len(raw) > JPEG_ABOVE_BYTES:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=82, optimize=True)
        return "image/jpeg", base64.b64encode(buf.getvalue()).decode()
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "image/png", base64.b64encode(buf.getvalue()).decode()


def img_tag(p, title, note=""):
    p = Path(p)
    if not p.exists():
        return ""
    mime, b64 = _img_src(p)
    cap = f"{title}" + (f" &middot; <span class='cap-note'>{note}</span>" if note else "")
    return (f'<figure><figcaption>{cap}</figcaption>'
            f'<img src="data:{mime};base64,{b64}" alt="{title}"></figure>')


def gallery(folder, title):
    """Per-street overlays, as a collapsible strip."""
    folder = Path(folder)
    if not folder.is_dir():
        return ""
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        return ""
    items = []
    for p in pngs:
        mime, b64 = _img_src(p)
        name = p.stem.replace("overlay_", "").replace("_", " ")
        items.append(f'<figure><figcaption>{name}</figcaption>'
                     f'<img src="data:{mime};base64,{b64}" alt="{name}"></figure>')
    return (f"<details><summary>{title} &mdash; {len(pngs)} panels "
            f"(click to expand)</summary>{''.join(items)}</details>")


def table(df, n=None, fmt="{:.3f}"):
    if df is None or not len(df):
        return "<p class='none'>not available</p>"
    d = (df.head(n) if n else df).copy()
    for c in d.select_dtypes("number"):
        d[c] = d[c].map(lambda v: fmt.format(v) if pd.notna(v) else "")
    return d.to_html(index=False, border=0, classes="tbl", escape=False)


def swatch_table(cats):
    """The class legend, with the colour each class is actually drawn in."""
    if cats is None or not len(cats):
        return "<p class='none'>run tools/face_samples.py --categories</p>"
    rows = []
    for _, r in cats.iterrows():
        rows.append(
            f"<tr><td><span style=\'display:inline-block;width:12px;"
            f"height:12px;border-radius:3px;background:{r.colour};"
            f"vertical-align:middle;margin-right:7px\'></span>"
            f"{r.category}</td><td>{r.source}</td>"
            f"<td style=\'font-size:12.5px;color:#555\'>{r.matches}</td></tr>")
    return ("<table class='tbl'><tr><th>class</th><th>source</th>"
            "<th>ADE20K synonyms / prompt</th></tr>"
            + "".join(rows) + "</table>")


def kpi(label, value, note=""):
    return (f'<div class="kpi"><div class="k-val">{value}</div>'
            f'<div class="k-lab">{label}</div>'
            f'<div class="k-note">{note}</div></div>')


def main():
    metrics = read(PROC / "metrics.csv")
    dm = read(PROC / "directional_metrics.csv")
    face = read(PROC / "block_faces.csv")
    reg_dir = read(RES / "tables" / "regression_by_direction.csv")
    reg_ac = read(RES / "tables" / "regression_along_cross.csv")
    reg_sw = read(RES / "tables" / "regression_sidewalk.csv")
    streets = read(RES / "tables" / "by_street.csv")
    nomet = read(RES / "tables" / "nodes_without_metrics.csv")
    gvi_vei = read(RES / "tables" / "regression_gvi_vei.csv")
    partial = read(RES / "tables" / "partial_correlations.csv")
    robust = read(RES / "tables" / "robust_associations.csv")
    env_dir = read(RES / "tables" / "enclosure_envelope_directional.csv")
    eyelevel = read(RES / "tables" / "eyelevel_bands.csv")
    cats = read(RES / "tables" / "segmentation_categories.csv")
    fsamp = read(RES / "tables" / "face_samples.csv")
    dob = read(PROC / "dob_shed_by_node.csv")
    fshares = read(RES / "tables" / "face_sample_shares.csv")
    ovv = read(RES / "tables" / "openvocab_eval.csv")
    if fsamp is not None and len(fsamp):
        # The absolute path is for the tool, not the reader.
        fsamp = fsamp.drop(columns=["path", "typology", "along_street"],
                           errors="ignore")
        for c in ["n_nodes", "travel_bearing"]:
            if c in fsamp.columns:
                fsamp[c] = fsamp[c].astype(int).astype(str)
    inv_u = read(RES / "tables" / "enclosure_invertedU.csv")
    race = read(RES / "tables" / "enclosure_shape_race.csv")
    bands = read(RES / "tables" / "enclosure_bands.csv")
    ped = read(RES / "tables" / "pedestrian_realm.csv")
    ped_st = read_multi(RES / "tables" / "pedestrian_by_street.csv")

    if metrics is None:
        sys.exit(f"no metrics.csv in {PROC} -- run the pipeline first")

    kpis = [
        kpi("nodes", f"{len(metrics):,}", "20 m spacing"),
        kpi("block faces", f"{len(face):,}" if face is not None else "&mdash;",
            "the n for inference"),
        kpi("median GVI", f"{metrics.GVI.median():.2f}%", "greenness"),
        kpi("median VEI", f"{metrics.VEI.median():.3f}", "enclosure"),
    ]
    if "HW_ratio" in metrics:
        kpis.append(kpi("median H/W", f"{metrics.HW_ratio.median():.2f}",
                        f"measured, n={int(metrics.HW_ratio.notna().sum())}"))
    if "SVF_band" in metrics:
        kpis.append(kpi("median SVF_band", f"{metrics.SVF_band.median():.3f}",
                        "band-limited, not SVF"))
    if nomet is not None and len(nomet):
        kpis.append(kpi("no metric", f"{len(nomet)}", "see coverage audit"))

    by_typ = (metrics.groupby("typology")[["GVI", "VEI"]]
              .agg(["median", "count"]).round(3))
    by_typ.columns = [f"{a}_{b}" for a, b in by_typ.columns]
    by_typ = by_typ.reset_index()

    # ---------------------------------------------------- gradient
    gradient, grad_rho = None, None
    if "northing_m" in metrics.columns:
        from scipy.stats import spearmanr
        b = pd.cut(metrics.northing_m, 5, precision=0)
        gradient = (metrics.groupby(b, observed=True)[["GVI", "VEI"]]
                    .agg(["median", "count"]).round(3))
        gradient.columns = [f"{a}_{b_}" for a, b_ in gradient.columns]
        gradient = gradient.reset_index()
        gradient.columns = ["uptown_band_m"] + list(gradient.columns[1:])
        gradient["uptown_band_m"] = gradient.uptown_band_m.astype(str)
        g = metrics[["northing_m", "GVI"]].dropna()
        if len(g) > 20:
            grad_rho = spearmanr(g.northing_m, g.GVI)

    by_dir = None
    if dm is not None:
        by_dir = (dm.groupby("direction")[["GVI", "VEI"]]
                  .agg(["median", "count"]).round(3))
        by_dir.columns = [f"{a}_{b}" for a, b in by_dir.columns]
        by_dir = by_dir.reset_index()

    # Numbers quoted in the plain-language section below. Computed here
    # rather than hardcoded into the prose, so the sentence and the data
    # cannot drift apart.
    from scipy.stats import spearmanr as _sp

    def rho(a, b):
        d = metrics.dropna(subset=[a, b])
        return _sp(d[a], d[b])[0] if len(d) > 20 else float("nan")

    gvi_vei_disp = None
    if gvi_vei is not None and len(gvi_vei):
        gvi_vei_disp = gvi_vei.rename(columns={
            "slope": "slope (GVI pts per VEI unit)", "r2": "R2",
            "spearman_rho": "rho", "ci_lo": "95% lo", "ci_hi": "95% hi"})
        gvi_vei_disp = gvi_vei_disp.drop(columns=["rho_p"], errors="ignore")
        gvi_vei_disp["n"] = gvi_vei_disp.n.astype(int).astype(str)

    R = {"hw_gvi": rho("HW_ratio", "GVI"), "hw_vei": rho("HW_ratio", "VEI"),
         "vei_gvi": rho("VEI", "GVI"), "veg_sky": rho("GVI", "SVF_band"),
         "veg_bld": rho("GVI", "BVF_band")}

    robust_tbl = None
    if robust is not None and len(robust):
        robust_tbl = robust.copy()
        robust_tbl["rho 95% CI"] = [
            f"[{lo:+.2f}, {hi:+.2f}]" if pd.notna(lo) else ""
            for lo, hi in zip(robust_tbl.rho_lo, robust_tbl.rho_hi)]
        robust_tbl["Theil-Sen 95% CI"] = [
            f"[{lo:+.2f}, {hi:+.2f}]"
            for lo, hi in zip(robust_tbl.ts_lo, robust_tbl.ts_hi)]
        robust_tbl["n"] = robust_tbl.n.astype(int).astype(str)
        robust_tbl = robust_tbl[["unit", "x", "subset", "n", "ols_slope",
                                 "r2", "theilsen_slope", "Theil-Sen 95% CI",
                                 "rho", "rho 95% CI"]]
        robust_tbl = robust_tbl.rename(columns={"ols_slope": "OLS slope",
                                                "r2": "R2",
                                                "theilsen_slope": "Theil-Sen"})

    if partial is not None and len(partial):
        partial = partial.copy()
        partial["n"] = partial.n.astype(int).astype(str)

    ov_tbl = None
    if ovv is not None and len(ovv):
        import sys as _s
        _s.path.insert(0, str(Path("tools")))
        from scaffold_eval import auc as _auc
        rows = []
        for name, g in ovv.groupby("class"):
            y = g.label.values.astype(bool)
            rows.append({"class": name, "n": str(len(g)),
                         "positives": str(int(y.sum())),
                         "Mask2Former (closed set)": round(_auc(g.closed_set.values, y), 2),
                         "CLIPSeg (open vocab)": round(_auc(g.open_vocab.values, y), 2)})
        ov_tbl = pd.DataFrame(rows)

    sim_tbl = None
    if fshares is not None and len(fshares):
        terms = [("eye_green", "GVI_eye (below camera height)"),
                 ("canopy_green", "canopy, above camera height"),
                 ("hard_barrier", "EBC hard: fences, railings"),
                 ("soft_buffer", "EBC soft: planters, hedges"),
                 ("shelter", "TEF: awnings, canopies, columns"),
                 ("rest", "SAI: benches, seats, steps"),
                 ("building", "denominator"), ("sky", "denominator"),
                 ("sidewalk", "denominator")]
        rows = []
        for col, what in terms:
            if col not in fshares.columns:
                continue
            v = fshares[col] * 100
            rows.append({"SIM term": col, "role": what,
                         "mean %": round(v.mean(), 2),
                         "median %": round(v.median(), 2),
                         "max %": round(v.max(), 2),
                         "faces with any": f"{int((v > 0).sum())} / {len(v)}"})
        sim_tbl = pd.DataFrame(rows)

    dob_tbl = None
    if dob is not None and len(dob):
        dob_tbl = (dob.groupby("osm_name")
                      .agg(nodes=("node_id", "size"),
                           shed_within_30m=("dob_shed", "mean"),
                           any_permit=("dob_any", "mean"),
                           median_nearest_m=("dob_nearest_m", "median"))
                      .sort_values("shed_within_30m", ascending=False)
                      .reset_index())
        dob_tbl["shed_within_30m"] = (dob_tbl.shed_within_30m * 100).round(0)
        dob_tbl["any_permit"] = (dob_tbl.any_permit * 100).round(0)
        dob_tbl["nodes"] = dob_tbl.nodes.astype(str)

    env_dir_tbl = None
    if env_dir is not None and len(env_dir):
        env_dir_tbl = (env_dir.pivot_table(index="hw_mid", columns="view",
                                           values="median")
                              .round(2).reset_index())
        env_dir_tbl = env_dir_tbl.rename(columns={"hw_mid": "H/W (bin median)"})

    eyelevel_tbl = None
    if eyelevel is not None and len(eyelevel):
        keep = ["node_id", "osm_name", "GVI"] + [c for c in eyelevel.columns
                                                 if c.startswith("GVI_")]
        eyelevel_tbl = eyelevel[[c for c in keep if c in eyelevel.columns]]

    corr_cols = [c for c in ["GVI", "VEI", "SVF_band", "HW_ratio", "H_m",
                             "W_facade", "northing_m", "scaffold_frac"]
                 if c in metrics]
    corr = metrics[corr_cols].corr(method="spearman").round(3).reset_index()

    # ---------------------------------------------------- enclosure verdict
    verdict_html = "<p class='none'>stage 7 has not been run</p>"
    if inv_u is not None and len(inv_u):
        r = inv_u.iloc[0]
        ok = bool(r.get("inverted_u", False))
        checks = [("b<sub>2</sub> &lt; 0 (concave)", bool(r.get("pass_a"))),
                  ("slope &gt; 0 at the low end", bool(r.get("pass_b"))),
                  ("slope &lt; 0 at the high end", bool(r.get("pass_c")))]
        rowsh = "".join(
            f"<tr><td>{n}</td><td class='{'pass' if v else 'fail'}'>"
            f"{'PASS' if v else 'FAIL'}</td></tr>" for n, v in checks)
        tp = r.get("turning_point", float("nan"))
        verdict_html = f"""
<div class="verdict {'ok' if ok else 'no'}">
  <div class="v-head">{'Inverted-U supported' if ok else 'Inverted-U not supported'}</div>
  <table class="tbl mini">{rowsh}</table>
  <p>b<sub>2</sub> = {r.get('b2', float('nan')):+.4f} (p = {r.get('p_b2', float('nan')):.2e}),
     turning point at H/W = {tp:.2f}.
     {'A positive b<sub>2</sub> is a U, not an inverted-U: the curve is convex.'
      if r.get('b2', 0) > 0 else ''}</p>
</div>"""

    figs = "".join([
        img_tag(RES / "figures" / "figure_enclosure.png",
                "Enclosure envelope",
                "quadratic against the pre-specified bands"),
        img_tag(RES / "figures" / "figure_maps.png", "Spatial distribution"),
        img_tag(RES / "figures" / "figure_scatter.png", "GVI against VEI"),
        img_tag(RES / "figures" / "figure_directional.png",
                "Regression by travel direction"),
        img_tag(RES / "figures" / "figure_rose.png", "Azimuthal profiles"),
    ])

    ped_figs = "".join([
        img_tag(RES / "figures" / "figure_pedestrian_overlay.png",
                "Pedestrian realm segmentation"),
        img_tag(RES / "figures" / "figure_pedestrian_directions.png",
                "Composition by travel direction"),
    ])
    ped_gallery = gallery(RES / "figures" / "pedestrian_by_street",
                          "Per-street segmentation overlays")

    ped_dir_tbl = None
    if ped is not None:
        cols = [c for c in ["sidewalk", "road", "vegetation", "seating",
                            "barrier", "steps", "furniture"] if c in ped.columns]
        if cols:
            ped_dir_tbl = (ped.groupby("view")[cols].mean() * 100).round(1).reset_index()

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Murray Hill streetscape results &middot; v12</title>
<style>
 :root {{ --ink:#1a1a1a; --mute:#6b6b6b; --line:#e2e0dc; --bg:#faf9f7;
          --accent:#2a6b45; --warn:#a8442a; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
 .wrap {{ max-width:1120px; margin:0 auto; padding:40px 28px 80px; }}
 h1 {{ font-size:30px; margin:0 0 4px; letter-spacing:-.02em; }}
 .sub {{ color:var(--mute); margin:0 0 32px; }}
 h2 {{ font-size:19px; margin:44px 0 6px; padding-top:20px;
       border-top:1px solid var(--line); }}
 h3 {{ font-size:15px; margin:20px 0 6px; }}
 .hint {{ color:var(--mute); font-size:13.5px; margin:0 0 14px; }}
 .kpis {{ display:flex; flex-wrap:wrap; gap:12px; margin:24px 0 8px; }}
 .kpi {{ flex:1 1 150px; background:#fff; border:1px solid var(--line);
   border-radius:10px; padding:14px 16px; }}
 .k-val {{ font-size:24px; font-weight:600; letter-spacing:-.02em; }}
 .k-lab {{ font-size:13px; margin-top:2px; }}
 .k-note {{ font-size:12px; color:var(--mute); }}
 table.tbl {{ border-collapse:collapse; width:100%; background:#fff;
   font-size:13.5px; border:1px solid var(--line); border-radius:8px; }}
 .tbl th {{ text-align:left; padding:9px 11px; background:#f2f0ec;
   font-weight:600; border-bottom:1px solid var(--line); white-space:nowrap; }}
 .tbl td {{ padding:8px 11px; border-bottom:1px solid #f0eee9;
   font-variant-numeric:tabular-nums; }}
 .tbl tr:last-child td {{ border-bottom:none; }}
 .tbl.mini {{ width:auto; margin:10px 0; }}
 .pass {{ color:var(--accent); font-weight:600; }}
 .fail {{ color:var(--warn); font-weight:600; }}
 figure {{ margin:22px 0; background:#fff; border:1px solid var(--line);
   border-radius:10px; padding:14px; }}
 figcaption {{ font-size:13px; color:var(--mute); margin-bottom:10px; }}
 .cap-note {{ font-style:italic; }}
 img {{ width:100%; height:auto; display:block; }}
 .none {{ color:var(--mute); font-style:italic; font-size:13.5px; }}
 .note {{ background:#fff; border-left:3px solid var(--accent);
   padding:12px 16px; margin:16px 0; font-size:13.5px;
   border-radius:0 8px 8px 0; }}
 .note.warn {{ border-left-color:var(--warn); }}
 .verdict {{ background:#fff; border:1px solid var(--line); border-radius:10px;
   padding:16px 18px; margin:16px 0; }}
 .verdict.no {{ border-left:4px solid var(--warn); }}
 .verdict.ok {{ border-left:4px solid var(--accent); }}
 .v-head {{ font-size:16px; font-weight:600; margin-bottom:4px; }}
 details {{ margin:14px 0; background:#fff; border:1px solid var(--line);
   border-radius:10px; padding:12px 16px; }}
 summary {{ cursor:pointer; font-size:13.5px; color:var(--accent);
   font-weight:600; }}
</style></head><body><div class="wrap">

<h1>Murray Hill streetscape analysis</h1>
<p class="sub">v12 &middot; Green View Index and Visual Enclosure Index from
Street View imagery &middot; Madison to First Avenue, E 34th to E 42nd<br>
<em>Working document, not paper text.</em> It records what was measured,
what it showed, and where the reading changed &mdash; including the places
an earlier reading was wrong.</p>

<div class="kpis">{''.join(kpis)}</div>

<h2>By street typology</h2>
<p class="hint">Canyon = Madison, Park, Lexington. Secondary = 3rd, 2nd, 1st.
Mid-block = E 34th&ndash;42nd.</p>
{table(by_typ)}

<h2>How strongly do greenery and enclosure actually move together?</h2>
<p class="hint">The same relationship measured four ways. The unit of
analysis changes down the rows; the conclusion should not depend on which
row you quote, and where it does, that is the finding.</p>
{table(gvi_vei_disp) if gvi_vei_disp is not None
 else "<p class='none'>run stage 6</p>"}

<details open><summary>What these numbers mean, in plain terms</summary>
<p><strong>Spearman&rsquo;s &rho; (rho) is a rank correlation.</strong> Put
every node in a queue from least to most enclosed. Now put the same nodes
in a queue from least to most green. &rho; asks how well the two queues
agree. &rho; = +1 means identical order, 0 means no relation, &minus;1
means exactly reversed. It only looks at <em>positions</em> in the queue,
never at the sizes, so one freak value cannot drag it &mdash; Park Avenue
at GVI 27 when every other street sits under 5 is simply &ldquo;the top
one&rdquo;, worth no more than the node just below it.</p>

<p><strong>The slope and R&sup2; answer different questions.</strong> The
slope says how many GVI points you gain per unit of VEI, and it is computed
from covariance over variance &mdash; so a handful of extreme values pull
it hard. R&sup2; says what share of the node-to-node variation in greenery
the predictor accounts for. A relationship can be reliable in direction
(&rho; well away from zero) and still explain almost nothing
(R&sup2; near zero). That is exactly this dataset.</p>

<p><strong>Reading ours:</strong></p>
<ul>
<li>&rho;(VEI, GVI) = <strong>{R['vei_gvi']:+.2f}</strong> at node level.
    More enclosed goes with less green &mdash; a moderate, consistent
    tendency, not a tight one. R&sup2; = 0.011 says enclosure accounts for
    about <strong>1%</strong> of why one node is greener than another.
    Both statements are true at once.</li>
<li>&rho;(H/W, GVI) = <strong>{R['hw_gvi']:+.2f}</strong>. The same story
    told by building geometry instead of by pixels, and told
    <em>more strongly</em>. Taller-and-narrower streets are reliably the
    less green ones.</li>
<li>&rho;(H/W, VEI) = <strong>{R['hw_vei']:+.2f}</strong>. This one is a
    check, not a finding: it says the image-based enclosure measure and the
    footprint-based one agree about half the time in rank order. VEI is
    measuring enclosure &mdash; but only about half of what it registers is
    the geometry, and the rest is everything else that fills a photograph.
    </li>
</ul>

<p><strong>Why VEI is the weaker of the two, mechanically.</strong> VEI is
facade &divide; (facade + sky), so anything that hides facade lowers it.
In this frame vegetation displaces <em>facade</em>, not sky: &rho;(GVI,
sky share) = <strong>{R['veg_sky']:+.2f}</strong> while &rho;(GVI, facade
share) = <strong>{R['veg_bld']:+.2f}</strong>. Street trees stand in front
of buildings at eye level, and at pitch 0 the sky band sits above them. So
part of the GVI&ndash;VEI link is pixel bookkeeping rather than urban form:
a green node is one whose facades are partly hidden, which is definitionally
a lower VEI. H/W, measured from footprints, cannot suffer this &mdash;
which is one reason to prefer it as the enclosure variable and to read VEI
as a corroborating measure rather than the primary one.</p>
</details>

<div class="note warn">A weak VEI result is not evidence that enclosure
does not matter. The geometric measure of the same construct reaches
&rho; = {R['hw_gvi']:+.2f} on the same nodes. Before replacing enclosure
with new parameters, the honest statement is that enclosure predicts
greenery moderately and monotonically &mdash; just not in the envelope's
predicted shape.</div>

<h2>Three estimators, one relationship</h2>
<p class="hint">Each row is the same association read three ways, with and
without Park Avenue. Every interval is bootstrapped over block faces, never
over nodes.</p>
{table(robust_tbl, fmt="{:.2f}") if robust_tbl is not None
 else "<p class='none'>run stage 6</p>"}
<div class="note"><strong>What each column is for.</strong>
<ul>
<li><strong>OLS slope</strong> &mdash; covariance over variance, so points
with ordinary x and extreme y drag it. This is the column Park Avenue
moves.</li>
<li><strong>Theil&ndash;Sen</strong> &mdash; the median of all pairwise
slopes. Half the points would have to move before it does, so it survives
Park Avenue without anyone deciding whether to delete a real street. This
is the robust slope to quote.</li>
<li><strong>&rho;</strong> &mdash; agreement of rank orders only, immune to
magnitude.</li>
</ul>
Read across the <code>all</code> and <code>no Park Ave</code> pairs. At
face level for VEI the OLS slope moves from &minus;1.34 to &minus;4.85
while Theil&ndash;Sen moves from &minus;10.13 to &minus;8.86 and &rho; from
&minus;0.49 to &minus;0.53. <strong>Where OLS moves and the other two do
not, the OLS number was reporting one street's leverage rather than the
association.</strong></div>
<div class="note warn"><strong>And now read the intervals.</strong> At face
level &mdash; 21 units &mdash; the 95% CI on &rho; for H/W against GVI is
<strong>[&minus;0.81, +0.01]</strong>. It contains zero. That is not a
defect of &rho; &mdash; it is 21 units. The point estimate of &minus;0.49
remains the best guess, the association is probably real and moderate, and
the data simply do not pin it down. Note also that the interval
<em>excludes</em> zero once Park Avenue is out
(&minus;0.93 to &minus;0.27): a green street at moderate H/W is precisely
what blurs a monotone rank pattern. Every honest inference in this project is a 22-unit
inference, and this is what that costs. Describe patterns and report
intervals; do not test many hypotheses on them.</div>

<h2>Is GVI an inverted-U in H/W, on the framework's bands?</h2>
<p class="hint">The bands below come from section 4 of the framework
document and were fixed before any GVI was inspected. That prior
specification is what makes the shape testable.</p>

<div class="note warn"><strong>What this does and does not test.</strong>
Section 4's three bullets say that under-enclosure leaves eye-level
greenery <em>visually diluted</em>, that human scale <em>frames</em> it,
and that a deep canyon is oppressive <em>unless offset by</em> rich
greenery. Every one is about what enclosure does to the <em>experience</em>
of greenery; none says greenery is more abundant at human scale. The
document's own model puts GVI under Place Attachment and canyon H/W under
Place Identity &mdash; separate arguments to the same function. So
&ldquo;GVI peaks at human scale&rdquo; is a proxy of our choosing, and what
follows tests the proxy. A failure is a finding about greenery and
geometry in Murray Hill; it is not a verdict on the enclosure envelope,
which is a claim about sense of place and is not measured here.</div>

<div class="note warn"><strong>Why a curve with a turning point, and not a
correlation line.</strong> Because the band edges were fixed in advance, so
a non-monotone shape over them was specified before the data were seen
&mdash; not because a straight line fitted poorly. A low linear
R&sup2; carries no information about curvature: it is equally consistent
with no relationship, with a monotone one buried in noise, and with any
non-linear shape. And the quadratic <em>nests</em> the linear, so its
R&sup2; can only rise; "the curve fits better" is arithmetic, not evidence.
Choosing the form after seeing the line fail, on the same data, is
specification search and the resulting p-value is not the one it claims to
be. The test below is therefore the Lind &amp; Mehlum (2010) joint test,
which a merely-significant squared term does not pass.</div>

{verdict_html}

<h3>Pre-specified bands</h3>
{table(bands, fmt="{:.2f}")}
<div class="note">A significant difference between bands is not a peak.
Read the medians in order: monotone decline and a mid-range maximum both
produce a significant Kruskal&ndash;Wallis statistic. A peak requires at
least one band-to-band increase before the decline.</div>

<h3>Against monotone alternatives</h3>
<p class="hint">CV RMSE uses folds split on block face, so no face straddles
train and test. It is the only column here that can fall either way.</p>
{table(race, fmt="{:.4f}")}

<h2>The pedestrian reading: what you see while walking</h2>
<p class="hint">A pedestrian walks along a street and looks where they are
going, so the view modelled here is the 180&deg; forward cone on the
bearing of travel &mdash; wide enough to include peripheral vision, and
restricted to the nodes where that bearing runs <em>along</em> the street.</p>
{img_tag(RES / "figures" / "figure_enclosure_directional.png",
         "Enclosure curve from the forward view while walking")}
<div class="note"><strong>Two conditions are deliberately absent.</strong>
<em>Cross-street views</em> &mdash; facing east while walking north &mdash;
are a side glance at a wall, not a way anyone experiences a street, so they
are excluded rather than plotted as a contrast. <em>The full 360 index</em>
is excluded for the same reason: it averages what a person sees with what
is behind their head, and the premise here is that the second half never
reaches them. Both used to be drawn on this figure and invited exactly the
comparisons the design says are meaningless.
<br><br>
What remains is four conditions that a person can actually be in: walking
uptown or downtown on an avenue, walking east or west on a cross street.
Uptown and downtown describe the same avenue nodes turned around, so the
gap between those two curves is a within-node contrast &mdash; what changes
when you reverse direction &mdash; not two independent samples.</div>
{table(env_dir_tbl, fmt="{:.2f}") if env_dir_tbl is not None
 else "<p class='none'>run stage 7</p>"}
<div class="note warn"><strong>The avenue spike is one street.</strong>
Walking an avenue means looking straight down the corridor, and at H/W
&asymp; 1.2 the forward view runs along Park Avenue's planted mall: median
GVI <strong>11.2 walking uptown</strong>, 9.7 walking downtown, against
roughly 3 everywhere else on the curve. Take Park Avenue out and the same
bin reads <strong>1.7</strong> and 2.9 &mdash; the dotted lines on the
right panel. Park Avenue is 35 of the 67 avenue nodes with a measured H/W,
so on this subset it is not an outlier to be handled, it is half the
sample. The mall is real streetscape and the pedestrian experience of it is
real; what cannot be said is that avenues at human scale are green.</div>
<div class="note">Across the pooled forward view the shape is the same one
the 360 index gives: flat from H/W 0.45 to 1.5, then a steep decline, and
no peak in the human-scale band. The confidence band is wide exactly where
Park Avenue sits.</div>

<h2>Why the first reading said &ldquo;no relationship&rdquo;, and why it was wrong</h2>
<p class="hint">One street, three block faces, and what it does to an
ordinary least-squares fit.</p>
{img_tag(RES / "figures" / "figure_leverage.png",
         "Park Avenue's leverage on the slope")}
<div class="note"><strong>What happened.</strong> Park Avenue carries a
planted central mall: GVI 14&ndash;17 where every other street sits under
5, and it does so at H/W near 1.2 &mdash; the middle of the range, not the
edge. An OLS slope is covariance over variance, so a cluster of points with
<em>ordinary x</em> and <em>extreme y</em> drags the fitted line towards
flat and takes R&sup2; down with it. At block-face level that produced
R&sup2; = 0.002 and p = 0.85 for GVI against VEI, which reads as "no
relationship". Drop those three faces and the same fit gives R&sup2; =
0.165; for GVI against H/W it goes from 0.096 to <strong>0.501</strong>.
<br><br>
<strong>Why Spearman saw through it.</strong> A rank correlation asks only
whether the greener faces are the less enclosed ones. Park Avenue is the
top of the greenness order either way, so removing it barely moves &rho;:
&minus;0.49 to &minus;0.53 against VEI, &minus;0.49 to &minus;0.71 against
H/W. The rank statistic was stable while the slope swung five-fold, and
<em>that divergence is the diagnostic</em> &mdash; it is how leverage
announces itself. The same signature produced v11's spurious
&ldquo;west-only&rdquo; directional finding.
<br><br>
<strong>What to report.</strong> Both fits, and which one the text quotes.
Park Avenue's median is real streetscape, not an error to be cleaned away
&mdash; but a result that exists only with it, or only without it, is a
result about Park Avenue.</div>

<h2>North&ndash;south gradient</h2>
<p class="hint">Metres uptown along the grid axis (bearing 029) from the
south edge of the frame. Bands are for display; the statistics use the
continuous value.</p>
{table(gradient) if gradient is not None
 else "<p class='none'>no northing_m in metrics.csv</p>"}
{f'''<div class="note">Spearman &rho; = {grad_rho[0]:+.3f} (p =
{grad_rho[1]:.1e}) between distance uptown and GVI. Reported as a gradient,
not a boundary. The three named zones used through v11 were a latitude cut
across a grid rotated 29&deg;, so the boundary ran diagonally through the
street pattern and split all fifteen streets between two zones; at
block-face level they explained nothing (adj R&sup2; 0.02, p = 0.33) while
the continuous version reaches p = 0.03.</div>'''
 if grad_rho is not None else ""}

<h2>Is the enclosure result just geography?</h2>
<p class="hint">Streets get taller and narrower towards Grand Central and
greener towards the south, so grid-axis position is correlated with both
sides of the enclosure result. This partials it out.</p>
{table(partial, fmt="{:.3f}") if partial is not None
 else "<p class='none'>run stage 6</p>"}
<div class="note warn"><strong>Read the face rows, and read them as bad
news.</strong> At block-face level &mdash; the only unit here whose
observations are independent &mdash; the H/W&ndash;GVI association is
&rho; = &minus;0.49 raw and <strong>+0.05 (p = 0.85)</strong> once
grid-axis position is held constant. It does not survive. Position
correlates +0.71 with H/W and &minus;0.74 with GVI at that level, so both
sides of the "enclosure predicts greenery" story are carried by how far
uptown a face sits. VEI holds up better (&minus;0.49 to &minus;0.33) but
not significantly. The node-level partials look stronger only because they
are computed on an n the data do not have.
<br><br>
This is what grid-axis position is <em>for</em>. It is not a finding in its
own right and it should not be presented as one &mdash; it is the covariate
that decides whether an enclosure effect is an enclosure effect or a
north&ndash;south gradient wearing its clothes. Remove it and this check
becomes impossible.</div>

<h2>By travel direction</h2>
<p class="hint">180&deg; forward view on the Manhattan grid
(029/119/209/299), plus the full 360&deg;. A pedestrian does not look
behind.</p>
{table(by_dir)}
<div class="note warn">The four views are <em>not</em> four independent
samples. N and S are disjoint halves that tile the circle, and so are E and
W &mdash; two complete decompositions of the same node, not four
measurements. Compare the <code>rho</code> and <code>slope</code> columns
below: if rank correlation is near-constant while the OLS slope swings
several-fold, the spread is leverage from the low-VEI tail, since slope is
cov/var and a few extreme-x points move it a long way.</div>

<h2>Regressions</h2>
<p class="hint">Read R&sup2;, not p. At this n a slope can be significant
while explaining almost nothing.</p>
<h3>GVI ~ VEI, by view</h3>
{table(reg_dir)}
<h3>GVI ~ VEI, by viewing situation</h3>
<p class="hint">Which compass views are along-street flips with typology
&mdash; on the avenues N and S run along the corridor, on the cross streets
E and W do. Every compass row above therefore mixes both situations. This
is the contrast the design supports.</p>
{table(reg_ac)}
<h3>Sidewalk geometry</h3>
{table(reg_sw)}

<h2>Pedestrian realm</h2>
<p class="hint">Semantic segmentation of the ground plane a pedestrian
negotiates: sidewalk, seating, barriers, steps, street furniture. ADE20K
classes carry the quantitative claims; the open-vocabulary classes
(bollards, planters, tree pits, tactile paving) are indicative only &mdash;
their threshold is untuned against hand-labelled data.</p>
<h3>Mean share of view by street and avenue (%)</h3>
{table(ped_st, fmt="{:.1f}")}
<h3>By travel direction (%)</h3>
{table(ped_dir_tbl, fmt="{:.1f}")}
<div class="note">Differences between directions at the same node are the
anisotropy a 360&deg; index averages away. Sidewalk share in particular
should differ along-street versus cross-street.</div>
{ped_figs}
{ped_gallery}

<h2>Segmentation samples, one per block face</h2>
<p class="hint">The metrics are shares of a segmentation, and a share is
only as good as the labelling beneath it. One sample per block face, chosen
by rule: the node whose GVI is closest to that face's median &mdash; never
the photogenic one.</p>
<div class="note"><strong>These are the 180&deg; view the metrics are
computed over</strong>, not a raw frame. The imagery was fetched on
0/90/180/270 true north while the grid runs 029/119/209/299, so no single
frame is the pedestrian's forward view. Each strip is reprojected from the
three frames that cover it: a source frame is a gnomonic projection, so a
ray at azimuth offset <em>a</em> and elevation <em>e</em> sits at
x = tan(a)/tan(fov/2) and y = tan(e)/(cos(a)&middot;tan(fov/2)), and it is
the cos(a) that a plain side-by-side paste gets wrong. The
<em>labels</em> are reprojected too rather than the strip being
re-segmented, so what you see is exactly what produced the number beside
it.
<br><br>
Vertical extent is &plusmn;35&deg;: a 90&deg; frame reaches
&plusmn;45&deg; at its centre but only &plusmn;35.3&deg; at its corners,
since tan(e) &le; cos(45&deg;). Nothing in the metrics changes &mdash; the
profiles are built per frame, before any of this.</div>
{gallery(RES / "figures" / "face_samples", "Per-face segmentation samples")}
{table(fsamp, 30, fmt="{:.2f}") if fsamp is not None
 else "<p class='none'>run <code>python tools/face_samples.py --dry-run</code> "
      "to choose the samples, then the same command without --dry-run on a "
      "GPU box to segment them.</p>"}

<h3>The Street Interface Matrix, as pixels</h3>
<p class="hint">The classes drawn are now the study's own four domains
&mdash; GVI<sub>eye</sub>, EBC, TEF, SAI &mdash; plus the three denominators
the formula names, so a panel can be read against the equation instead of
against a different taxonomy.</p>
{table(sim_tbl, fmt="{:.2f}") if sim_tbl is not None
 else "<p class='none'>run tools/face_samples.py</p>"}
<div class="note warn"><strong>Three of the four terms are empty at this
resolution.</strong> Across the 25 sampled faces, soft buffers average
<strong>0.03%</strong> of the view and appear on 6 faces, shelter
<strong>0.01%</strong> on 3 faces, and micro-rest <strong>0.11%</strong> on
8. Hard barriers (2.04%, 15 faces) and eye-level greenery (0.83%) are the
only terms with enough signal to weight.
<br><br>
The reason is the class list, not the streets. Checked against the model,
ADE20K has no <em>arcade</em>, <em>bollard</em>, <em>hedge</em>,
<em>shrub</em>, <em>planter</em>, <em>pergola</em>, <em>porch</em>,
<em>balcony</em>, <em>gate</em> or <em>grille</em> &mdash; and those are
the study's own examples for &Omega;<sub>soft</sub>,
&Omega;<sub>shelter</sub> and &Omega;<sub>rest</sub>. What is left are
proxies: the flowerpot class for planters, awning and canopy for arcades,
undifferentiated steps for stoops. Two further compromises are recorded in
<code>config.yaml</code>: ADE20K's <em>wall</em> is the building face, so
including it in hard barriers would swallow every facade, and it is left
out; <em>column</em> is counted as shelter on the argument that a
ground-level column in Manhattan is usually a portico or an arcade support.
<br><br>
And in Manhattan &Omega;<sub>shelter</sub> is dominated by sidewalk sheds
&mdash; the one class with a validated detector, which scores AUC 0.51.
That term has both the largest gap and the least evidence that any current
method closes it.</div>

<h3>Does open-vocabulary grounding see these classes? Measured, per class</h3>
<p class="hint">Each class scored against a geocoded city register, on the
same nodes, with both detectors. AUC 0.5 is a coin flip.</p>
{table(ov_tbl, fmt="{:.2f}") if ov_tbl is not None
 else "<p class='none'>run tools/openvocab_eval.py</p>"}
<div class="note"><strong>street_tree is the control</strong>, against the
2015 Street Tree Census: ADE20K certainly has a tree class, so a harness
that could not score it would be measuring itself. It scores 0.83, which
means the tests below it are readable &mdash; and it retroactively confirms
that the scaffolding result of 0.51 was a real failure rather than a broken
benchmark.
<br><br>
<strong>The study's premise holds; its conclusion does not follow.</strong>
Where ADE20K has the class, closed-set segmentation wins &mdash; trees 0.83
against 0.78, benches 0.66 against 0.60. Where ADE20K is blind, open
vocabulary wins &mdash; bus shelters 0.65 against 0.56, the only reversal
in the table, and exactly the case the morphology study is arguing about.
So "closed-set segmentation cannot see three of the four SIM terms" is
correct, and "therefore use open-vocabulary grounding" buys 0.65 where it
matters. That is better than blind and nowhere near enough to carry a
weighted term in an index.
<br><br>
<strong>What this does and does not bound.</strong> It bounds CLIPSeg, a
small 2022 open-vocabulary segmenter &mdash; not the frontier VLM with
visual-spatial grounding the study proposes. That model remains untested.
What has changed is that testing it is now a fixed cost rather than an
argument: <code>tools/openvocab_eval.py</code> will score any detector on
the same nodes against the same registers, and the bar to beat is on this
table.
<br><br>
The bench row carries a caveat: DOT's City Bench register lists only 10
objects inside the study bbox, so that test rests on 8 positives. Read it
as a hint, not a result.</div>

<h3>Scaffolding: where the ground truth comes from</h3>
<p class="hint">New York requires a permit for every sidewalk shed,
supported or suspended scaffold and construction fence. DOB NOW publishes
each with an address, a latitude and longitude, an issue date and an
expiry date.</p>
<div class="note"><strong>Why permits, and why they are the accurate
source.</strong> The alternative was more prompt engineering, which has no
ground truth behind it &mdash; the threshold was set by eye on two frames.
The permit record does not depend on what a model thinks a photograph looks
like, and three properties make it usable here:
<ul>
<li><strong>It is compulsory and enforced.</strong> A shed without a permit
is a violation, so the register is close to a census of the thing rather
than a sample of it.</li>
<li><strong>It is dated at both ends.</strong> A shed is only evidence for
an image if it was standing the day the image was taken, and this frame's
imagery is filtered to a single capture month. "Live at 2026-04" is a sharp
filter, not a guess: <strong>977 permits</strong> were live that month
inside the study bbox &mdash; 390 sidewalk sheds, 236 supported and 246
suspended scaffolds, 105 construction fences.</li>
<li><strong>It is geocoded.</strong> Each permit carries a lat/lon, so it
joins to sampling nodes by distance without geocoding addresses ourselves.</li>
</ul>
<strong>Where it is wrong.</strong> A permit is filed against a building, so
its point is the address rather than the structure; permits are signed off
late often enough that a live record can outlast the shed; and a shed across
a wide avenue is within matching distance while being barely visible. None
of those errors is correlated with what a CLIPSeg prompt thinks a shed looks
like &mdash; which is the property that matters. An imperfect label with
independent errors still measures the detector.</div>
{table(dob_tbl, fmt="{:.0f}") if dob_tbl is not None
 else "<p class='none'>run <code>python tools/dob_sheds.py</code></p>"}
<div class="note warn"><strong>And the permits say the detector does not
work.</strong> Scored on a balanced subsample &mdash; 30 nodes with a
permitted shed within 30 m against 30 without, run through the same
180&deg; forward view the panels draw:

<table class="tbl mini">
<tr><th>permit label</th><th>n positive</th><th>median share, shed</th>
    <th>median share, no shed</th><th>AUC</th></tr>
<tr><td>within 30 m</td><td>30</td><td>0.19%</td><td>0.03%</td>
    <td class="fail">0.55</td></tr>
<tr><td>within 30 m <em>and inside the forward cone</em></td><td>22</td>
    <td>0.18%</td><td>0.06%</td><td class="fail">0.51</td></tr>
</table>

AUC 0.5 is a coin flip, and 0.51 is what restricting the label to what the
camera can actually see gives. The detector does not rank a node with a
sidewalk shed above one without. It also barely fires: the median share is
two tenths of one percent either way, so the handful of frames scoring 10%
or 20% are not "the scaffolded ones", they are outliers of something else.
<br><br>
Contrastive scoring was still worth doing &mdash; it removed the specific
false positives it was built for, East 42nd from 11.8% to 0.0% &mdash; but
removing false positives from a detector that has no signal only makes it
quieter. <strong>No scaffolding share is reportable from this method.</strong>
The permits themselves are reportable: they are a census, dated, and they
say a third of the frame's nodes stand within 30 m of a live shed.
<br><br>
What this now enables is model selection on evidence. Any candidate &mdash;
OWLv2, a local VLM, a hosted one &mdash; runs through
<code>tools/scaffold_eval.py</code> and comes back with an AUC on the same
60 nodes. The benchmark exists before the spending does.</div>

<h3>What each class is</h3>
<p class="hint">Matching is exact against ADE20K's comma-separated
synonyms, never substring: &ldquo;tree&rdquo; is inside &ldquo;street
lamp&rdquo;, &ldquo;sky&rdquo; inside &ldquo;skyscraper&rdquo;. Definitions
live in <code>config.yaml</code> under <code>pedestrian:</code>, and both
the sample panels and the pedestrian-realm shares read them from there.</p>
{swatch_table(cats)}
<div class="note warn"><strong>Scaffolding is drawn but not measured.</strong>
ADE20K has no class for a sidewalk shed, so until now every shed in the
frame was counted as the building or the road it hides &mdash; which in
Manhattan is not a rounding error. It is now painted as its own class from
CLIPSeg prompts, on top of whatever ADE20K called it. The threshold is
untuned: on the one hand-checked pair it puts 15% of pixels on a frame with
a shed against 4% on one without, and in the panels it finds scaffolded
facades reliably while missing at least one canopy-style shed (f18, right
of centre, still labelled building). Use it to see where sheds are; do not
report a share of it. Making it a measured class needs either a
hand-labelled sample to tune the threshold, or DOB sidewalk-shed permits
(<code>permit_subtype = 'SH'</code>) as ground truth.</div>
<div class="note warn">The ADE20K rows carry the quantitative claims. The
open-vocabulary rows &mdash; bollards, planters, tree pits, tactile paving,
bike racks, play equipment &mdash; are <em>indicative only</em>: their
threshold has never been tuned against hand-labelled data, so they locate
features rather than measure them. Do not build a metric on them without
tuning against a hand-labelled sample first.</div>

<h2>Framework parameters &mdash; what is measured and what is not</h2>
<p class="hint">The five parameters of the framework document, and the
state of each in this pipeline. Sections below fill in as the tools that
produce them are run.</p>
<table class="tbl">
<tr><th>parameter</th><th>state</th><th>produced by</th></tr>
<tr><td>1. Eye-level vegetation vs overhead canopy</td>
    <td>{"<span class='pass'>measured</span>" if eyelevel_tbl is not None
        else "<span class='fail'>built, not yet run</span>"}</td>
    <td><code>tools/eyelevel.py</code> &mdash; cached imagery, no API spend</td></tr>
<tr><td>2. Thermal mitigation (&Delta;MRT)</td>
    <td><span class="fail">out of scope</span></td>
    <td>needs meteorological data or microclimate simulation</td></tr>
<tr><td>3. Edge effect &mdash; stoops, yards, planters, railings</td>
    <td>{"<span class='pass'>measured</span>" if ped is not None
        else "<span class='fail'>built, not yet run</span>"}</td>
    <td><code>tools/pedestrian.py</code> &mdash; ADE20K barrier/steps/seating</td></tr>
<tr><td>4. Street canyon geometry (H/W)</td>
    <td><span class="pass">measured</span></td>
    <td>Stage 5 footprints &rarr; Stage 7</td></tr>
<tr><td>5. Ground effect &mdash; pavers, materiality, level change</td>
    <td><span class="fail">partial &mdash; no tool yet</span></td>
    <td>sidewalk/road exist in ADE20K; materiality would be
        open-vocabulary and indicative only</td></tr>
<tr><td>Outcome: sense of place / belonging</td>
    <td><span class="fail">not measured</span></td>
    <td>needs survey, dwell time or intercept counts</td></tr>
</table>

<h3>Eye-level against overhead vegetation</h3>
<p class="hint">At pitch 0 the horizon row is the camera-height plane, so
anything below it stands below roughly 2.5 m at any distance &mdash; the
framework's eye-level ceiling, recovered without a depth estimate. Bands
are stated in degrees of elevation, not metres, for the same reason
<code>SVF_band</code> is not called SVF.</p>
{table(eyelevel_tbl, 40, fmt="{:.2f}") if eyelevel_tbl is not None else
 "<p class='none'>Not yet run. <code>python tools/eyelevel.py --selftest</code> "
 "checks the geometry with no GPU; <code>python tools/eyelevel.py --n 4</code> "
 "prototypes on four stratified nodes from imagery already on disk.</p>"}
<div class="note">Stage 3 averages each image down its columns, so the
vertical axis is gone from <code>azimuth_profiles.npz</code>. This is the
one framework claim the saved profiles cannot answer &mdash; it needs
re-segmentation, which is why it runs on a subset first.</div>

<h2>Correlations</h2>
<p class="hint">Spearman &rho;. Height and H/W are near-duplicates by
construction; the informative cell is height against facade width.</p>
{table(corr, fmt="{:.2f}")}

<h2>Per street</h2>
<p class="hint">Bootstrap 95% CI on the median. Overlapping intervals are
the honest picture.</p>
{table(streets, fmt="{:.2f}")}

<h2>Figures</h2>
{figs if figs else "<p class='none'>no figures found</p>"}

<h2>Coverage audit</h2>
{f"<p class='hint'>{len(nomet)} frame nodes have no metric, with the reason "
 f"from the Stage 2 metadata probe.</p>" + table(nomet, 40)
 if nomet is not None and len(nomet)
 else "<p class='hint'>Every node in the frame produced a metric.</p>"}
<div class="note">Check the reason column before calling these coverage
gaps. A node excluded by the capture-date filter has usable imagery from
another date; a node with no panorama does not. Only one of those is
recoverable, and recovering it would mix leaf-on and leaf-off imagery into
the very variable being measured.</div>

<h2>Working notes &mdash; what changed, and what to do differently</h2>
<div class="note"><strong>Three readings that turned out to be wrong, and
what corrected them.</strong>
<ol>
<li><em>&ldquo;GVI and enclosure are unrelated.&rdquo;</em> That was Park
Avenue's leverage on an OLS slope. Rank correlation was stable where the
slope was not; the divergence between them is the diagnostic, not a
curiosity. <strong>Use both, always, and treat a gap between them as a
finding about leverage.</strong></li>
<li><em>&ldquo;The framework's enclosure envelope fails.&rdquo;</em> The
document's section 4 makes claims about how enclosure changes the
<em>experience</em> of greenery, never about how much greenery there is.
What fails is a proxy we chose. <strong>Quote the source text before
building a test on it.</strong></li>
<li><em>&ldquo;Enclosure predicts greenery, moderately.&rdquo;</em> At
block-face level it does not survive controlling for grid-axis position.
<strong>Run the confound check before the headline, not after.</strong></li>
</ol></div>
<div class="note"><strong>Method choices worth revisiting.</strong>
<ul>
<li><strong class="pass">Done &mdash; robust slope beside the OLS one.</strong>
Theil&ndash;Sen is now reported for every association, and it is stable
across the Park Avenue split where OLS is not. See "Three estimators, one
relationship".</li>
<li><strong class="pass">Done &mdash; rank correlations bootstrapped over
faces.</strong> Every &rho; now carries a 95% interval resampled over block
faces. At face level several of them contain zero, which is the honest
picture at n = 21.</li>
<li><strong class="pass">Done &mdash; directional views restricted to the
pedestrian condition.</strong> Cross-street views and the 360 index are out
of the pedestrian figure: you look where you walk, and nothing behind your
head reaches you.</li>
<li><strong>The face n is 22.</strong> Every honest inference in this
project is a 22-unit inference. That is the real constraint on what can be
claimed, and it argues for describing patterns and reporting intervals
rather than testing many hypotheses.</li>
<li><strong>Still open: a directional statistic, not just a curve.</strong>
The pedestrian figure is descriptive. If a number is wanted, the
informative quantity is the <em>anisotropy</em> &mdash; the difference
between walking uptown and downtown at the same node &mdash; not four
separate levels, which re-average by construction.</li>
<li><strong>Still open: the avenue subset is half Park Avenue.</strong>
Any avenue-level claim needs either more avenues or an explicit
Park-Avenue-and-others framing. Six avenues, 67 nodes with H/W, 35 of them
one street.</li>
</ul></div>

<h2>Reading these numbers</h2>
<div class="note">Nodes 20 m apart on the same block face photograph nearly
the same scene, so node-level p-values are inflated. Report the block-face
n, or standard errors clustered by face. Both appear in the Stage 6 console
output.</div>
<div class="note"><strong>The viewpoint is the roadway, not the
sidewalk.</strong> Panoramas sit a median of 2.7 m from the node (max 8.3),
which is a traffic lane, while a pedestrian is another 4&ndash;8 m to the
side. For the forward view this matters less than it sounds: a lateral
shift of a few metres barely moves anything more than ~30 m down the
street, and within the forward 180&deg; the vegetation is spread
1.32 / 1.47 / 1.04 GVI points across the 0&ndash;30&deg;, 30&ndash;60&deg;
and 60&ndash;90&deg; wedges off the direction of travel. About
<strong>26%</strong> of the forward view's greenery sits in the outer
wedges &mdash; the near-field beside you &mdash; and that is the part a
sidewalk viewpoint would change, raising near-side content and lowering
far-side. The other three quarters are effectively viewpoint-invariant.
Untestable without sidewalk-level imagery, and it matters most where
sidewalk width varies systematically between the streets being
compared.</div>
<div class="note">Imagery covers &plusmn;45&deg; of elevation only. Overhead
canopy and most sky are never sampled, so GVI is biased low and VEI high by
an unmeasured amount. Sky fractions here are band-limited and must not be
reported as SVF.</div>
<div class="note warn">The framework's dependent variable is sense of
place, and the claim its third bullet makes &mdash; a deep canyon is
oppressive <em>unless offset by</em> rich eye-level greenery &mdash; is an
explicit GVI &times; H/W interaction on that outcome. Nothing in this
dataset measures sense of place: no survey, no dwell time, no intercept
counts. That interaction is therefore not weakly supported here, it is
<em>unmeasured</em>. GVI and H/W are both <em>inputs</em>, so the section
above tests a proxy of our own construction &mdash; whether GVI happens to
peak at human scale &mdash; and the envelope proper remains untested for
want of an outcome variable.</div>

<p class="sub" style="margin-top:40px;font-size:12.5px">
Generated by make_dashboard.py &middot; Street View imagery &copy; Google
&middot; Building footprints &copy; NYC Open Data &middot; Street network
&copy; OpenStreetMap contributors</p>
</div></body></html>"""

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, self-contained)")
    print(f"\n  macOS:   open {out}")
    print(f"  Windows: start {out}")


if __name__ == "__main__":
    main()
