"""Do the VLM's rungs correspond to anything measurable?

The ratings are the study's instrument, so the first question a reviewer asks
is whether they track something outside the model. Each field is paired with a
MEASURED TWIN -- a pixel share from segmentation that should move with it if
the rung set means what it says -- and two things are drawn.

THE LADDER. Measured share by rated rung. If the scale is calibrated, rung 5
sits above rung 4 sits above rung 3, and the steps grow, because the rung sets
were written on a Weber-Fechner reading in which each rung is roughly two to
four times the previous one in pixel share. A flat or non-monotone ladder means
the model is not using the scale as written, whatever the correlation says.

THE CORRELATION, against the same distribution read a different way. The model
is held constant and only the readout changes, so the comparison isolates the
readout rather than crediting it with the model's work.

ONE FIELD HAS NO TWIN. facade_variation: every pixel source tried correlates
negatively or not at all. It is drawn as absent rather than quietly omitted --
a validation figure showing only what validated is not a validation figure.

    .venv/Scripts/python tools/validation_figure.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner
from sim_readout import prune_once, interpolated_median, K
from scipy.stats import spearmanr

# White ground: these two go on slides beside body text, where a dark panel
# reads as a hole in the page. --dark restores the palette used by the map
# figures, which are dark because a basemap is.
LIGHT = dict(bg="#ffffff", fg="#1a1a1a", mut="#6b7078", grid="#e6e6e8",
             spine="#c8c8cc", box="#2f7d3e", boxneg="#a33", face="#dcecdf",
             faceneg="#f2dede")
DARK = dict(bg="#0e0f12", fg="#e8e6e1", mut="#9a9aa2", grid="#20242a",
            spine="#3a3f46", box="#5fbf6a", boxneg="#c0392b", face="#1d5c2a",
            faceneg="#5c1d1d")
GHOST = "#8b929c"

# field, label, twin description, the columns that make the twin.
# __band and __gmi are joined from their own tables rather than summed here.
TWINS = [
    ("vertical_greenery", "vertical greenery", "map Vegetation, whole frame",
     ["map_Vegetation"]),
    ("green_eye_level", "green at eye level", "vegetation 0-15 deg below horizon",
     ["__band"]),
    ("green_softening", "green softening", "greenery on the lower 3 m of facade",
     ["__gmi"]),
    ("vertical_hardscape", "vertical hardscape", "map Building + Wall",
     ["map_Building", "map_Wall"]),
    ("sky_openness", "sky openness", "map Sky",
     ["map_Sky"]),
    ("signage_detail", "signage detail", "map Billboard",
     ["map_Billboard"]),
    ("walkable_ground", "walkable ground", "sidewalk + curb + pedestrian area",
     ["map_Sidewalk", "map_Curb", "map_Curb Cut", "map_Pedestrian Area"]),
    ("resting_affordance", "resting affordance", "bench + stairs + step",
     ["map_Bench", "ade_bench", "ade_stairs", "ade_step"]),
    ("ground_floor_activity", "ground floor activity", "windowpane + door",
     ["ade_windowpane", "ade_door"]),
    ("facade_variation", "facade variation", "no twin found", []),
]


def build(ratings, seg, bands, gmi):
    """Attach each field's measured twin to the ratings frame."""
    col = lambda d, c: d[c] if c in d.columns else pd.Series(0.0, index=d.index)
    tw = {"file": seg.file}
    for f, _, _, members in TWINS:
        if members and members[0] not in ("__band", "__gmi"):
            tw[f] = sum(col(seg, c) for c in members)
    t = pd.DataFrame(tw)
    t = t.merge(bands[["file", "veg_eye0_15"]]
                .rename(columns={"veg_eye0_15": "green_eye_level"}),
                on="file", how="left")
    t = t.merge(gmi[["file", "gmi_band"]]
                .rename(columns={"gmi_band": "green_softening"}),
                on="file", how="left")
    return ratings.merge(t, on="file", how="inner", suffixes=("", "_tw"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "sim_vlm_v3.csv")
    ap.add_argument("--slide", default="16:9", choices=["16:9", "free"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--dark", action="store_true",
                    help="dark ground instead of white")
    args = ap.parse_args()
    PAL = DARK if args.dark else LIGHT
    BG, FG, MUT = PAL["bg"], PAL["fg"], PAL["mut"]
    banner("validation: the rungs against measured pixel shares")

    j = build(pd.read_csv(args.table),
              pd.read_csv(PROC / "seg90_two_model.csv"),
              pd.read_csv(PROC / "seg90_bands.csv"),
              pd.read_csv(PROC / "seg90_gmi_band.csv"))
    print(f"{len(j)} half-views\n")

    rows = []
    for f, label, twin, members in TWINS:
        # A field with no twin must NOT fall back to its own rating column:
        # the merge suffixes only rename on collision, so `f` alone is the
        # RATING, and correlating it with itself gave facade_variation a
        # spurious rho of +1.000 against round(EV).
        c = f + "_tw" if f + "_tw" in j.columns else f
        if not members or c not in j.columns:
            rows.append(dict(field=f, label=label, twin=twin, rho=np.nan,
                             ev=np.nan, n=0, ladder=None))
            print(f"  {label:<24}no twin")
            continue
        P = j[[f"{f}_p{k}" for k in K]].to_numpy(float)
        P = P / P.sum(axis=1, keepdims=True)
        read = interpolated_median(prune_once(P))
        m = j[c].notna().to_numpy()
        rho = float(spearmanr(read[m], j.loc[m, c]).statistic)
        ev = float(spearmanr(j.loc[m, f], j.loc[m, c]).statistic)
        rung = np.clip(np.round(read), 1, 7).astype(int)
        ladder = [j.loc[m & (rung == k), c].to_numpy() for k in range(1, 8)]
        rows.append(dict(field=f, label=label, twin=twin, rho=rho, ev=ev,
                         n=int(m.sum()), ladder=ladder))
        print(f"  {label:<24}rho {rho:+.3f}   round(EV) {ev:+.3f}   n {int(m.sum())}")

    R = pd.DataFrame([{k: v for k, v in d.items() if k != "ladder"}
                      for d in rows])
    R.to_csv(RES / "tables" / "validation_twins.csv", index=False)

    size = (13.333, 7.5) if args.slide == "16:9" else (11.0, 7.0)

    # ---- the correlations -------------------------------------------------
    fig, ax = plt.subplots(figsize=size, facecolor=BG)
    ax.set_facecolor(BG)
    d = R.dropna(subset=["rho"]).sort_values("rho").reset_index(drop=True)
    y = np.arange(len(d))
    ax.barh(y, d.rho, height=.62, zorder=3,
            color=[PAL["box"] if v > 0 else PAL["boxneg"] for v in d.rho])
    ax.scatter(d.ev, y, s=30, facecolor="none", edgecolor=GHOST, linewidths=1.4,
               zorder=4, label="the same answer read as round(EV)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}\n{b}" for a, b in zip(d.label, d.twin)],
                       fontsize=8.5, color=FG)
    for i, v in zip(y, d.rho):
        ax.text(v + (.012 if v > 0 else -.012), i, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", color=FG, fontsize=9.5)
    ax.axvline(0, color=PAL["spine"], lw=1)
    ax.set_xlim(-.18, float(d.rho.max()) * 1.20)
    ax.set_ylim(-.7, len(d) - .3)
    ax.set_xlabel("Spearman rho against the measured pixel share",
                  color=FG, fontsize=10.5)
    ax.tick_params(colors=MUT, labelsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(PAL["spine"])
    ax.grid(axis="x", color=PAL["grid"], lw=.8, zorder=0)
    ax.legend(loc="lower right", facecolor=BG, edgecolor=PAL["spine"],
              labelcolor=FG, fontsize=9)
    miss = R[R.rho.isna()]
    if len(miss):
        ax.text(.005, .015, "  ".join(f"{m.label}: {m.twin}"
                                      for _, m in miss.iterrows()),
                transform=ax.transAxes, color=PAL["boxneg"], fontsize=9)
    fig.tight_layout()
    o1 = RES / "figures" / "validation_rho.png"
    fig.savefig(o1, dpi=args.dpi, facecolor=BG)
    print(f"\nwrote {o1}")

    # ---- the ladders ------------------------------------------------------
    have = [r for r in rows if r["ladder"] is not None]
    fig, axes = plt.subplots(3, 3, figsize=size, facecolor=BG)
    for axx, r in zip(axes.ravel(), have):
        axx.set_facecolor(BG)
        keep = [(k, x) for k, x in zip(range(1, 8), r["ladder"]) if len(x) >= 8]
        if keep:
            bp = axx.boxplot([x for _, x in keep], positions=[k for k, _ in keep],
                             widths=.62, patch_artist=True, showfliers=False,
                             medianprops=dict(color=FG, lw=1.3),
                             whiskerprops=dict(color=PAL["spine"]),
                             capprops=dict(color=PAL["spine"]))
            for b in bp["boxes"]:
                b.set(facecolor=PAL["face"] if r["rho"] > 0 else PAL["faceneg"],
                      edgecolor=PAL["spine"], lw=.9)
        axx.set_title(f"{r['label']}    rho {r['rho']:+.2f}", color=FG,
                      fontsize=9.5, pad=5)
        axx.set_xlim(.4, 7.6)
        axx.set_xticks(range(1, 8))
        axx.tick_params(colors=MUT, labelsize=7.5)
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            axx.spines[s].set_color(PAL["spine"])
    for axx in axes.ravel()[len(have):]:
        axx.set_visible(False)
    fig.text(.5, .014, "rated rung", color=FG, ha="center", fontsize=10.5)
    fig.text(.009, .5, "measured pixel share of the frame", color=FG,
             va="center", rotation=90, fontsize=10.5)
    fig.tight_layout(rect=[.024, .036, 1, 1])
    o2 = RES / "figures" / "validation_ladders.png"
    fig.savefig(o2, dpi=args.dpi, facecolor=BG)
    print(f"wrote {o2}")


if __name__ == "__main__":
    main()
