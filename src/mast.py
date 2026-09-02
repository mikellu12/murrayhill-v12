"""Find and remove Google's camera mast from a Street View frame.

Every svi_90 frame carries the capture vehicle's mast rising from the bottom
edge, with the Google wordmark on its upper right. Both segmenters label it and
neither has a class for it: masking it removes 47% of Mapillary's Pole pixels,
60% of ADE20K's pole, and 29% of ADE20K's signboard -- that last being the
wordmark read as a sign. The VLM is fooled differently: erasing the mast moves
signage_detail by -0.104 of a rung (Wilcoxon p<0.0001, n=47) while
walkable_ground and vertical_hardscape do not move at all. The segmenters see
the object, the VLM sees the text.

DETECTION IS MODEL-FREE, deliberately. Keying off Pole would miss the frames
where ADE20K called it signboard, and vice versa. Three measured properties do
the work instead, none of which depend on where Google mounted the camera:

  bottom-anchored   the blob reaches the bottom edge of the frame. A real pole
                    is detected as a blob floating higher up: across 143 blobs,
                    85% of those stopping below 20% height touched the bottom
                    and 0% of every other height band did.
  fixed height      16.1% of frame height, p25-p75 15.7-16.3.
  fixed width       5.4% of frame width, p25-p75 5.4-5.5. The lit face and the
                    wordmark extend right to about 14% in total.

THE MASK IS A FIXED-SIZE RECTANGLE AT THE DETECTED ANCHOR, not a bounding box
of the detected pixels. Sizing it from the blob meant that when the threshold
caught a shadow beside the mast the mask spanned both and erased up to 19.2% of
the frame -- three frames hit exactly that ceiling. The mast is a camera part:
its size is a constant and only its position moves.

THE ANCHOR IS THE LEFTMOST QUALIFYING BLOB, not the largest. A low-contrast
frame breaks the mast into slivers -- one had 14 spanning x=196..267 -- and the
biggest fragment is not the leftmost, which anchored the rectangle 3.9% of the
frame too far right and left the dark column uncovered.

CALIBRATION IS PER IMAGERY SET. W_FRAC and H_FRAC are svi_90 numbers. The mast
subtends a fixed ANGLE, so its share of the frame depends on the field of view:
svi_180 (1440x916, twice the horizontal angle) carries TWO masts, at x~8% and
x~57%, both narrower. To recalibrate for a new set, average a few hundred
frames -- fixed elements stay sharp while streetscape blurs away, which is how
these numbers were measured in the first place. It needs no labels and takes
about a minute.

    from mast import mast_mask, erase_mast
"""
import numpy as np
from scipy import ndimage

# Per imagery set. The mast subtends a fixed ANGLE, so its share of the frame
# scales with the field of view: svi_180 covers twice the horizontal angle in
# the same pixel width, and its mast measures 2.71% wide against svi_90's 5.4%
# -- almost exactly half, which is the check that these are the same object and
# not two different calibrations. Height barely moves (16.27% against 16.1%)
# because the vertical field of view is unchanged.
#
# `masts` is how many the projection puts in frame. svi_180 is a cylindrical
# strip spanning enough of the panorama to include the camera part twice, at
# x ~ 5% and x ~ 55%; svi_90 carries one.
# svi_90_wide is the 180-degree strip a pedestrian way gets from
# export_svi_90, at 2880x1833. It is NOT svi_180, which is the older
# 1440x916 export at half the angular resolution. The numbers here are
# derived, not separately measured, and the derivation is the whole point of
# the per-set table: the mast is one object of one angular size, so widening
# the field of view from 90 to 180 degrees at the same pixels per degree
# halves its share of the frame width and leaves its share of the height
# alone, the vertical field being unchanged. Two masts rather than one
# because a 180-degree strip spans two of the source frames' seams.
#
# Confirm rather than assume for a new city: tools/mast_calibrate.py measures
# these from the imagery itself in about a minute, and the check that matters
# is whether the capture rig is the same object, not whether the arithmetic
# is right.
SETS = {
    # 1440x1833, 90 degrees, 16 px/deg -- one mast
    "svi_90":  dict(w=0.140, h=0.170, band=0.22, max_w=0.12, masts=1),
    # 2880x1833, 180 degrees, 16 px/deg -- two, at half the fractional width
    "svi_180": dict(w=0.070, h=0.170, band=0.22, max_w=0.06, masts=2),
}
# svi_90_wide was this entry's name while the 180 tree still held the older
# 1440x916 render and the name svi_180 was taken by it. That render no longer
# exists -- data/raw/svi_180 is 2880x1833 throughout -- so the set is named for
# the geometry it describes. The alias keeps older commands working.
SETS["svi_90_wide"] = SETS["svi_180"]
DEFAULT = "svi_90"

DARK = 0.55        # fraction of the band's median luminance
MIN_PX = 0.0003    # of the search band, to ignore specks
MIN_H = 0.45       # of the search band


def _cfg(name):
    if name not in SETS:
        raise KeyError(f"no mast calibration for {name!r}; "
                       f"have {sorted(SETS)}. Measure one by averaging a few "
                       f"hundred frames -- fixed elements stay sharp.")
    return SETS[name]


def anchors(im, sets=DEFAULT):
    """Left edge and top of each mast, left to right.

    Returns up to `masts` entries. Fragments are merged by proximity: a
    low-contrast frame breaks the column into slivers -- one svi_90 frame had
    14 spanning x=196..267 -- and each is a separate blob, so anything within
    the calibrated width of an existing anchor extends it rather than starting
    a new mast.
    """
    c = _cfg(sets)
    a = np.asarray(im.convert("L"), float) / 255.0
    H, W = a.shape
    lo = int(H * (1 - c["band"]))
    sub = a[lo:]
    lab, n = ndimage.label(sub < np.median(sub) * DARK)
    # Bounding boxes and pixel counts for every blob in one pass each. The
    # obvious loop -- `for k in range(1, n+1): np.where(lab == k)` -- rescans
    # the whole band per blob, and a low-contrast band has hundreds, so it cost
    # 1.2 s per 1440-wide frame and 2.9 s per 2880-wide one. That is CPU time
    # with the GPU idle: about two hours across a full rating pass.
    boxes = ndimage.find_objects(lab)
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    found = []
    for k, box in enumerate(boxes, start=1):
        if box is None or counts[k] < MIN_PX * sub.size:
            continue
        ys, xs = box
        h = ys.stop - ys.start
        if (ys.stop - 1 >= sub.shape[0] - 2
                and h / sub.shape[0] > MIN_H
                and (xs.stop - xs.start) / W < c["max_w"]):
            found.append((int(xs.start), lo + int(ys.start)))
    found.sort()
    merged = []
    for x, y in found:
        if merged and x - merged[-1][0] < c["w"] * W:
            merged[-1] = (merged[-1][0], min(merged[-1][1], y))
        else:
            merged.append((x, y))
    return merged[:c["masts"]]


def mast_mask(im, sets=DEFAULT):
    """Boolean mask of every mast in frame, all False when none is found."""
    c = _cfg(sets)
    H, W = np.asarray(im).shape[:2]
    m = np.zeros((H, W), bool)
    for x, ytop in anchors(im, sets):
        x0 = max(0, int(x - 0.004 * W))
        x1 = min(W, int(x0 + c["w"] * W))
        y0 = max(0, min(int(H - c["h"] * H), ytop - int(0.005 * H)))
        m[y0:, x0:x1] = True
    return m


def erase_mast(im, sets=DEFAULT):
    """Paint the mast out in the median colour of the surrounding band.

    Painted rather than cropped: cropping changes the field of view, which
    would alter every share in the frame as well as the mast's.
    """
    from PIL import Image
    a = np.asarray(im.convert("RGB")).copy()
    m = mast_mask(im, sets)
    if not m.any():
        return im, m
    band = a[int(a.shape[0] * 0.80):]
    a[m] = np.median(band.reshape(-1, 3), axis=0).astype(np.uint8)
    return Image.fromarray(a), m
