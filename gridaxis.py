"""
Position along the Manhattan grid's uptown axis, in metres.

WHY THIS EXISTS
---------------
It replaces the three named `zone` categories. Those were a latitude cut
across a grid rotated ~29 degrees east of true north, so the boundary ran
diagonally across the street pattern and split every one of the fifteen
streets between two zones -- East 38th Street had 43 nodes in one zone and
1 in another. The names implied a neighbourhood boundary that no source
draws.

What the zones were actually encoding is a monotone north-south gradient in
greenness (rho = -0.49 at node level, -0.46 across the 22 block faces).
That is worth keeping. It is a continuous quantity, so it is kept as one.

WHY NOT JUST USE LATITUDE
-------------------------
Same reason the zones failed. Uptown on this grid is bearing 029, not 000,
so latitude is the gradient projected onto the wrong axis and loses about
13% of its length to the rotation. `grid_northing` projects onto 029, which
is the direction the avenues actually run and the direction a pedestrian
means by "uptown".

PROJECTION
----------
Local equirectangular about the frame centroid rather than a full UTM
transform, so this needs only numpy and can be called from the dashboard,
which has no geopandas. Over a 1.5 km frame the error against EPSG:32618 is
well under a metre -- far below the 20 m node spacing.

    from gridaxis import grid_northing
    m["northing_m"] = grid_northing(m.lat, m.lon)
"""
import numpy as np

GRID_BEARING_DEG = 29.0        # config.yaml: directional.grid_bearing
_R_LAT = 110540.0              # metres per degree latitude
_R_LON = 111320.0              # metres per degree longitude at the equator


def local_xy(lat, lon, lat0=None, lon0=None):
    """Metres east and north of a local origin."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = float(np.nanmean(lat)) if lat0 is None else lat0
    lon0 = float(np.nanmean(lon)) if lon0 is None else lon0
    x = (lon - lon0) * _R_LON * np.cos(np.radians(lat0))
    y = (lat - lat0) * _R_LAT
    return x, y


def grid_northing(lat, lon, bearing_deg=GRID_BEARING_DEG, origin="min"):
    """Metres along the uptown axis (bearing 029), increasing uptown.

    origin="min"  -> zero at the southernmost node, so the value reads as
                     distance uptown from the bottom of the frame.
    origin="mean" -> centred on zero, which is the better scaling for a
                     regression intercept.
    """
    x, y = local_xy(lat, lon)
    t = np.radians(bearing_deg)
    u = x * np.sin(t) + y * np.cos(t)
    if origin == "min":
        return u - np.nanmin(u)
    if origin == "mean":
        return u - np.nanmean(u)
    return u


def grid_easting(lat, lon, bearing_deg=GRID_BEARING_DEG, origin="min"):
    """The cross-street axis (bearing 119), for symmetry. Increasing east."""
    x, y = local_xy(lat, lon)
    t = np.radians(bearing_deg)
    v = x * np.cos(t) - y * np.sin(t)
    if origin == "min":
        return v - np.nanmin(v)
    if origin == "mean":
        return v - np.nanmean(v)
    return v


def add_axes(df, lat="lat", lon="lon", origin="min"):
    """Attach northing_m and easting_m if the frame carries coordinates."""
    if lat not in df.columns or lon not in df.columns:
        return df
    df = df.copy()
    df["northing_m"] = grid_northing(df[lat], df[lon], origin=origin)
    df["easting_m"] = grid_easting(df[lat], df[lon], origin=origin)
    return df
