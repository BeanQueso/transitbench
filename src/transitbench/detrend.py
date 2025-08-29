from __future__ import annotations
import numpy as np

def _robust_median_trend(time: np.ndarray, flux: np.ndarray, window_days: float, min_pts: int = 5) -> np.ndarray:
    """
    Simple rolling-median trend in the time domain (no external deps).
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    n = t.size
    trend = np.full(n, np.nan, dtype=float)

    if n == 0 or window_days <= 0:
        return np.ones_like(f, dtype=float)

    half = 0.5 * window_days
    for i in range(n):
        ti = t[i]
        left = ti - half
        right = ti + half
        mask = (t >= left) & (t <= right) & np.isfinite(f)
        if mask.sum() >= min_pts:
            trend[i] = np.nanmedian(f[mask])

    return trend

def _interpolate_nan_edges(y: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation; extend edges with nearest finite."""
    x = np.arange(y.size, dtype=float)
    out = y.astype(float).copy()
    good = np.isfinite(out)
    if good.sum() == 0:
        return np.ones_like(out)  # total fallback
    out[~good] = np.interp(x[~good], x[good], out[good])
    return out

def rolling_median_highpass(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float = 0.5,
    min_pts: int = 5,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    High-pass by dividing flux by a rolling-median trend.
    Robust to zeros/NaNs: interpolates trend, floors by eps, then renormalizes median to ~1.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)

    trend = _robust_median_trend(t, f, window_days, min_pts=min_pts)

    bad = ~np.isfinite(trend) | (trend <= 0)
    if np.any(bad):
        trend = _interpolate_nan_edges(trend)
        trend[~np.isfinite(trend)] = 1.0
        trend[trend <= 0] = 1.0

    trend = np.maximum(trend, eps)
    hp = f / trend

    med = np.nanmedian(hp[np.isfinite(hp)])
    if np.isfinite(med) and med != 0:
        hp = hp / med

    bad_hp = ~np.isfinite(hp)
    if np.any(bad_hp):
        hp[bad_hp] = 1.0

    return hp
