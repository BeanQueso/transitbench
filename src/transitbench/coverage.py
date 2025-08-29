from __future__ import annotations
import numpy as np

def _as_array(x):
    return np.asarray(x, dtype=float)

# src/transitbench/coverage.py
import numpy as np

def phase_coverage(time, period, duration, *, t_ref=None, n_bins: int = 2000):
    """
    Returns (coverage_fraction, n_transits_observed).
    coverage ≈ fraction of phase bins (width ~ duration/period) that contain data.
    """
    t = np.asarray(time, float)
    fin = np.isfinite(t)
    if fin.sum() < 2 or not np.isfinite(period) or period <= 0 or not np.isfinite(duration) or duration <= 0:
        return float("nan"), 0

    t = t[fin]
    # Phase relative to reference
    tref = float(t_ref) if (t_ref is not None and np.isfinite(t_ref)) else 0.5 * (np.nanmin(t) + np.nanmax(t))
    phase = ((t - tref) / period) % 1.0

    # Bin width ~ duration/period; cap bins
    binw = max(min(duration / period, 0.25), 1.0 / max(n_bins, 10))
    nb = int(np.clip(np.round(1.0 / binw), 10, 10000))
    bins = np.linspace(0, 1, nb + 1)

    # Count occupancy
    idx = np.digitize(phase, bins) - 1
    idx = np.clip(idx, 0, nb - 1)
    occ = np.zeros(nb, dtype=bool)
    occ[np.unique(idx)] = True
    coverage = float(occ.mean())

    # crude n_transits observed from span
    span = np.nanmax(t) - np.nanmin(t)
    n_trans = int(max(0, np.floor(span / period)))

    return coverage, n_trans


def n_transits_observed(
    time, period, t0, duration, min_points: int = 1
) -> int:
    """
    Count how many distinct transit windows [t_k - dur/2, t_k + dur/2] contain at least
    `min_points` cadences, where t_k = t0 + k*period intersects the observed time span.
    """
    t = _as_array(time)
    t = t[np.isfinite(t)]
    if t.size == 0 or not np.isfinite(period) or period <= 0 or not np.isfinite(duration) or duration <= 0:
        return 0
    if not np.isfinite(t0):
        t0 = float(np.nanmedian(t))

    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))

    k_min = int(np.ceil((tmin - (t0 + duration / 2.0)) / period))
    k_max = int(np.floor((tmax - (t0 - duration / 2.0)) / period))
    if k_max < k_min:
        return 0

    count = 0
    for k in range(k_min, k_max + 1):
        center = t0 + k * period
        lo = center - duration / 2.0
        hi = center + duration / 2.0
        m = np.count_nonzero((t >= lo) & (t <= hi))
        if m >= min_points:
            count += 1
    return int(count)
