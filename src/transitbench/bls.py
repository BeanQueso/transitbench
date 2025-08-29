from __future__ import annotations
import numpy as np
from astropy.timeseries import BoxLeastSquares

def _robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s == 0:
        s = np.nanstd(x)
    return float(s)

def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray | None = None,
    durations: np.ndarray | float | None = None,
):
    """
    Run BLS and return best parameters.

    - Works across Astropy versions (does not rely on power.snr existing).
    - Computes an SNR estimate using med(out) - med(in) centered at phase 0,
      which is BLS's convention for transit centers.
    """
    m = np.isfinite(time) & np.isfinite(flux)
    t = time[m]; f = flux[m]

    if periods is None:
        periods = np.linspace(0.5, 20.0, 3000)
    if durations is None:
        durations = np.array([0.05, 0.1, 0.2])  # days

    bls = BoxLeastSquares(t, f)
    power = bls.power(periods, durations)
    i = int(np.nanargmax(power.power))

    period   = float(power.period[i])
    t0       = float(power.transit_time[i])   # transit center
    duration = float(power.duration[i])
    depth    = float(power.depth[i])
    metric   = float(power.power[i])          # raw BLS “power”

    # ---- Version-agnostic SNR centered at phase 0 ----
    phase = ((t - t0) / period) % 1.0
    # distance to nearest integer phase (0 or 1)
    dist = np.minimum(phase, 1.0 - phase)
    half_w = 0.5 * duration / period
    in_tr  = dist <= half_w
    out_tr = ~in_tr

    f_in, f_out = f[in_tr], f[out_tr]
    sigma_oot = _robust_std(f_out)
    depth_est = np.nanmedian(f_out) - np.nanmedian(f_in)  # positive for dips
    Nin = int(in_tr.sum())

    if np.isfinite(sigma_oot) and sigma_oot > 0 and Nin > 0 and np.isfinite(depth_est):
        snr_est = float((depth_est / sigma_oot) * np.sqrt(Nin))
    else:
        snr_est = float("nan")

    return {
        "period": period,
        "t0": t0,
        "duration": duration,
        "depth": depth,
        "snr": snr_est,          # computed SNR (positive for transit-like dips)
        "power_value": metric,   # raw BLS power at peak
        "power_array": power.power,
        "periods": np.asarray(periods),
    }
