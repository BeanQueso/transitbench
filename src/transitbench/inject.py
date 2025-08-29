from __future__ import annotations
import numpy as np

def inject_box_transit(time, flux, depth=0.005, duration=0.1, period=None, t0=None, copy=True):
    t = np.asarray(time)
    f = np.array(flux, dtype=float, copy=True) if copy else flux
    if t0 is None: t0 = np.nanmedian(t)
    if period is None:
        in_tr = (np.abs(t - t0) <= 0.5*duration)
        f[in_tr] *= (1.0 - depth)
        return f
    phase = ((t - t0) / period) % 1.0
    in_tr = (phase > 0.5 - 0.5*duration/period) & (phase < 0.5 + 0.5*duration/period)
    f[in_tr] *= (1.0 - depth)
    return f
