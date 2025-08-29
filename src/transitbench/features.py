from __future__ import annotations
import numpy as np
from typing import Dict, Optional

def basic_features(time: np.ndarray, flux: np.ndarray, mask: Optional[np.ndarray]=None) -> Dict[str, float]:
    if mask is None:
        mask = np.isfinite(time) & np.isfinite(flux)
    t = time[mask]; f = flux[mask]
    mu = np.nanmedian(f)
    s = 1.4826 * np.nanmedian(np.abs(f - mu))
    if not np.isfinite(s) or s == 0: s = np.nanstd(f)
    # simple red-noise proxy: 10-point bin test
    if f.size >= 20:
        b = 10
        fb = f[:(f.size//b)*b].reshape(-1,b).mean(axis=1)
        mu_b = np.nanmedian(fb)
        s_b = 1.4826 * np.nanmedian(np.abs(fb - mu_b))
        if not np.isfinite(s_b) or s_b == 0: s_b = np.nanstd(fb)
        beta10 = s_b / (s/np.sqrt(b)) if s>0 else np.nan
    else:
        beta10 = np.nan
    return {"rms_oot": float(s), "beta10": float(beta10), "n_pts": int(f.size)}
