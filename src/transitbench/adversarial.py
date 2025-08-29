from __future__ import annotations
from typing import Dict, Any, List, Iterable, Tuple, Optional
import numpy as np

# --- basic transit mask/injection (box) ---

def in_transit_mask(time: np.ndarray, period: float, t0: float, duration: float, pad: float = 0.0) -> np.ndarray:
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    half = 0.5 * duration * (1.0 + pad)
    return np.abs(phase) < half

def inject_box(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float, depth: float) -> np.ndarray:
    f = np.array(flux, float, copy=True)
    m = in_transit_mask(time, period, t0, duration, pad=0.0)
    f[m] *= (1.0 - float(depth))
    return f

# --- adversarial profiles ---

def apply_profile(
    time: np.ndarray,
    base_flux: np.ndarray,
    *,
    period: float, t0: float, duration: float, depth: float,
    profile: str = "vanilla",
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Returns a *new* flux array with the adversarial profile applied.
    Profiles:
      - 'vanilla'           : simple box injection
      - 'odd-even'          : alternate depths per epoch (param: odd_even_delta, default 0.15)
      - 'ttv'               : epoch-to-epoch timing jitter (param: ttv_amp_days, default = 0.5*duration)
      - 'gap'               : drop a fraction of transit windows (param: gap_frac, default 0.5)
      - 'spot-sine'         : sinusoidal modulation added (param: spot_amp, spot_period_days)
      - 'harmonic-confuser' : inject near harmonics of a given base period (param: base_period, ratio in {0.5,2.0,1.0±ε})
    """
    params = dict(params or {})
    t = np.asarray(time, float)
    f = np.array(base_flux, float, copy=True)

    prof = (profile or "vanilla").lower()
    if prof == "vanilla":
        return inject_box(t, f, period, t0, duration, depth)

    if prof == "odd-even":
        delta = float(params.get("odd_even_delta", 0.15))  # ±15% depth alternation
        # epoch index for each sample
        k = np.floor((t - t0) / period + 0.5).astype(int)
        # build per-epoch depth
        even = (k % 2 == 0)
        depth_per_sample = np.where(even, depth * (1.0 + delta), depth * (1.0 - delta))
        m = in_transit_mask(t, period, t0 + 0.0, duration, pad=0.0)
        f[m] *= (1.0 - depth_per_sample[m])
        return f

    if prof == "ttv":
        amp = float(params.get("ttv_amp_days", 0.5 * duration))  # peak jitter ~ 0.5*duration
        phi = float(params.get("ttv_phase", 0.0))
        # for each epoch center, shift by a sinusoid
        tmin, tmax = t.min(), t.max()
        k0 = int(np.floor((tmin - t0) / period)) - 1
        k1 = int(np.ceil((tmax - t0) / period)) + 1
        f = f.copy()
        for j, k in enumerate(range(k0, k1 + 1)):
            center = t0 + k * period + amp * np.sin(2 * np.pi * j / max(1, (k1 - k0)) + phi)
            m = np.abs(t - center) < (0.5 * duration)
            f[m] *= (1.0 - depth)
        return f

    if prof == "gap":
        frac = float(params.get("gap_frac", 0.5))  # remove ~50% of transit windows
        rng = np.random.default_rng(int(params.get("seed", 0)))
        tmin, tmax = t.min(), t.max()
        k0 = int(np.floor((tmin - t0) / period)) - 1
        k1 = int(np.ceil((tmax - t0) / period)) + 1
        keep_mask = np.ones_like(t, dtype=bool)
        for k in range(k0, k1 + 1):
            if rng.random() > frac:
                # keep this transit; inject it
                center = t0 + k * period
                m = np.abs(t - center) < (0.5 * duration)
                f[m] *= (1.0 - depth)
            else:
                # drop this transit window entirely
                center = t0 + k * period
                m = np.abs(t - center) < (0.5 * duration)
                keep_mask[m] = False
        f = f.copy()
        f[~keep_mask] = np.nan
        return f

    if prof == "spot-sine":
        amp = float(params.get("spot_amp", 0.01))            # 1% modulation
        pspot = float(params.get("spot_period_days", 2.0))   # default 2 d
        fmod = (1.0 + amp * np.sin(2 * np.pi * (t - t0) / max(pspot, 1e-6)))
        f = f * fmod
        return inject_box(t, f, period, t0, duration, depth)

    if prof == "harmonic-confuser":
        baseP = float(params.get("base_period", period))
        ratio = float(params.get("ratio", 0.5))  # default inject at P/2
        per2 = baseP * ratio
        eps = float(params.get("epsilon", 0.0))
        per2 = per2 * (1.0 + eps)  # slight offset if given
        return inject_box(t, f, per2, t0, duration, depth)

    # fallback
    return inject_box(t, f, period, t0, duration, depth)


def make_adversarial_grid(
    time: np.ndarray,
    *,
    depths: Tuple[float, ...] = (0.003, 0.005, 0.01),
    durations: Tuple[float, ...] = (0.08, 0.12, 0.20),
    periods: Tuple[float, ...] = (1.5, 3.0, 5.0),
    profiles: Tuple[str, ...] = ("vanilla", "odd-even", "ttv"),
    params_by_profile: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of injection dicts with keys:
      {'depth','duration','period','profile','params'}.
    """
    params_by_profile = dict(params_by_profile or {})
    out: List[Dict[str, Any]] = []
    for d in depths:
        for u in durations:
            for p in periods:
                for prof in profiles:
                    out.append({
                        "depth": float(d),
                        "duration": float(u),
                        "period": float(p),
                        "profile": prof,
                        "params": params_by_profile.get(prof, {}),
                    })
    return out
