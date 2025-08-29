# src/transitbench/recover.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from . import utils
from tqdm import tqdm
import os, json, hashlib
import multiprocessing as mp
import json as _json
from pathlib import Path

# Optional dependency: astropy BLS. We keep a graceful fallback error.
try:
    from astropy.timeseries import BoxLeastSquares
    _HAS_ASTROPY = True
except Exception:
    _HAS_ASTROPY = False


# ---------------------------
# Profile presets (decision policy)
# ---------------------------
# Minimum per-point uncertainty to avoid BLS z blow-ups (unitless, in flux)
DY_FLOOR = 1e-3        # minimum per-point uncertainty fed to BLS
Z_LIKE_CAP = 50.0   # cap for z-like values (to avoid numerical issues)
PROFILES = {
    # FPR targets are *per trial*; the helper will translate these into z-thresholds.
    "sensitive": {
        "decision_metric": "zscore",
        "tau_base": { "oot-replace": 7.861611950481118, "prewhiten": 12.203525095703986 },
        "rel_window": 0.020,
        "fpr_target": 0.025,      # ~2.5 %
        "beta_scale": "soft",     # "none"|"soft"|"linear"
        "method_pref": "auto",    # "auto"|"oot-replace"|"prewhiten"
    },
    "balanced": {
        "decision_metric": "zscore",
        "tau_base": { "oot-replace": 10.032698831999504, "prewhiten": 16.0 },
        "rel_window": 0.015,
        "fpr_target": 0.010,
        "beta_scale": "soft",
        "method_pref": "auto",
    },
    "strict": {
        "decision_metric": "zscore",
        "tau_base": { "oot-replace": 12.299699709689188, "prewhiten": 20.0 },
        "rel_window": 0.010,
        "fpr_target": 0.005,
        "beta_scale": "linear",
        "method_pref": "auto",
    },
}

def apply_profile_overrides(path: str | None = None, overrides: dict | None = None) -> bool:
    """
    Merge profile overrides into PROFILES. Expects a dict like:
    {"suggest_tau_base": {"oot-replace": {"balanced": 3000, ...}, "prewhiten": {...}}}
    Returns True if anything changed.
    """
    try:
        if overrides is None:
            if path is None:
                # Try several likely locations for profile_overrides.json:
                #   repo root (…/profile_overrides.json),
                #   alongside src (…/src/profile_overrides.json),
                #   or next to this file.
                _here = Path(__file__).resolve()
                _parents = list(_here.parents)
                _candidates = []
                # repo root is commonly 3 levels up from recover.py: …/src/transitbench/recover.py -> …/
                for i in range(1, min(5, len(_parents))):
                    _candidates.append(_parents[i] / "profile_overrides.json")
                _candidates.append(_here.parent / "profile_overrides.json")
                for _cand in _candidates:
                    if _cand.exists():
                        path = str(_cand)
                        break
            if path is None:
                raise FileNotFoundError("profile_overrides.json not found in candidate locations")
            with open(path, "r") as f:
                overrides = _json.load(f)
    except FileNotFoundError:
        return False
    except Exception:
        return False

    sugg = overrides.get("suggest_tau_base") or overrides.get("tau_base") or {}
    changed = False
    if not isinstance(globals().get("PROFILES", None), dict):
        globals()["PROFILES"] = {}
    for method, prof_map in sugg.items():
        for prof_name, tau in prof_map.items():
            if prof_name not in PROFILES:
                PROFILES[prof_name] = {}
            prof = PROFILES[prof_name]
            if not isinstance(prof.get("tau_base"), dict):
                prof["tau_base"] = {}
            prof["tau_base"][method] = float(tau)
            changed = True
    return changed

# Auto-apply overrides file at repo root, if present
try:
    apply_profile_overrides()
except Exception:
    pass

@dataclass
class CostModel:
    """Tiny container to parameterize budget consumption."""
    c_clean: float = 0.05   # “fixed” cost per cleaning pass
    c_bls: float = 0.002    # cost per (period × duration) evaluation

    def estimate(self, n_periods: int, n_durations: int) -> float:
        return float(self.c_clean + self.c_bls * n_periods * n_durations)

# ---------------------------
# Small utilities
# ---------------------------
def _finite_mask(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _nanmedian(x):
    return np.nanmedian(x) if np.size(x) else np.nan


def _local_trend(t, f, window_days=0.75):
    """
    Robust moving median trend using a two-pointer sliding window (O(N)).
    Works on *sorted* time; we sort internally and unsort back.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)

    idx = np.argsort(t)
    t_sorted = t[idx]
    f_sorted = f[idx]

    n = t_sorted.size
    if n == 0:
        return np.full_like(f, np.nan)

    half = 0.5 * window_days
    left = 0
    right = 0
    trend_sorted = np.empty(n, float)

    # Use a deque-like approach via indices. We’ll recompute median from slice;
    # this is O(n * w log w) in worst case but fine for typical TESS lengths.
    for i in range(n):
        ti = t_sorted[i]
        # expand right to include points within window
        while right < n and t_sorted[right] <= ti + half:
            right += 1
        # shrink left to exclude points left of window
        while left < n and t_sorted[left] < ti - half:
            left += 1
        window = f_sorted[left:right]
        # robust median ignoring NaNs
        trend_sorted[i] = _nanmedian(window)

    # Interpolate over any NaNs in the trend
    bad = ~np.isfinite(trend_sorted)
    if np.any(bad):
        good = ~bad
        if np.any(good):
            trend_sorted[bad] = np.interp(t_sorted[bad], t_sorted[good], trend_sorted[good])
        else:
            trend_sorted[:] = 1.0

    # Undo sort
    inv = np.empty_like(idx)
    inv[idx] = np.arange(n)
    trend = trend_sorted[inv]
    return trend


def _find_transit_mask(t, period, t0, duration, pad=1.0):
    """
    Boolean mask for points within +/- 0.5*duration*pad of transit centers.
    """
    t = np.asarray(t, float)
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0) and np.isfinite(duration) and duration > 0):
        return np.zeros_like(t, dtype=bool)

    # Choose k range that covers [tmin, tmax]
    tmin, tmax = np.nanmin(t), np.nanmax(t)
    k0 = int(np.floor((tmin - t0) / period)) - 1
    k1 = int(np.ceil((tmax - t0) / period)) + 1

    mask = np.zeros_like(t, dtype=bool)
    half = 0.5 * duration * pad
    for k in range(k0, k1 + 1):
        tc = t0 + k * period
        in_win = (t >= (tc - half)) & (t <= (tc + half))
        mask |= in_win
    return mask


def _apply_profile(t, f, *, period, t0, duration, depth, profile="box"):
    """
    Multiply flux by (1 - depth) inside the transit window (simple box model).
    """
    y = np.array(f, float).copy()
    if not (np.isfinite(depth) and depth >= 0):
        return y
    if not (np.isfinite(period) and period > 0 and np.isfinite(duration) and duration > 0 and np.isfinite(t0)):
        return y

    m = _find_transit_mask(t, period, t0, duration, pad=1.0)
    y[m] *= (1.0 - depth)
    return y


def _robust_sigma(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 1.0
    mad = np.nanmedian(np.abs(y - np.nanmedian(y)))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0:
        sig = np.nanstd(y)
    return float(sig if np.isfinite(sig) and sig > 0 else 1.0)


def _compute_baseline_bls(t, f, durations, n_periods=4000, min_period=None, max_period=None):
    if not _HAS_ASTROPY:
        raise RuntimeError("astropy is required for BLS; install astropy to use transitbench.")

    t = np.asarray(t, float); f = np.asarray(f, float)
    m = _finite_mask(t, f); t, f = t[m], f[m]
    if t.size < 10:
        return dict(period=np.nan, t0=np.nan, duration=np.nan, depth=np.nan, depth_err=np.nan, z_top=np.nan)

    med = _nanmedian(f); med = 1.0 if (not np.isfinite(med) or med == 0) else med
    y = f / med
    dy0 = max(_robust_sigma(y), DY_FLOOR)
    dy = np.full_like(y, dy0, dtype=float)
    if (not np.isfinite(dy0)) or (dy0 <= 0):
        dy0 = 1.0
    dy0 = max(dy0, DY_FLOOR)
    dy = np.full_like(y, dy0, dtype=float)

    if min_period is None:
        min_period = max(0.3, 2.0 * np.nanmedian(np.diff(np.unique(t))))
    if max_period is None:
        max_period = max(1.1 * (t.max() - t.min()) / 5.0, min_period + 0.5)

    periods = np.linspace(min_period, max_period, int(n_periods))
    bls = BoxLeastSquares(t, y, dy=dy)

    best = dict(period=np.nan, t0=np.nan, duration=np.nan, depth=np.nan, depth_err=np.nan, z_top=-np.inf)
    for dur in durations:
        if not (np.isfinite(dur) and dur > 0): 
            continue
        res = bls.power(periods, dur, objective="likelihood")
        # Preferred z from depth/depth_err; fallback to sqrt(power)
        z_like = res.depth / np.where(res.depth_err > 0, res.depth_err, np.inf)
        z_pow  = np.sqrt(np.clip(res.power, 0, np.inf))
        # guard against runaway z from tiny depth_err
        z_like_safe = np.where(np.isfinite(z_like) & (np.abs(z_like) < Z_LIKE_CAP), z_like, np.nan)
        z = np.where(np.isfinite(z_like_safe), z_like_safe, z_pow)

        i = int(np.nanargmax(z))
        if np.isfinite(z[i]) and z[i] > best["z_top"]:
            best.update(
                period=float(res.period[i]),
                t0=float(res.transit_time[i]),
                duration=float(dur),
                depth=float(res.depth[i]),
                depth_err=float(res.depth_err[i]),
                z_top=float(z[i]),
            )
    return best


def _bls_grid_local(t, y, periods, durations):
    if not _HAS_ASTROPY:
        raise RuntimeError("astropy is required for BLS; install astropy to use transitbench.")

    t = np.asarray(t, float); y = np.asarray(y, float)
    m = _finite_mask(t, y); t, y = t[m], y[m]
    if t.size < 10:
        return np.nan, np.nan, np.nan

    med = _nanmedian(y); med = 1.0 if (not np.isfinite(med) or med == 0) else med
    y = y / med
    dy0 = max(_robust_sigma(y), DY_FLOOR)
    dy = np.full_like(y, dy0, dtype=float)
    # Robust uncertainty with floor to prevent numerical blow-ups
    if (not np.isfinite(dy0)) or (dy0 <= 0):
        dy0 = 1.0
    dy0 = max(dy0, DY_FLOOR)
    dy = np.full_like(y, dy0, dtype=float)

    bls = BoxLeastSquares(t, y, dy=dy)

    z_best = -np.inf; p_best = np.nan; d_best = np.nan
    for dur in durations:
        if not (np.isfinite(dur) and dur > 0):
            continue
        res = bls.power(periods, dur, objective="likelihood")
        z_like = res.depth / np.where(res.depth_err > 0, res.depth_err, np.inf)
        z_pow  = np.sqrt(np.clip(res.power, 0, np.inf))
        z_like_safe = np.where(np.isfinite(z_like) & (np.abs(z_like) < Z_LIKE_CAP), z_like, np.nan)
        z = np.where(np.isfinite(z_like_safe), z_like_safe, z_pow)

        i = int(np.nanargmax(z))
        if np.isfinite(z[i]) and z[i] > z_best:
            z_best = float(z[i]); p_best = float(res.period[i]); d_best = float(dur)
    return z_best, p_best, d_best


def _snr_near_period(t, y, period, durations, rel_window=0.0, n_local=800):
    if not (np.isfinite(period) and period > 0):
        return np.nan, np.nan
    rel = float(rel_window)
    if rel <= 0: rel = 0.002
    p0 = period * (1.0 - rel); p1 = period * (1.0 + rel)
    periods = np.linspace(p0, p1, int(max(21, n_local)))
    z, p_best, _ = _bls_grid_local(t, y, periods, durations)
    return z, p_best


def _depth_snr_harmonics(t, y, p_inj, durations, rel_window=0.0):
    """
    Compute SNR around P, P/2, 2P and return the best (snr, match_label, match_period).
    """
    cand = []
    for lab, pp in [("P", p_inj), ("P/2", p_inj / 2.0), ("2P", 2.0 * p_inj)]:
        z, pbest = _snr_near_period(t, y, pp, durations, rel_window=rel_window)
        cand.append((z, lab, pbest))
    # Pick the highest z
    z_vals = [c[0] for c in cand]
    i = int(np.nanargmax(z_vals)) if np.any(np.isfinite(z_vals)) else 0
    z, lab, pbest = cand[i]
    return float(z) if np.isfinite(z) else np.nan, lab, float(pbest) if np.isfinite(pbest) else np.nan


def _n_transits_observed(t, period, duration, t_ref=None):
    """
    Count how many distinct transit *windows* contain data and the coverage fraction.
    We say a window is 'observed' if at least one datum falls in it.
    """
    t = np.asarray(t, float)
    if not (np.isfinite(period) and period > 0 and np.isfinite(duration) and duration > 0):
        return 0.0, 0

    tmin, tmax = np.nanmin(t), np.nanmax(t)
    # Pick t_ref to center windows; default to median time
    if not (np.isfinite(t_ref)):
        t_ref = np.nanmedian(t)

    k0 = int(np.floor((tmin - t_ref) / period)) - 1
    k1 = int(np.ceil((tmax - t_ref) / period)) + 1
    half = 0.5 * duration

    n_expected = 0
    n_obs = 0
    for k in range(k0, k1 + 1):
        tc = t_ref + k * period
        if tc + half < tmin or tc - half > tmax:
            continue
        n_expected += 1
        in_win = (t >= (tc - half)) & (t <= (tc + half))
        if np.any(in_win):
            n_obs += 1

    cov = float(n_obs) / float(n_expected) if n_expected > 0 else 0.0
    return cov, int(n_obs)

def _robust_std(y: np.ndarray) -> float:
    """Robust std via MAD (handles NaNs)."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    return 1.4826 * mad  # MAD->sigma

def _beta_n(y: np.ndarray, N: int = 10) -> float:
    """
    Pont+06 beta factor: ratio of observed std of N-binned residuals
    to white-noise expectation (sigma_1 / sqrt(N)).
    Uses simple consecutive binning; ignores gaps.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < max(2*N, 20):
        return float("nan")

    sigma1 = _robust_std(y)
    # consecutive binning
    m = (n // N) * N
    if m < N:
        return float("nan")
    y_trim = y[:m].reshape(-1, N)
    bin_means = np.nanmean(y_trim, axis=1)
    sigmaN = _robust_std(bin_means)
    expected = sigma1 / np.sqrt(N) if sigma1 > 0 else float("nan")

    if not np.isfinite(sigmaN) or not np.isfinite(expected) or expected == 0:
        return float("nan")
    return float(sigmaN / expected)

def basic_features(t: np.ndarray, f: np.ndarray) -> dict:
    """
    Compute quick headline stats for a light curve:
      - n_pts   : number of finite samples
      - rms_oot : robust RMS of (flux / median - 1)
      - beta10  : Pont beta factor at N=10
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    n_pts = int(t.size)

    if n_pts == 0:
        return {"n_pts": 0, "rms_oot": float("nan"), "beta10": float("nan")}

    med = np.nanmedian(f)
    # Normalize so continuum ~ 0
    y = (f / med - 1.0) if med not in (0.0, np.nan) else (f - np.nanmedian(f))
    rms = _robust_std(y)
    beta10 = _beta_n(y, N=10)

    return {"n_pts": n_pts, "rms_oot": float(rms), "beta10": float(beta10)}

# ---------------------------
# Cleaning strategies
# ---------------------------
def get_clean_series(
    t, f,
    method="oot-replace",
    *,
    window_days=0.75,
    oot_strategy="median",   # for oot-replace: "median" | "resample"
    pad=1.2,                 # widen transit window when removing real transit
    n_periods=4000,
    durations_for_baseline=(0.12, 0.20),   # used to identify the real transit
    label=None,
    **kwargs
):
    """
    Produce a cleaned series (t_clean, f_clean, meta).
    - prewhiten: robust trend removal; trend is fit with in-transit windows masked (from baseline BLS).
    - oot-replace: remove the strongest real transit by replacing in-transit samples with OOT-like flux.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    m = _finite_mask(t, f)
    t, f = t[m], f[m]
    if t.size == 0:
        return t, f, {"mode": method, "label": label}

    meta = {"mode": method, "label": label}

    # Identify strongest real signal to avoid biasing the detrend
    base = _compute_baseline_bls(t, f, durations_for_baseline, n_periods=n_periods)
    meta.update({
        "baseline_period": base["period"],
        "baseline_t0": base["t0"],
        "baseline_duration": base["duration"],
        "baseline_z": base["z_top"],
    })

    if method == "prewhiten":
        # Mask baseline transits and estimate trend from OOT data
        mask_tr = _find_transit_mask(t, base["period"], base["t0"], base["duration"], pad=pad) if np.isfinite(base["period"]) else np.zeros_like(t, bool)
        f_for_trend = f.copy()
        f_for_trend[mask_tr] = np.nan
        trend = _local_trend(t, f_for_trend, window_days=window_days)
        # Guard against zeros/NaNs
        ok = np.isfinite(trend) & (trend != 0)
        trend[~ok] = _nanmedian(trend[ok]) if np.any(ok) else 1.0
        y = f / trend
        # Normalize around 1.0
        med = _nanmedian(y)
        if np.isfinite(med) and med != 0:
            y = y / med
        meta["trend_window_days"] = float(window_days)
        return t, y, meta

    elif method == "oot-replace":
        # Replace in-transit samples with OOT-like values
        mask_tr = _find_transit_mask(t, base["period"], base["t0"], base["duration"], pad=pad) if np.isfinite(base["period"]) else np.zeros_like(t, bool)
        oot = f[~mask_tr]
        oot_med = _nanmedian(oot)
        if not np.isfinite(oot_med) or oot.size == 0:
            y = f.copy()
            return t, y, meta

        y = f.copy()
        if oot_strategy == "resample" and oot.size >= 4:
            # resample with replacement to keep realistic noise
            rng = np.random.default_rng()
            y[mask_tr] = rng.choice(oot, size=int(mask_tr.sum()), replace=True)
        else:
            y[mask_tr] = oot_med

        # Normalize
        med = _nanmedian(y)
        if np.isfinite(med) and med != 0:
            y = y / med
        meta["oot_strategy"] = oot_strategy
        return t, y, meta

    else:
        # Unknown mode -> no-op normalization
        med = _nanmedian(f)
        y = f / med if np.isfinite(med) and med != 0 else f.copy()
        return t, y, meta


# ---------------------------
# Helper: resolve tau from profile (scalar or per-method dict)
# ---------------------------
def _resolve_tau_from_profile(profile_cfg: dict, method: str, fallback: float) -> float:
    """
    Return a threshold (tau) from a profile. Accepts either a scalar or a
    per-method dict under 'tau_base'. Falls back to `fallback` if missing.
    """
    tau = profile_cfg.get("tau_base", fallback)
    if isinstance(tau, dict):
        # Prefer exact method; otherwise first value in dict; else fallback.
        if method in tau:
            try:
                return float(tau[method])
            except Exception:
                return float(fallback)
        try:
            # next(iter(...)) is safe for non-empty dicts
            return float(next(iter(tau.values())))
        except Exception:
            return float(fallback)
    try:
        return float(tau)
    except Exception:
        return float(fallback)

# ---------------------------
# Fast null-z sampler (for profile tuning)
# ---------------------------

def _stable_clean_key_for_null(t_arr, f_arr, method, n_periods, kwargs_dict):
    """Short, stable key for cleaned-series cache (null sampler)."""
    n = t_arr.size
    if n == 0:
        samp_t = b""; samp_f = b""; n_bytes = b"0"
    else:
        k = min(256, n)
        idx = np.linspace(0, n - 1, k).astype(int)
        samp_t = t_arr[idx].astype(np.float64).tobytes()
        samp_f = f_arr[idx].astype(np.float64).tobytes()
        n_bytes = str(n).encode("utf-8")

    clean_params = {
        "method": method,
        "n_periods": int(n_periods),
        "window_days": kwargs_dict.get("window_days", 0.75),
        "pad": kwargs_dict.get("pad", 1.2),
        "oot_strategy": kwargs_dict.get("oot_strategy", "median"),
        "durations_for_baseline": tuple(kwargs_dict.get("durations_for_baseline", (0.12, 0.20))),
        "cache_version": "v1",
    }
    blob = json.dumps(clean_params, sort_keys=True).encode("utf-8")
    h = hashlib.sha1()
    h.update(n_bytes); h.update(samp_t); h.update(samp_f); h.update(blob)
    return h.hexdigest()[:16], clean_params


def sample_null_z(
    t, f,
    *,
    method="oot-replace",
    durations=(0.08, 0.12, 0.20),
    n_samples=600,
    n_local=256,
    window_frac=0.02,
    min_period=None,
    max_period=None,
    seed=None,
    cache_clean_root=None,
    cache_null_root=None,
    label=None,
    show_progress=False,
    **kwargs,
):
    """
    Quickly estimate the *null* SNR (z) distribution **without injections**.

    Strategy:
      1) Clean once with `method` (using on-disk cache if provided).
      2) Draw `n_samples` random trial periods uniformly in [min_period, max_period].
      3) For each trial, run a narrow BLS scan with `n_local` periods within
         ±`window_frac` of the trial period, and record the max z across `durations`.

    Returns
    -------
    z_vals : np.ndarray
        Array of length `n_samples` (or from cache if available).
    meta : dict
        Metadata about cleaning, grids and caches used.
    """
    t = np.asarray(t, float); f = np.asarray(f, float)
    m = _finite_mask(t, f)
    t, f = t[m], f[m]
    if t.size == 0:
        return np.array([], float), {"empty": True, "label": label, "method": method}

    # Determine period range if not given
    if min_period is None:
        min_period = max(0.3, 2.0 * np.nanmedian(np.diff(np.unique(t))))
    if max_period is None:
        max_period = max(1.1 * (t.max() - t.min()) / 5.0, float(min_period) + 0.5)

    # 1) Clean once (with cache)
    cache_meta = {"cache_clean_hit": False, "cache_clean_file": None}
    meta_clean = {}

    t_clean = f_clean = None
    if cache_clean_root:
        cache_dir = os.path.expanduser(cache_clean_root)
        os.makedirs(cache_dir, exist_ok=True)
        key, _clean_params = _stable_clean_key_for_null(t, f, method, kwargs.get("n_periods", 1500), kwargs)
        cache_file = os.path.join(cache_dir, f"clean-{method}-{key}.npz")
        if os.path.exists(cache_file):
            try:
                dat = np.load(cache_file, allow_pickle=False)
                t_clean = dat["t"].astype(float)
                f_clean = dat["f"].astype(float)
                meta_json = dat["meta"].item() if dat["meta"].shape == () else dat["meta"][()]
                meta_clean = json.loads(str(meta_json))
                cache_meta.update(cache_clean_hit=True, cache_clean_file=cache_file)
            except Exception:
                t_clean = f_clean = None
    if t_clean is None or f_clean is None:
        # Use a smaller baseline n_periods for speed during tuning
        t_clean, f_clean, meta_clean = get_clean_series(
            t, f, method=method, n_periods=int(kwargs.get("n_periods", 1500)), label=label, **kwargs
        )
        if cache_clean_root:
            try:
                np.savez_compressed(
                    os.path.join(cache_dir, f"clean-{method}-{key}.npz"),
                    t=t_clean.astype(np.float64),
                    f=f_clean.astype(np.float64),
                    meta=np.array(json.dumps(meta_clean)),
                )
                cache_meta.update(cache_clean_hit=False, cache_clean_file=os.path.join(cache_dir, f"clean-{method}-{key}.npz"))
            except Exception:
                pass

    # 2) Build BLS once and reuse
    med = _nanmedian(f_clean); med = 1.0 if (not np.isfinite(med) or med == 0) else med
    y = f_clean / med
    dy0 = max(_robust_sigma(y), DY_FLOOR)
    dy = np.full_like(y, dy0, dtype=float)
    if (not np.isfinite(dy0)) or (dy0 <= 0):
        dy0 = 1.0
    dy0 = max(dy0, DY_FLOOR)
    dy = np.full_like(y, dy0, dtype=float)
    bls = BoxLeastSquares(t_clean, y, dy=dy)

    # 3) Random periods & local scans (with optional cache)
    if cache_null_root:
        null_dir = os.path.expanduser(cache_null_root)
        os.makedirs(null_dir, exist_ok=True)
        key_z = hashlib.sha1(
            json.dumps({
                "method": method,
                "durations": tuple(map(float, durations)),
                "n_samples": int(n_samples),
                "n_local": int(n_local),
                "window_frac": float(window_frac),
                "min_period": float(min_period),
                "max_period": float(max_period),
                "clean_key": cache_meta.get("cache_clean_file"),
                "cache_v": "v1",
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        z_cache_file = os.path.join(null_dir, f"nullz-{method}-{key_z}.npz")
        if os.path.exists(z_cache_file):
            try:
                dat = np.load(z_cache_file, allow_pickle=False)
                z_vals = dat["z"].astype(float)
                meta = json.loads(str(dat["meta"].item() if dat["meta"].shape == () else dat["meta"][()]))
                meta.update(cache_meta)
                return z_vals, meta
            except Exception:
                pass

    rng_local = np.random.default_rng(seed)
    trial_p = rng_local.uniform(float(min_period), float(max_period), int(n_samples))

    def _scan_one(p):
        p0 = p * (1.0 - float(window_frac))
        p1 = p * (1.0 + float(window_frac))
        periods = np.linspace(p0, p1, int(max(21, n_local)))
        z_best = -np.inf
        for dur in durations:
            if not (np.isfinite(dur) and dur > 0):
                continue
            res = bls.power(periods, float(dur), objective="likelihood")
            z_like = res.depth / np.where(res.depth_err > 0, res.depth_err, np.inf)
            z_pow  = np.sqrt(np.clip(res.power, 0, np.inf))
            z_like_safe = np.where(np.isfinite(z_like) & (np.abs(z_like) < Z_LIKE_CAP), z_like, np.nan)
            z = np.where(np.isfinite(z_like_safe), z_like_safe, z_pow)
            i = int(np.nanargmax(z))
            if np.isfinite(z[i]) and z[i] > z_best:
                z_best = float(z[i])
        return z_best

    it = tqdm(trial_p, desc=f"null {method}", leave=False) if show_progress else trial_p
    # Single-process by default (avoids mp overhead for small n_samples)
    z_vals = np.array([_scan_one(p) for p in it], dtype=float)

    # Tail trimming (winsorize) can be done by the caller; we keep raw here.
    meta = {
        "label": label,
        "method": method,
        "n_samples": int(n_samples),
        "n_local": int(n_local),
        "window_frac": float(window_frac),
        "durations": list(map(float, durations)),
        "min_period": float(min_period),
        "max_period": float(max_period),
        **meta_clean,
        **cache_meta,
    }

    if cache_null_root:
        try:
            np.savez_compressed(z_cache_file, z=z_vals.astype(np.float64), meta=np.array(json.dumps(meta)))
        except Exception:
            pass

    return z_vals, meta

# ---------------------------
# Main injection–recovery
# ---------------------------
def inject_and_recover(
    t, f,
    method="oot-replace",
    *,
    durations=(0.08, 0.12, 0.20),
    periods=(1.5, 3.0, 5.0),
    depths=(0.003, 0.005, 0.01),
    rel_window=0.015,              # relative period window for match/P,P/2,2P
    decision_metric="zscore",      # "zscore" or "depth_over_err"
    threshold=6.0,
    per_injection_search=True,
    n_periods=4000,
    min_period=None,
    max_period=None,
    profile=None,
    rng=None,
    label=None,
    # Caching / checkpoint
    cache_clean_root=None,         # e.g. "~/.cache/transitbench/clean"
    checkpoint=None,               # e.g. "runs/toi3669_balanced.jsonl"
    show_progress=False,
    # Budgeting
    compute_budget=None,           # total budget across whole grid (approx)
    cost_model: CostModel | None = None,
    # Adversarial robustness
    adversarial=False,
    adversarial_eps=0.0,
    adversarial_seed=None,
    **kwargs
):
    """
    Run a grid of injections and recoveries on (t,f) after cleaning with `method`.

    - Profile support: uses PROFILES to set decision_metric, rel_window, tau_base
      and applies beta-aware threshold scaling ("soft" or "linear").
    - Budgeting: if `compute_budget` is given, downsample the period grid to fit
      cost_model over the full injection grid (depths × durations × periods).
    - Caching: cleaned series are cached to disk if `cache_clean_root` is given.
    - Checkpointing: per-injection results are appended to `checkpoint` (JSONL)
      and restored automatically to avoid recompute.
    - Adversarial: add small worst-case perturbations before BLS if enabled.
    """
    # Accept forward-compatible 'snr_threshold' (float or {method: float})
    snr_kw = kwargs.pop("snr_threshold", None)
    # -----------------------
    # Profile overrides and β-scaling of threshold
    # -----------------------
    threshold_src = "default"
    if profile:
        p = PROFILES.get(str(profile).lower())
        if p:
            decision_metric = p.get("decision_metric", decision_metric)
            rel_window = float(p.get("rel_window", rel_window))
            threshold = _resolve_tau_from_profile(p, method, threshold)
            threshold_src = "profile_dict" if isinstance(p.get("tau_base"), dict) else "profile_float"

    # Explicit snr_threshold (if provided) overrides both default and profile
    if snr_kw is not None:
        try:
            if isinstance(snr_kw, dict):
                # pick per-method threshold; fallback to any value in the dict
                try:
                    threshold = float(snr_kw.get(method, next(iter(snr_kw.values()))))
                except StopIteration:
                    pass
                else:
                    threshold_src = "snr_threshold_dict"
            else:
                threshold = float(snr_kw)
                threshold_src = "snr_threshold_float"
        except Exception:
            # keep previously resolved threshold on any casting error
            pass

    # Basic β of the raw series
    try:
        feats = basic_features(t, f)
        beta10_raw = float(feats.get("beta10", 1.0))
        if not np.isfinite(beta10_raw) or beta10_raw <= 0:
            beta10_raw = 1.0
    except Exception:
        beta10_raw = 1.0

    beta_mode = None
    if profile:
        p = PROFILES.get(str(profile).lower())
        beta_mode = p.get("beta_scale") if p else None

    if beta_mode == "linear":
        threshold = threshold * max(1.0, beta10_raw)
    elif beta_mode == "soft":
        threshold = threshold * (0.75 + 0.25 * max(1.0, beta10_raw))

    # -----------------------
    # Normalize input & finite mask
    # -----------------------
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    m = _finite_mask(t, f)
    t, f = t[m], f[m]

    # -----------------------
    # Clean series with on-disk cache
    # -----------------------
    def _stable_clean_key(t_arr, f_arr, method, n_periods, kwargs_dict):
        """Build a short, stable hash key for the cleaned-series cache."""
        n = t_arr.size
        if n == 0:
            samp_t = b""; samp_f = b""; n_bytes = b"0"
        else:
            k = min(256, n)
            idx = np.linspace(0, n - 1, k).astype(int)
            samp_t = t_arr[idx].astype(np.float64).tobytes()
            samp_f = f_arr[idx].astype(np.float64).tobytes()
            n_bytes = str(n).encode("utf-8")

        clean_params = {
            "method": method,
            "n_periods": int(n_periods),
            "window_days": kwargs_dict.get("window_days", 0.75),
            "pad": kwargs_dict.get("pad", 1.2),
            "oot_strategy": kwargs_dict.get("oot_strategy", "median"),
            "durations_for_baseline": tuple(kwargs_dict.get("durations_for_baseline", (0.12, 0.20))),
            "cache_version": "v1",
        }
        blob = json.dumps(clean_params, sort_keys=True).encode("utf-8")
        h = hashlib.sha1()
        h.update(n_bytes); h.update(samp_t); h.update(samp_f); h.update(blob)
        return h.hexdigest()[:16], clean_params

    cache_meta = {"cache_clean_hit": False, "cache_clean_file": None}
    meta_clean = {}

    if cache_clean_root:
        cache_dir = os.path.expanduser(cache_clean_root)
        os.makedirs(cache_dir, exist_ok=True)
        key, _clean_params = _stable_clean_key(t, f, method, n_periods, kwargs)
        cache_file = os.path.join(cache_dir, f"clean-{method}-{key}.npz")
        if os.path.exists(cache_file):
            try:
                dat = np.load(cache_file, allow_pickle=False)
                t_clean = dat["t"].astype(float)
                f_clean = dat["f"].astype(float)
                meta_json = dat["meta"].item() if dat["meta"].shape == () else dat["meta"][()]
                meta_clean = json.loads(str(meta_json))
                cache_meta.update(cache_clean_hit=True, cache_clean_file=cache_file)
            except Exception:
                t_clean = f_clean = meta_clean = None
        else:
            t_clean = f_clean = meta_clean = None

        if meta_clean == {} or t_clean is None or f_clean is None:
            t_clean, f_clean, meta_clean = get_clean_series(
                t, f, method=method, n_periods=n_periods, label=label, **kwargs
            )
            try:
                np.savez_compressed(
                    cache_file,
                    t=t_clean.astype(np.float64),
                    f=f_clean.astype(np.float64),
                    meta=np.array(json.dumps(meta_clean)),
                )
                cache_meta.update(cache_clean_hit=False, cache_clean_file=cache_file)
            except Exception:
                pass
    else:
        t_clean, f_clean, meta_clean = get_clean_series(
            t, f, method=method, n_periods=n_periods, label=label, **kwargs
        )

    # -----------------------
    # Budgeting: downsample n_periods for the *injection* scans
    # -----------------------
    N_inj = int(len(durations) * len(periods) * len(depths))
    grid_downsampled = False
    n_periods_eff = int(n_periods)

    if compute_budget is not None:
        cm = cost_model or CostModel()
        # total cost ≈ c_clean (once) + c_bls * n_periods_eff * N_inj
        # solve for n_periods_eff
        denom = max(cm.c_bls * max(1, N_inj), 1e-12)
        n_max = int(max(8, math.floor((float(compute_budget) - cm.c_clean) / denom)))
        if n_max < n_periods_eff:
            n_periods_eff = n_max
            grid_downsampled = True

    # Clamp to sane bounds
    n_periods_eff = int(max(8, min(n_periods_eff, n_periods)))

    # -----------------------
    # Checkpoint preload (skip already-done combos)
    # -----------------------
    durations = tuple(float(d) for d in durations)
    periods = tuple(float(p) for p in periods)
    depths = tuple(float(d) for d in depths)
    tmid = float(np.nanmedian(t_clean)) if t_clean.size else np.nan

    def _k(dpt, dur, per):
        return f"{dpt:.6e}|{dur:.6e}|{per:.6e}"

    want_keys = {_k(dpt, dur, per) for dpt in depths for dur in durations for per in periods}
    existing_by_key = {}
    restored_from_checkpoint = False
    checkpoint_path = None

    if checkpoint:
        checkpoint_path = os.path.expanduser(checkpoint)
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(rec, dict):
                            continue
                        if rec.get("method") != method:
                            continue
                        if (label is not None) and (rec.get("label") != label):
                            continue
                        k = _k(float(rec.get("depth", np.nan)),
                               float(rec.get("duration", np.nan)),
                               float(rec.get("period", np.nan)))
                        if k in want_keys:
                            existing_by_key[k] = rec
                if existing_by_key:
                    restored_from_checkpoint = True
            except Exception:
                pass

    # -----------------------
    # Helper: adversarial perturbation
    # -----------------------
    def _adv(y, eps, seed=None):
        y = np.array(y, float, copy=True)
        if (not adversarial) or (float(eps) <= 0):
            return y
        try:
            g = np.gradient(y)
            s = np.sign(g)
            s[~np.isfinite(s)] = 0.0
            return y + float(eps) * s
        except Exception:
            # fallback: tiny random with fixed seed
            rng_local = np.random.default_rng(seed)
            return y + float(eps) * rng_local.standard_normal(y.size)

    # -----------------------
    # Iterate grid
    # -----------------------
    rng = np.random.default_rng(rng)
    new_records = []
    depth_iter = tqdm(depths, desc=f"{method} depths", leave=False) if show_progress else depths

    for depth in depth_iter:
        for dur in durations:
            for per in periods:
                k = _k(depth, dur, per)
                if k in existing_by_key:
                    continue

                # 1) Inject
                f_inj = _apply_profile(t_clean, f_clean, period=per, t0=tmid, duration=dur, depth=depth)
                f_eval = _adv(f_inj, adversarial_eps, adversarial_seed)

                # 2) Global top z anywhere (BLS scan)
                base_best = _compute_baseline_bls(
                    t_clean, f_eval, durations=(dur,), n_periods=n_periods_eff,
                    min_period=min_period, max_period=max_period
                )
                snr_top = float(base_best["z_top"]) if np.isfinite(base_best["z_top"]) else np.nan

                # 3) SNR near injected period (and harmonics if enabled)
                if per_injection_search:
                    snr_inj, match_label, match_period = _depth_snr_harmonics(
                        t_clean, f_eval, per, durations=(dur,), rel_window=rel_window
                    )
                else:
                    snr_inj, match_period = _snr_near_period(
                        t_clean, f_eval, per, durations=(dur,), rel_window=0.0
                    )
                    match_label = "P"

                # 4) Decision
                if decision_metric in ("depth_over_err", "zscore"):
                    detected = bool(np.isfinite(snr_inj) and (snr_inj >= float(threshold)))
                else:
                    detected = False

                # 5) Coverage info
                cov, ntr = _n_transits_observed(t_clean, per, dur, t_ref=tmid)

                rec = {
                    "label": label,
                    "method": method,
                    "depth": float(depth),
                    "duration": float(dur),
                    "period": float(per),
                    "snr_top": float(snr_top),
                    "snr_injected": float(snr_inj) if np.isfinite(snr_inj) else np.nan,
                    "snr_injected_z": float(snr_inj) if np.isfinite(snr_inj) else np.nan,
                    "snr_match": match_label,
                    "match_period": float(match_period) if np.isfinite(match_period) else np.nan,
                    "detected": bool(detected),
                    "phase_cov": float(cov),
                    "n_transits_obs": int(ntr),
                    "grid_n_periods": int(n_periods_eff),
                    "grid_n_durations": 1,
                    "grid_downsampled": bool(grid_downsampled),
                    "decision_metric": str(decision_metric),
                }

                new_records.append(rec)
                if checkpoint_path:
                    try:
                        with open(checkpoint_path, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                    except Exception:
                        pass

    # -----------------------
    # Combine records (checkpoint + new)
    # -----------------------
    if checkpoint_path:
        records = []
        try:
            with open(checkpoint_path, "r") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if rec.get("method") != method:
                        continue
                    if (label is not None) and (rec.get("label") != label):
                        continue
                    k = _k(float(rec.get("depth", np.nan)),
                           float(rec.get("duration", np.nan)),
                           float(rec.get("period", np.nan)))
                    if k in want_keys:
                        records.append(rec)
        except Exception:
            records = list(existing_by_key.values()) + new_records
    else:
        records = new_records

    # Sort for readability
    try:
        records.sort(key=lambda r: (r["depth"], r["duration"], r["period"]))
    except Exception:
        pass

    # -----------------------
    # Aggregate metrics
    # -----------------------
    det = np.array([1 if (isinstance(r, dict) and r.get("detected")) else 0 for r in records], dtype=int)
    snrs = np.array([r.get("snr_injected", np.nan) for r in records], dtype=float)

    cm = cost_model or CostModel()
    est_total_cost = float(cm.c_clean + cm.c_bls * n_periods_eff * max(1, len(records)))  # loose upper bound

    meta = {
        "mode": method,
        "label": label,
        "decision_metric": str(decision_metric),
        "threshold_final": float(threshold),
        "per_injection_search": bool(per_injection_search),
        "grid": f"{int(n_periods_eff)}×{1}",
        "durations": list(durations),
        "rel_window": float(rel_window),
        "beta10_raw": float(beta10_raw),
        "profile": str(profile) if profile else None,
        "threshold_source": threshold_src,
        **(meta_clean or {}),
        **cache_meta,
    }

    block = {
        "meta": meta,
        "records": records,
        "detected": int(det.sum()),
        "n_injections": int(len(records)),
        "tpr": float(det.mean()) if len(det) else 0.0,
        "snr_mean": float(np.nanmean(snrs)) if np.size(snrs) else float("nan"),
        "snr_median": float(np.nanmedian(snrs)) if np.size(snrs) else float("nan"),
        "provenance": {
            "budget_requested": float(compute_budget) if compute_budget is not None else None,
            "budget_estimated": float(est_total_cost),
            "c_clean": float(cm.c_clean),
            "c_bls": float(cm.c_bls),
            "grid": f"{int(n_periods_eff)}×{1}",
            "durations": list(durations),
            "decision_metric": str(decision_metric),
            "per_injection_search": bool(per_injection_search),
            "cache_clean_file": cache_meta.get("cache_clean_file"),
            "restored_from_checkpoint": bool(restored_from_checkpoint),
            "checkpoint": checkpoint_path,
            "skipped_due_to_checkpoint": int(len(existing_by_key)),
            "grid_downsampled": bool(grid_downsampled),
            "adversarial": bool(adversarial),
            "adversarial_eps": float(adversarial_eps),
        },
    }
    return block