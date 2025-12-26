# src/transitbench/core.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, Optional, Union, List, Tuple, cast
import pandas as pd
from . import recover
from .io import load_timeseries  # uses the robust _find_col-based loader under the hood
from pathlib import Path  # <-- ensure this import exists at top of core.py

__all__ = [
    "LightCurve",
    "load",
]

# ---------- small helpers ----------

def _metrics_from_records(recs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    recs = list(recs) if not isinstance(recs, list) else recs
    n = len(recs)
    if n == 0:
        return dict(n_injections=0, detected=0, tpr=0.0, snr_mean=np.nan, snr_median=np.nan)

    det = np.array([1 if (isinstance(r, dict) and r.get("detected")) else 0 for r in recs], dtype=int)
    snr = np.array([r.get("snr_injected", np.nan) if isinstance(r, dict) else np.nan for r in recs], dtype=float)
    detected = int(det.sum())
    tpr = float(detected) / n if n > 0 else 0.0
    return dict(
        n_injections=n,
        detected=detected,
        tpr=tpr,
        snr_mean=float(np.nanmean(snr)) if np.isfinite(snr).any() else np.nan,
        snr_median=float(np.nanmedian(snr)) if np.isfinite(snr).any() else np.nan,
    )


def _canonicalize_method_block(block: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """
    Return a canonical method block:

        { "records": [...], "provenance": {...}, "meta": {...} }

    Accepts legacy shapes produced by older recover.inject_and_recover versions:
      - {"records": {"methods": {method_name: {"records":[...], "provenance":{...}}}, "basic":{...}}}
      - {"records": {"records":[...], "provenance":{...}}}
    """
    if not isinstance(block, dict):
        return {"records": [], "provenance": {}, "meta": {}}

    recs = block.get("records", None)

    # Legacy A: wrapper with {"methods": {method_name: {...}}}
    if isinstance(recs, dict) and "methods" in recs and method_name in recs["methods"]:
        inner = recs["methods"][method_name]
        return {
            "records": inner.get("records", []),
            "provenance": inner.get("provenance", {}),
            "meta": inner.get("meta", {}),
        }

    # Legacy B: wrapper with {"records":[...], "provenance":{...}}
    if isinstance(recs, dict) and "records" in recs:
        inner = recs
        out = {
            "records": inner.get("records", []),
            "provenance": inner.get("provenance", {}),
            "meta": inner.get("meta", {}),
        }
        # carry over top-level if present
        for k in ("provenance", "meta"):
            if k in block and not out.get(k):
                out[k] = block[k]
        return out

    # Canonical already
    if isinstance(recs, list):
        return {
            "records": recs,
            "provenance": block.get("provenance", {}),
            "meta": block.get("meta", {}),
        }

    # Fallback
    return {"records": [], "provenance": block.get("provenance", {}), "meta": block.get("meta", {})}


# ---------- main class ----------

@dataclass
class LightCurve:
    t: np.ndarray
    f: np.ndarray
    label: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)  # <-- NEW

    def __post_init__(self):
        # Ensure arrays are float and same length
        self.t = np.asarray(self.t, dtype=float)
        self.f = np.asarray(self.f, dtype=float)
        if self.t.shape != self.f.shape:
            raise ValueError(f"t and f must have the same shape (got {self.t.shape} vs {self.f.shape})")
        # Ensure meta is a dict
        if not isinstance(self.meta, dict):
            self.meta = {}

    @property
    def n_pts(self) -> int:
        return int(self.t.size)

    def copy(self) -> "LightCurve":
        # Helpful when you want a safe mutable copy
        return LightCurve(self.t.copy(), self.f.copy(), label=self.label, meta=self.meta.copy())

    def with_meta(self, **updates) -> "LightCurve":
        # Convenience to add/override metadata
        m = self.meta.copy()
        m.update(updates)
        return LightCurve(self.t, self.f, label=self.label, meta=m)

    def benchmark(
        self,
        method: Union[str, Iterable[str]] = ("oot-replace",),
        *,
        compare_with: Optional[Iterable[str]] = None,
        profile: str = "balanced",
        # search / decision controls propagated to recover
        per_injection_search: bool = True,
        decision_metric: str = "zscore",
        snr_threshold: Optional[float] = None,
        rel_window: Optional[float] = None,     # fractional window around injected P
        durations: Optional[Iterable[float]] = (0.08, 0.12, 0.20),
        depths: Optional[Iterable[float]] = (0.003, 0.005, 0.010),
        periods: Optional[Iterable[float]] = (1.5, 3.0, 5.0),
        compute_budget: Optional[int] = None,
        cost_model: Optional["recover.CostModel"] = None,  # forward-ref OK
        # detrending specifics
        oot_strategy: str = "median",
        prewhiten_nharm: int = 3,
        # adversarial / robustness knobs (forwarded; default off)
        adversarial: bool = False,
        adversarial_eps: float = 0.0,
        adversarial_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run inject-and-recover for one or multiple methods and return a canonical result dict.
        """
        from . import recover  # local import to avoid circulars

        # Allow method as str or list
        methods: List[str] = []
        if isinstance(method, str):
            methods.append(method)
        else:
            methods.extend(list(method))
        if compare_with:
            for m in compare_with:
                if m not in methods:
                    methods.append(m)

        # Basic stats for header
        basic = recover.basic_features(self.t, self.f)

        # Map profile into defaults if not overridden
        prof = (profile or "balanced").lower()
        if prof == "sensitive":
            decision_metric = decision_metric or "zscore"
            snr_threshold = 5.0 if snr_threshold is None else snr_threshold
            rel_window = 0.02 if rel_window is None else rel_window
        elif prof == "strict":
            decision_metric = decision_metric or "zscore"
            snr_threshold = 8.0 if snr_threshold is None else snr_threshold
            rel_window = 0.01 if rel_window is None else rel_window
        else:  # balanced
            decision_metric = decision_metric or "zscore"
            snr_threshold = 6.0 if snr_threshold is None else snr_threshold
            rel_window = 0.015 if rel_window is None else rel_window

        # Merge per-method threshold overrides from recover.PROFILES (if present)
        prof_cfg = recover.PROFILES.get(prof, {}) if hasattr(recover, "PROFILES") else {}
        tau_base = prof_cfg.get("tau_base", {})
        snr_map: Dict[str, float] = {}
        if isinstance(tau_base, dict):
            for _m in methods:
                if _m in tau_base:
                    try:
                        snr_map[_m] = float(tau_base[_m])
                    except Exception:
                        pass

        out: Dict[str, Any] = {
            "label": self.label or "lightcurve",
            "basic": basic,
            "methods": {}
        }

        for m in methods:
            blk = recover.inject_and_recover(
                self.t,
                self.f,
                method=m,
                per_injection_search=bool(per_injection_search),
                decision_metric=str(decision_metric),
                snr_threshold=(snr_map if snr_map else snr_threshold),
                rel_window=float(rel_window) if rel_window is not None else 0.0,
                durations=list(durations) if durations is not None else None,
                periods=list(periods) if periods is not None else None,
                depths=list(depths) if depths is not None else None,
                compute_budget=compute_budget,
                cost_model=cost_model,
                oot_strategy=oot_strategy,
                prewhiten_nharm=prewhiten_nharm,
                adversarial=bool(adversarial),
                adversarial_eps=float(adversarial_eps),
                adversarial_seed=adversarial_seed,
                label=self.label or "lightcurve",
            )
            blk = _canonicalize_method_block(blk, m)
            metrics = _metrics_from_records(blk.get("records", []))
            out["methods"][m] = {
                "records": blk.get("records", []),
                "provenance": blk.get("provenance", {}),
                "meta": blk.get("meta", {"mode": m}),
                "metrics": metrics,
            }

        return out



# ---------- convenience loader ----------



def _coerce_timeseries_result(res: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Accepts a variety of shapes from io.load_timeseries and coerces to (t, f, meta).
    Supported:
      - (t, f) or (t, f, meta)
      - dict with 'time'/'flux' (case-insensitive) and optional 'meta'
      - pandas.DataFrame with recognizable time/flux columns
      - ndarray with shape (N,>=2)
      - numpy structured/recarray with named fields
    """
    meta: Dict[str, Any] = {}

    # dict-like
    if isinstance(res, dict):
        keys_lower = {str(k).lower(): k for k in res.keys()}

        def _get(resdict, *names):
            for nm in names:
                k = keys_lower.get(nm)
                if k is not None:
                    return resdict[k]
            return None

        t = _get(res, "time", "t", "bjd_tdb", "btjd", "jd", "mjd")
        f = _get(res, "flux", "f", "rel_flux", "sap_flux", "pdcsap_flux")
        meta = res.get("meta", {}) if isinstance(res.get("meta", {}), dict) else {}

        if t is None or f is None:
            # Maybe the dict holds a DataFrame/array under 'data'
            data = res.get("data", None)
            if data is not None:
                return _coerce_timeseries_result(data)

        if t is not None and f is not None:
            return np.asarray(t, float), np.asarray(f, float), meta

    # tuple/list
    if isinstance(res, (tuple, list)):
        if len(res) == 2:
            t, f = res
            meta = {}
            return np.asarray(t, float), np.asarray(f, float), meta
        elif len(res) == 3:
            t, f, meta = res
            if not isinstance(meta, dict):
                meta = {}
            return np.asarray(t, float), np.asarray(f, float), meta
        # else fall through to error

    # pandas.DataFrame
    if pd is not None and isinstance(res, pd.DataFrame):
        cols = {c.lower(): c for c in res.columns}
        tcol = next((cols[c] for c in ("time", "t", "bjd_tdb", "btjd", "jd", "mjd") if c in cols), None)
        fcol = next((cols[c] for c in ("flux", "f", "rel_flux", "sap_flux", "pdcsap_flux") if c in cols), None)
        if tcol and fcol:
            return res[tcol].to_numpy(dtype=float), res[fcol].to_numpy(dtype=float), {}

    # numpy array
    if isinstance(res, np.ndarray):
        if res.ndim == 2 and res.shape[1] >= 2:
            return res[:, 0].astype(float), res[:, 1].astype(float), {}
        if res.dtype.names:  # structured or recarray
            names = {n.lower(): n for n in res.dtype.names}
            tname = next((names[n] for n in ("time", "t", "bjd_tdb", "btjd", "jd", "mjd") if n in names), None)
            fname = next((names[n] for n in ("flux", "f", "rel_flux", "sap_flux", "pdcsap_flux") if n in names), None)
            if tname and fname:
                return res[tname].astype(float), res[fname].astype(float), {}

    raise ValueError("io.load_timeseries returned an unexpected shape/type.")


def _load_fits_lightcurve(
    path: Union[str, Path],
    time_hint: Optional[str] = None,
    flux_hint: Optional[str] = None,
    quality_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Minimal FITS reader (no auto-downloads). Expects a Kepler/TESS-like light curve.
    """
    path = Path(path)
    try:
        from astropy.io import fits as fits_mod  # local import to keep astropy optional
    except Exception as e:
        raise ImportError("Reading FITS requires astropy. Please `pip install astropy`.") from e

    with cast(Any, fits_mod).open(str(path), memmap=False) as hdul:
        # Heuristics: look for the first table HDU with TIME + FLUX-like columns.
        table_hdu = None
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "columns"):
                table_hdu = hdu
                break
        if table_hdu is None:
            raise ValueError(f"No table HDU found in FITS file: {path}")

        data = table_hdu.data
        cols = {c.name.lower(): c.name for c in data.columns}

        # Choose time & flux columns (hints override heuristics)
        time_candidates = [time_hint, "time", "bjd_tdb", "btjd", "jd", "mjd"]
        flux_candidates = [flux_hint, "flux", "pdcsap_flux", "sap_flux", "rel_flux", "f"]

        def pick(cands):
            for nm in cands:
                if nm and nm.lower() in cols:
                    return cols[nm.lower()]
            return None

        tcol = pick(time_candidates)
        fcol = pick(flux_candidates)
        if not tcol or not fcol:
            raise ValueError(f"Could not find time/flux columns in FITS. Found: {list(cols.keys())}")

        t = np.array(data[tcol], dtype=float)
        f = np.array(data[fcol], dtype=float)

        # Optional quality masking
        if quality_col:
            qname = cols.get(quality_col.lower())
            if qname:
                q = np.array(data[qname])
                # Keep rows with q==0
                ok = (np.isfinite(t) & np.isfinite(f) & (q == 0))
                t, f = t[ok], f[ok]

    meta = {"source": "fits", "path": str(path)}
    return t, f, meta


def load(
    path: Union[str, Path],
    time: Optional[str] = None,
    flux: Optional[str] = None,
    quality_col: Optional[str] = None,
    label: Optional[str] = None,
    save_csv_from_fits: bool = False,
    csv_outdir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> "LightCurve":
    """
    Public loader that wraps io.load_timeseries with very forgiving output coercion.
    - Accepts CSV/TSV and FITS paths.
    - For FITS: optionally write a CSV for downstream use.

    Returns a LightCurve instance.
    """
    path = Path(path)

    if path.suffix.lower() in (".fits", ".fit", ".fits.gz"):
        # FITS flow
        t, f, meta = _load_fits_lightcurve(path, time_hint=time, flux_hint=flux, quality_col=quality_col)
        # Optional CSV export
        if save_csv_from_fits:
            if pd is None:
                raise ImportError("save_csv_from_fits=True requires pandas. Please `pip install pandas`.")
            outdir = Path(csv_outdir) if csv_outdir else path.parent
            outdir.mkdir(parents=True, exist_ok=True)
            outcsv = outdir / (path.stem.replace(".fits", "").replace(".fit", "") + ".csv")
            pd.DataFrame({"time": t, "flux": f}).to_csv(outcsv, index=False)
            meta["csv_path"] = str(outcsv)
            meta["csv_saved"] = True
    else:
        # Text/CSV flow via io.load_timeseries (support both new/legacy signatures)
        try:
            res = load_timeseries(
                str(path),
                time_col=time,
                flux_col=flux,
                quality_col=quality_col,
                **kwargs,
            )
        except TypeError:
            # Legacy io.load_timeseries without quality_col
            res = load_timeseries(str(path), time_col=time, flux_col=flux, **kwargs)

        t, f, meta = _coerce_timeseries_result(res)

    # Final sanitation: drop NaNs/inf and sort by time
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if t.size == 0:
        raise ValueError("Loaded lightcurve is empty after filtering NaNs/Infs.")

    order = np.argsort(t)
    t, f = t[order], f[order]

    # Create LightCurve
    lbl = label or (meta.get("label") if isinstance(meta, dict) else None) or path.stem
    return LightCurve(t, f, label=lbl, meta=meta if isinstance(meta, dict) else {})
