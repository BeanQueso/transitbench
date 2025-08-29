# api.py
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
import numpy as np
import io
import sys
from . import recover

# Public API surface (some functions are convenience wrappers that other modules call)
__all__ = [
    "to_text_report",
    "print_report",
    "save_report",
    "score_lightcurve",
    "score_lightcurve_metrics",
    "run_adversarial",
]

# ----------------- canonicalization helpers (keep in sync with core.py) -----------------

def _coerce_method_block(block: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """
    Return a canonical method block:

        { "records": [...], "provenance": {...}, "meta": {...}, "metrics": {...?} }

    Accepts legacy shapes produced by earlier versions.
    """
    if not isinstance(block, dict):
        return {"records": [], "provenance": {}, "meta": {}, "metrics": {}}

    recs = block.get("records", None)

    # Legacy A: {"records": {"methods": {method_name: {...}}}}
    if isinstance(recs, dict) and "methods" in recs and method_name in recs["methods"]:
        inner = recs["methods"][method_name]
        out = {
            "records": inner.get("records", []),
            "provenance": inner.get("provenance", {}),
            "meta": inner.get("meta", {}),
            "metrics": inner.get("metrics", {}),
        }
        return out

    # Legacy B: {"records": {"records":[...], "provenance":{...}}}
    if isinstance(recs, dict) and "records" in recs:
        inner = recs
        out = {
            "records": inner.get("records", []),
            "provenance": inner.get("provenance", {}),
            "meta": inner.get("meta", {}),
            "metrics": block.get("metrics", {}),
        }
        # carry over top-level if present
        for k in ("provenance", "meta"):
            if k in block and not out.get(k):
                out[k] = block[k]
        return out

    # Canonical
    if isinstance(recs, list):
        return {
            "records": recs,
            "provenance": block.get("provenance", {}),
            "meta": block.get("meta", {}),
            "metrics": block.get("metrics", {}),
        }

    # Fallback
    return {"records": [], "provenance": block.get("provenance", {}), "meta": block.get("meta", {}), "metrics": block.get("metrics", {})}


def _metrics_from_records(recs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    recs = list(recs) if not isinstance(recs, list) else recs
    n = len(recs)
    if n == 0:
        return dict(n_injections=0, detected=0, tpr=0.0, snr_mean=np.nan, snr_median=np.nan)

    det = np.array([1 if r.get("detected") else 0 for r in recs], dtype=int)
    snr = np.array([r.get("snr_injected", np.nan) for r in recs], dtype=float)
    detected = int(det.sum())
    tpr = float(detected) / n if n > 0 else 0.0
    return dict(
        n_injections=n,
        detected=detected,
        tpr=tpr,
        snr_mean=float(np.nanmean(snr)) if np.isfinite(snr).any() else np.nan,
        snr_median=float(np.nanmedian(snr)) if np.isfinite(snr).any() else np.nan,
    )


# ----------------- text report builders -----------------

def _fmt_float(x: Optional[float], nd: int = 2, none: str = "None") -> str:
    if x is None:
        return none
    try:
        if np.isnan(x):
            return "nan"
    except Exception:
        pass
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def to_text_report(res: Dict[str, Any], header: Optional[str] = None) -> str:
    """
    Build a human-readable text report from the canonical (or legacy) result object.
    """
    buf = io.StringIO()
    hdr = header or "TransitBench — injection–recovery summary"
    print(hdr, file=buf)
    print("-" * len(hdr), file=buf)

    label = res.get("label", "lightcurve")
    basic = res.get("basic", {})
    print(f"Label: {label}", file=buf)
    print(
        f"Basic: n_pts={basic.get('n_pts','?')}, rms_oot={_fmt_float(basic.get('rms_oot'))}, beta10={_fmt_float(basic.get('beta10'))}",
        file=buf,
    )
    print("", file=buf)

    methods = res.get("methods", {})
    for m in methods:
        blk = _coerce_method_block(methods[m], m)
        metrics = blk.get("metrics") or _metrics_from_records(blk.get("records", []))
        n_inj = metrics.get("n_injections", 0)
        det = metrics.get("detected", 0)
        tpr = metrics.get("tpr", 0.0)
        snr_mean = metrics.get("snr_mean", np.nan)
        snr_median = metrics.get("snr_median", np.nan)
        meta = blk.get("meta", {})
        prov = blk.get("provenance", {})

        print(f"Method: {m}", file=buf)
        print(f"  Injections: {n_inj}, Detected: {det}, TPR: {_fmt_float(tpr, 3)}", file=buf)
        print(f"  SNR_injected: mean={_fmt_float(snr_mean)}  median={_fmt_float(snr_median)}", file=buf)

        # A small, stable subset of provenance that is useful
        prov_bits = []
        if "grid" in prov:
            prov_bits.append(f"grid={prov['grid']}")
        if "decision_metric" in prov:
            prov_bits.append(f"decision_metric={prov['decision_metric']}")
        if "per_injection_search" in prov:
            prov_bits.append(f"per_injection_search={prov['per_injection_search']}")
        if "budget" in prov and prov["budget"]:
            prov_bits.append(f"budget≈{prov['budget']}")
        if prov_bits:
            print(f"  Provenance: " + ", ".join(prov_bits), file=buf)

        # Short top-5 table
        recs = list(blk.get("records", []))
        if recs:
            # sort by snr_injected desc
            recs_sorted = sorted(recs, key=lambda r: (r.get("snr_injected") or -np.inf), reverse=True)[:5]
            print("  Top 5 records (by injected SNR):", file=buf)
            for r in recs_sorted:
                d = r.get("depth")
                du = r.get("duration")
                p = r.get("period")
                snr_i = r.get("snr_injected")
                snr_t = r.get("snr_top")
                match = r.get("snr_match", "?")
                mp = r.get("match_period", "?")
                det_f = r.get("detected", False)
                print(
                    f"    depth={_fmt_float(d,3)}, dur={_fmt_float(du,2)}, P_inj={_fmt_float(p,2)}"
                    f" -> snr_inj={_fmt_float(snr_i,2)} (match {match} @ {_fmt_float(mp,4)}),"
                    f" snr_top={_fmt_float(snr_t,2)}, detected={'True' if det_f else 'False'}",
                    file=buf
                )
        print("", file=buf)

    # Optional comparison if present
    cmpd = res.get("comparison", {})
    if cmpd:
        print("Comparison (baseline = {})".format(cmpd.get("baseline", "?")), file=buf)
        for other, stats in (cmpd.get("delta", {}) or {}).items():
            dtpr = stats.get("delta_tpr")
            dmean = stats.get("delta_snr_mean")
            dmed = stats.get("delta_snr_median")
            print(f"  vs {other}: ΔTPR={_fmt_float(dtpr,3)}  ΔSNR_mean={_fmt_float(dmean,2)}  ΔSNR_median={_fmt_float(dmed,2)}", file=buf)
        print("", file=buf)

    return buf.getvalue()


def print_report(res: Dict[str, Any], header: Optional[str] = None, file=None) -> None:
    """
    Print the text report to stdout (or a given file-like).
    Robust to legacy/canonical result shapes.
    """
    s = to_text_report(res, header=header)
    (file or sys.stdout).write(s)


def save_report(res: Dict[str, Any], path: str, header: Optional[str] = None) -> None:
    """
    Save the text report to a file (only when explicitly called).
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_text_report(res, header=header))


# ----------------- convenience scoring wrappers (used by batch) -----------------

def score_lightcurve(t: np.ndarray, f: np.ndarray, *, method: str = "oot-replace", **kwargs) -> Dict[str, Any]:
    """
    Thin wrapper to run a single method and return its canonical method block directly.
    Useful for batch code that only needs one method’s records/metrics.
    """
    blk = recover.inject_and_recover(t, f, method=method, **kwargs)
    blk = _coerce_method_block(blk, method)
    # make sure metrics exist
    if not blk.get("metrics"):
        blk["metrics"] = _metrics_from_records(blk.get("records", []))
    return blk


def score_lightcurve_metrics(t: np.ndarray, f: np.ndarray, *, method: str = "oot-replace", **kwargs) -> Dict[str, Any]:
    """
    Return only the metrics dict for a given method.
    """
    blk = score_lightcurve(t, f, method=method, **kwargs)
    return blk.get("metrics", {})


# ----------------- adversarial demo runner (API stayed here to avoid import loops) -----------------

def run_adversarial(
    t: np.ndarray,
    f: np.ndarray,
    *,
    method: str = "oot-replace",
    eps: float = 0.0,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run a one-off adversarial perturbation + evaluate.
    """
    blk = recover.inject_and_recover(
        t, f, method=method, adversarial=True, adversarial_eps=eps, adversarial_seed=seed, **kwargs
    )
    blk = _coerce_method_block(blk, method)
    if not blk.get("metrics"):
        blk["metrics"] = _metrics_from_records(blk.get("records", []))
    return blk

def estimate_cost(n_pts:int, periods:int, durations:int, n_injections:int, n_methods:int,
                  c_clean:float=0.03, c_bls:float=0.002,
                  per_injection_search:bool=True) -> int:
    """
    Crude op-count estimate to size jobs before running.
    """
    C0 = c_clean
    Cb = periods * durations * c_bls
    per_inj = (C0 + Cb) if per_injection_search else (C0 + Cb / max(n_injections,1))
    return int(n_methods * n_injections * per_inj)

def plan(n_pts:int, periods:int, durations:int, n_injections:int,
         methods=("oot-replace","prewhiten"),
         per_injection_search:bool=True,
         c_clean:float=0.03, c_bls:float=0.002) -> str:
    est = estimate_cost(n_pts, periods, durations, n_injections, len(methods),
                        c_clean=c_clean, c_bls=c_bls,
                        per_injection_search=per_injection_search)
    return (f"Planned cost ≈ {est:,} ops | grid={periods}×{durations} | "
            f"injections={n_injections} | methods={len(methods)} | "
            f"per_injection_search={per_injection_search}")