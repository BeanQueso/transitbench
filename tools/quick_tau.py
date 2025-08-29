# /transitbench/tools/quick_tau.py
import json, os, glob, argparse, numpy as np
import transitbench as tb
from transitbench import recover

# --- defaults (no user-specific paths) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.environ.get("TB_DATA_ROOT", os.path.join(ROOT, "data", "raw"))
DEFAULT_GLOB = os.environ.get("TB_GLOB", os.path.join(DEFAULT_DATA_ROOT, "**", "hlsp_tess-data-alerts_*_lc.csv"))
DEFAULT_OUTFILE = os.environ.get("TB_OUTFILE", os.path.join(ROOT, "profile_overrides.json"))

METHODS = ("oot-replace", "prewhiten")
N_SAMPLES = 256
SEED = 42
Z_CAP = 200.0          # clip null z-scores to [-Z_CAP, Z_CAP] before percentiles (helps tame outliers)
OUTLIER_P95_MAX = 50.0    # drop files whose per-file p95 exceeds this (after z-cap)
# -----------------------------------------

def per_file_null(method, path, n=N_SAMPLES, seed=SEED):
    """
    Compute per-file null z percentiles for a given method.

    NOTE: recover.sample_null_z expects (t, f) arrays, not a file path.
    We load the file via tb.load() and cap z at Z_CAP before computing percentiles.
    """
    try:
        lc = tb.load(path)  # auto-detects CSV/FITS/TBL time/flux columns
        z = recover.sample_null_z(lc.t, lc.f, method=method, n_samples=n, seed=seed)

        # Allow for implementations that return (values, meta)
        if isinstance(z, tuple):
            z = z[0]

        # Coerce to 1-D float array, keep finite values only
        z = np.asarray(z, dtype=float).ravel()
        z = z[np.isfinite(z)]

        # Optional cap to limit pathological outliers
        if Z_CAP is not None:
            z = np.clip(z, -Z_CAP, Z_CAP)

        if z.size == 0:
            return None

        return np.nanpercentile(z, [50, 95])
    except Exception as e:
        print(f"{method}: {os.path.basename(path)} -> error: {e}; skipped")
        return None

def suggest_from_p95(p95_vals):
    # robust aggregate across files
    p95_vals = np.asarray(p95_vals, float)
    p95_vals = p95_vals[np.isfinite(p95_vals)]
    if p95_vals.size == 0:
        return dict(sensitive=6.0, balanced=8.0, strict=10.0)  # safe fallback
    # Trim extremes (10% each side) and take percentiles
    q_lo, q_hi = np.nanpercentile(p95_vals, [10, 90])
    core = p95_vals[(p95_vals >= q_lo) & (p95_vals <= q_hi)]
    if core.size < 5: core = p95_vals
    sens = float(np.nanpercentile(core, 60))   # a bit above median-of-p95
    bal  = float(np.nanpercentile(core, 75))
    strict = float(np.nanpercentile(core, 85))
    # keep within reasonable z ranges
    sens = max(4.0, min(sens, 14.0))
    bal  = max(5.0, min(bal, 16.0))
    strict = max(6.0, min(strict, 20.0))
    return dict(sensitive=sens, balanced=bal, strict=strict)

def parse_args():
    p = argparse.ArgumentParser(description="Suggest tau_base per method from null z-score percentiles.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help="Root directory containing raw data; can also set TB_DATA_ROOT env var.")
    p.add_argument("--glob", default=DEFAULT_GLOB,
                   help="Glob pattern for light curve files. If provided, overrides --data-root. Can also set TB_GLOB.")
    p.add_argument("--outfile", default=DEFAULT_OUTFILE,
                   help="Where to write profile_overrides.json; can also set TB_OUTFILE env var.")
    p.add_argument("--methods", nargs="+", default=list(METHODS),
                   help="Methods to evaluate (e.g., oot-replace prewhiten).")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Null samples per file.")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    p.add_argument("--z-cap", type=float, default=Z_CAP, help="Clip null z to ±Z_CAP before percentiles.")
    p.add_argument("--outlier-p95-max", type=float, default=OUTLIER_P95_MAX,
                   help="Drop files with per-file p95 above this (after z-cap).")
    return p.parse_args()

def main():
    args = parse_args()

    # allow CLI overrides of module defaults
    global METHODS, N_SAMPLES, SEED, Z_CAP, OUTLIER_P95_MAX
    METHODS = tuple(args.methods) if args.methods else METHODS
    N_SAMPLES = args.n_samples
    SEED = args.seed
    Z_CAP = args.z_cap
    OUTLIER_P95_MAX = args.outlier_p95_max

    # resolve glob pattern
    pattern = args.glob or os.path.join(args.data_root, "**", "hlsp_tess-data-alerts_*_lc.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(files)} files (pattern={pattern})")
    payload = {"suggest_tau_base": {}}

    for method in METHODS:
        per_file = []
        for p in files:
            pp = per_file_null(method, p)
            if pp is None:
                print(f"{method}: {os.path.basename(p)} -> skipped (no null)")
                continue
            p50, p95 = pp
            print(f"{method}: {os.path.basename(p)} -> p50={p50:.2f}, p95={p95:.2f}")
            if p95 <= OUTLIER_P95_MAX:
                per_file.append(p95)
            else:
                print(f"  (excluded: p95>{OUTLIER_P95_MAX})")
        tau = suggest_from_p95(per_file)
        payload["suggest_tau_base"][method] = tau
        print(f"{method}: tau suggestions: {tau}")

    # Write overrides
    with open(args.outfile, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.outfile}")

    # Apply in-memory immediately
    for method, tau in payload["suggest_tau_base"].items():
        for prof_name, val in tau.items():
            tbp = recover.PROFILES.get(prof_name, {}).get("tau_base")
            if not isinstance(tbp, dict):
                recover.PROFILES[prof_name]["tau_base"] = {}
            recover.PROFILES[prof_name]["tau_base"][method] = float(val)
    print("Applied profile overrides to recover.PROFILES.")

if __name__ == "__main__":
    main()