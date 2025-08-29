# tools/run_completeness.py
import os, json, glob, argparse
import numpy as np

from transitbench.recover import inject_and_recover

# --- robust CSV reader (handles HLSP/TESS variants) ---
POSS_TIME = ("time","bjd","btjd","t_bjd","TIME")
POSS_FLUX = ("flux","pdcsap_flux","sap_flux","pdcsap_fluxdtr","FLUX","SAP_FLUX","PDCSAP_FLUX")

def read_csv(path):
    import numpy as np
    try:
        dat = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    except Exception:
        # fallback for odd encodings
        dat = np.genfromtxt(path, delimiter=",", names=True)

    names = tuple(n.lower() for n in dat.dtype.names or ())
    def pick(cands):
        for c in cands:
            if c.lower() in names:
                # keep original case key to index the recarray
                for real in dat.dtype.names:
                    if real.lower() == c.lower():
                        return real
        return None

    tkey = pick(POSS_TIME)
    fkey = pick(POSS_FLUX)
    if not tkey or not fkey:
        raise ValueError(f"Could not find time/flux columns in {os.path.basename(path)}")
    t = np.asarray(dat[tkey], float)
    f = np.asarray(dat[fkey], float)
    m = np.isfinite(t) & np.isfinite(f)
    return t[m], f[m]

def summarize(blocks):
    """TPR by depth & duration, plus global TPR for each method."""
    out = {}
    for b in blocks:
        meta = b.get("meta", {})
        method = meta.get("mode")
        if method not in out: out[method] = {"records": [], "tpr_global": 0.0}
        out[method]["records"].extend(b.get("records", []))

    for method, d in out.items():
        recs = d["records"]
        if not recs:
            d["tpr_global"] = 0.0
            d["by_depth"] = {}
            d["by_duration"] = {}
            continue
        det = np.array([1 if r.get("detected") else 0 for r in recs], int)
        d["tpr_global"] = float(det.mean())

        # group by depth
        by_depth = {}
        for depth in sorted(set(float(r["depth"]) for r in recs)):
            mask = np.array([np.isclose(float(r["depth"]), depth) for r in recs])
            dd = np.array([1 if r.get("detected") else 0 for r in np.array(recs, dtype=object)[mask]], int)
            by_depth[f"{depth:.6f}"] = float(dd.mean()) if dd.size else 0.0
        d["by_depth"] = by_depth

        # group by duration
        by_dur = {}
        for dur in sorted(set(float(r["duration"]) for r in recs)):
            mask = np.array([np.isclose(float(r["duration"]), dur) for r in recs])
            dd = np.array([1 if r.get("detected") else 0 for r in np.array(recs, dtype=object)[mask]], int)
            by_dur[f"{dur:.6f}"] = float(dd.mean()) if dd.size else 0.0
        d["by_duration"] = by_dur

        # light weight size/budget info
        d["n_injections"] = int(sum(b.get("n_injections", 0) for b in blocks if b.get("meta",{}).get("mode")==method))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "/Users/eshaantripathi/Documents/transitbench/data/raw/false_positives",
        "/Users/eshaantripathi/Documents/transitbench/data/raw/transit_candidates",
    ], help="Directories to search for CSVs (recursive)")
    ap.add_argument("--glob", default="**/*.csv", help="Glob pattern under each root")
    ap.add_argument("--max-files", type=int, default=120, help="Cap number of LCs to sample")
    ap.add_argument("--profile", default="balanced", choices=["sensitive","balanced","strict"])
    ap.add_argument("--methods", nargs="+", default=["oot-replace","prewhiten"])
    ap.add_argument("--depths", nargs="+", type=float, default=[0.003, 0.005, 0.010])
    ap.add_argument("--durations", nargs="+", type=float, default=[0.08, 0.12, 0.20])
    ap.add_argument("--periods", nargs="+", type=float, default=[1.5, 3.0, 5.0])
    ap.add_argument("--budget", type=float, default=12.0, help="Compute budget knob (smaller=faster)")
    ap.add_argument("--nperiods", type=int, default=4000, help="Baseline n_periods for BLS")
    ap.add_argument("--cache-clean", default="~/.cache/transitbench/clean", help="Directory for cleaned-series cache")
    ap.add_argument("--runs-dir", default="/Users/eshaantripathi/Documents/transitbench/runs", help="Where to store outputs")
    args = ap.parse_args()

    # gather files
    files = []
    for r in args.roots:
        files.extend(glob.glob(os.path.join(r, args.glob), recursive=True))
    files = sorted(set(files))[: args.max_files]
    if not files:
        raise SystemExit("No CSVs found—check --roots/--glob.")

    os.makedirs(args.runs_dir, exist_ok=True)

    all_blocks = []
    for mth in args.methods:
        for i, pth in enumerate(files, 1):
            try:
                t, f = read_csv(pth)
            except Exception as e:
                print(f"[{mth}] {os.path.basename(pth)} -> read error: {e}; skipped")
                continue

            # per-file checkpoint lets you resume if interrupted
            base = os.path.splitext(os.path.basename(pth))[0]
            ckpt = os.path.join(args.runs_dir, f"comp_{args.profile}_{mth}_{base}.jsonl")

            block = inject_and_recover(
                t, f,
                method=mth,
                profile=args.profile,
                depths=tuple(args.depths),
                durations=tuple(args.durations),
                periods=tuple(args.periods),
                n_periods=int(args.nperiods),
                compute_budget=float(args.budget),
                cache_clean_root=args.cache_clean,
                checkpoint=ckpt,
                show_progress=False,
                label=base,
            )
            all_blocks.append(block)
            if i % 10 == 0:
                print(f"[{mth}] processed {i}/{len(files)}")

    # summarize & save
    summary = summarize(all_blocks)
    out_json = {
        "profile": args.profile,
        "methods": args.methods,
        "depths": args.depths,
        "durations": args.durations,
        "periods": args.periods,
        "budget": args.budget,
        "results": summary,
    }
    out_path = os.path.join(args.runs_dir, f"completeness_{args.profile}.json")
    with open(out_path, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"Wrote {out_path}")
    for mth, stats in summary.items():
        print(f"\n== {mth} ({args.profile}) ==")
        print(f"Global TPR: {stats['tpr_global']:.3f} over {stats['n_injections']} injections")
        print(f"TPR by depth: {stats['by_depth']}")
        print(f"TPR by duration: {stats['by_duration']}")
        
if __name__ == "__main__":
    main()