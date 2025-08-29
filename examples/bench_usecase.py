#!/usr/bin/env python3
"""
examples/bench_usecase.py

End-to-end example:
- load a light curve (CSV/FITS/TBL; auto-detect columns when possible)
- run injection–recovery with both oot-replace and prewhiten
- optional budget + adversarial toggles
- write outputs into a timestamped runs/ folder via utils.Run

Usage:
  $ python examples/bench_usecase.py                      # uses examples/sample_data/* by default
  $ python examples/bench_usecase.py --data-root <dir>    # pick first file under <dir>
  $ python examples/bench_usecase.py --path <file>        # run a specific file
"""

from __future__ import annotations
import argparse, os, sys
from typing import Iterable, List, Optional, Tuple

import transitbench as tb
from transitbench.utils import Run, save_records_csv, write_md_summary

DEFAULT_METHODS = ("oot-replace", "prewhiten")

def guess_demo_root() -> str:
    """
    Return the default demo data root shipped with the repo.
    examples/sample_data/ is recommended for tiny, redistributable files.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_data"))

def find_first_lightcurve(root: Optional[str], exts: str = ".csv,.fits,.tbl") -> Optional[str]:
    """
    Walk `root` and return the first file whose extension matches `exts`.
    `exts` is a comma-separated list; case-insensitive.
    """
    if not root:
        return None
    wanted = [e.strip().lower() for e in exts.split(",") if e.strip()]
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if any(fn.lower().endswith(e) for e in wanted):
                return os.path.join(dirpath, fn)
    return None

def _compute_tpr(records: List[dict]) -> float:
    if not records:
        return 0.0
    det = sum(1 for r in records if r.get("detected"))
    return det / float(len(records))

def run_benchmark(
    path: str,
    *,
    label: Optional[str] = None,
    profile: str = "balanced",
    methods: Iterable[str] = DEFAULT_METHODS,
    time_col: Optional[str] = None,
    flux_col: Optional[str] = None,
    compute_budget: Optional[int] = None,
    adversarial: bool = False,
    adversarial_eps: float = 0.0,
    adversarial_seed: Optional[int] = 123,
    runs_keep: int = 12,
) -> str:
    """
    Run a benchmark and write artifacts into a fresh runs/ directory.
    Returns the path to the run folder.
    """
    # 1) Load light curve (your tb.load handles CSV/FITS/TBL & common column names)
    lc = tb.load(path, time=time_col, flux=flux_col, label=label)

    # 2) Configure grid and decision defaults (you can tweak as you like)
    depths   = (0.003, 0.005, 0.010)
    durations= (0.08, 0.12, 0.20)
    periods  = (1.5, 3.0, 5.0)

    # 3) Use a run context so everything lands in runs/bench-YYYYMMDD-.../
    with Run("bench", profile=profile, keep_last=runs_keep,
             meta={"source_path": os.path.abspath(path)}) as run:

        # 4) Execute benchmark
        res = lc.benchmark(
            method=tuple(methods),
            profile=profile,
            per_injection_search=True,
            decision_metric="zscore",
            snr_threshold=None,   # use profile default
            rel_window=None,      # use profile default
            durations=durations,
            depths=depths,
            periods=periods,
            compute_budget=compute_budget,  # e.g., 20000, or None for unlimited
            cost_model=None,                # or pass a recover.CostModel(...)
            oot_strategy="median",
            prewhiten_nharm=3,
            adversarial=adversarial,
            adversarial_eps=adversarial_eps,
            adversarial_seed=adversarial_seed,
        )

        # 5) Persist canonical JSON result
        run.write_json(res, "result.json")

        # 6) Save per-method record CSVs + a quick Markdown summary
        all_records: List[dict] = []
        for m, blk in res.get("methods", {}).items():
            recs = list(blk.get("records", []))
            all_records.extend(recs)
            save_records_csv(recs, run.path(f"{m}_records.csv"))

        write_md_summary(
            all_records,
            run.path("summary.md"),
            title=f"TransitBench — injection–recovery ({lc.label or 'lightcurve'})"
        )

        # 7) Print a tiny console summary
        print("\n== TransitBench summary ==")
        basic = res.get("basic", {})
        print(f"Label: {res.get('label')}")
        print(f"Basic: n_pts={basic.get('n_pts')} rms_oot={basic.get('rms_oot')} beta10={basic.get('beta10')}")
        for m, blk in res.get("methods", {}).items():
            recs = list(blk.get("records", []))
            tpr = _compute_tpr(recs)
            print(f"[{m}] TPR={tpr:.3f}  N={len(recs)}")

        print(f"\nArtifacts written to:\n  {run.dir}\n")
        return run.dir

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TransitBench use-case example")
    p.add_argument("--path", default=None,
                   help="Path to a single light curve file (CSV / FITS / TBL). If omitted, the first matching file under --data-root is used.")
    p.add_argument("--data-root", default=guess_demo_root(),
                   help="Directory containing demo light curves. Defaults to the bundled examples/sample_data/.")
    p.add_argument("--exts", default=".csv,.fits,.tbl",
                   help="Comma-separated file extensions to search under --data-root when --path is not provided.")
    p.add_argument("--label", default=None, help="Optional label for the light curve.")
    p.add_argument("--profile", default="balanced", choices=["sensitive", "balanced", "strict"])
    p.add_argument("--methods", default="oot-replace,prewhiten",
                   help="Comma-separated detrenders to run, e.g. 'oot-replace,prewhiten'")
    p.add_argument("--time-col", default=None, help="Optional time column name override.")
    p.add_argument("--flux-col", default=None, help="Optional flux column name override.")
    p.add_argument("--budget", type=int, default=None,
                   help="Optional compute budget (approx ops cost).")
    p.add_argument("--adversarial", action="store_true",
                   help="Enable adversarial stress (tiny flux perturbations).")
    p.add_argument("--adv-eps", type=float, default=0.0,
                   help="Adversarial epsilon (fractional).")
    p.add_argument("--runs-keep", type=int, default=12,
                   help="Keep only the latest N bench runs.")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    methods = tuple([m.strip() for m in args.methods.split(",") if m.strip()])

    # Resolve input file: prefer --path; otherwise pick the first demo file.
    selected_path = args.path
    if selected_path is None:
        selected_path = find_first_lightcurve(args.data_root, args.exts)
        if selected_path:
            print(f"[examples] --path not provided. Using demo file: {selected_path}")
        else:
            print(f"ERROR: no light curve found under {args.data_root} (exts={args.exts}).")
            sys.exit(2)

    run_benchmark(
        path=selected_path,
        label=args.label,
        profile=args.profile,
        methods=methods,
        time_col=args.time_col,
        flux_col=args.flux_col,
        compute_budget=args.budget,
        adversarial=args.adversarial,
        adversarial_eps=args.adv_eps,
        runs_keep=args.runs_keep,
    )