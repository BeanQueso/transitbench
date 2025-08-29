# TransitBench — Examples

This folder contains runnable, end-to-end examples that show how to use **TransitBench** to run injection–recovery benchmarks on light curves, save results, and produce quick summaries and figures.

> If you’re new here, start with **`examples/bench_usecase.py`** — a single-file script that:
> - loads a light curve (CSV / FITS / TBL; columns auto-detected when possible),
> - runs injection–recovery with `oot-replace` and `prewhiten`,
> - optionally enables adversarial “stress”,
> - and writes all artifacts to a timestamped folder under `runs/`.

---

## Table of Contents

- [Requirements](#requirements)
- [Quick Start (60 seconds)](#quick-start-60-seconds)
- [What the Example Does](#what-the-example-does)
- [Command Line Usage](#command-line-usage)
- [Outputs & Directory Layout](#outputs--directory-layout)
- [Interpreting Results (TPR, SNR, etc.)](#interpreting-results-tpr-snr-etc)
- [Programmatic Use (import the helper)](#programmatic-use-import-the-helper)
- [Making Plots & Aggregates](#making-plots--aggregates)
- [Troubleshooting](#troubleshooting)
- [Cite / Acknowledge](#cite--acknowledge)
- [License](#license)

---

## Requirements

- Python **≥ 3.9**
- Installed TransitBench and its dependencies

Install TransitBench from the repo root in editable mode:

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate  # or use your preferred env manager
pip install --upgrade pip
pip install -e .

# (optional) developer extras, linters, etc.
# pip install -e .[dev]
```

---

## Quick Start (60 seconds)

Run the example against one of your light-curve files:

```bash
# from the repository root
source .venv/bin/activate

python examples/bench_usecase.py \
  --path data/raw/false_positives/9727392_236.01/sector2/hlsp_tess-data-alerts_tess_phot_00009727392-s02_tess_v1_lc.csv \
  --profile balanced \
  --methods oot-replace,prewhiten
```

This will print a compact summary and create a timestamped folder under `runs/` with JSON, CSVs, and a Markdown summary.

---

## What the Example Does

`examples/bench_usecase.py` demonstrates a typical benchmarking flow:

1. **Load** a light curve with `transitbench.load(...)`  
   (CSV/FITS/TBL supported; time/flux columns auto-detected for common names, or override via `--time-col`/`--flux-col`).

2. **Run injection–recovery** with one or more detrenders (default: `oot-replace`, `prewhiten`) over a small grid:
   - depths: `0.003, 0.005, 0.010`
   - durations: `0.08, 0.12, 0.20` (days)
   - periods: `1.5, 3.0, 5.0` (days)

3. **Save artifacts** in a fresh `runs/bench-YYYYMMDD-HHMMSS/` folder:
   - canonical `result.json`
   - per-method `*_records.csv`
   - `summary.md` with a human-friendly overview

4. **(Optional) Adversarial stress**: tiny flux perturbations to test robustness (`--adversarial --adv-eps 0.5`).

---

## Command Line Usage

```bash
python examples/bench_usecase.py --help
```

You’ll see flags like:

- `--path` — path to the light curve file (CSV / FITS / TBL)
- `--label` — optional custom label for the LC
- `--profile` — `sensitive | balanced | strict`
- `--methods` — comma-separated detrenders (e.g., `oot-replace,prewhiten`)
- `--time-col` / `--flux-col` — override column names if auto-detect fails
- `--budget` — compute budget proxy (integer); leave empty for unlimited
- `--adversarial` — enable adversarial perturbations
- `--adv-eps` — adversarial epsilon (e.g., `0.5`)
- `--runs-keep` — keep only the last N run folders (auto-prunes older runs)

**Examples**

```bash
# 1) Default two-method benchmark
python examples/bench_usecase.py --path /path/to/lightcurve.csv

# 2) Stricter profile, custom columns, and a small budget
python examples/bench_usecase.py \
  --path /path/to/lightcurve.tbl \
  --profile strict \
  --time-col TIME --flux-col SAP_FLUX \
  --budget 20000

# 3) With adversarial stress
python examples/bench_usecase.py \
  --path /path/to/lightcurve.fits \
  --adversarial --adv-eps 0.5
```

---

## Outputs & Directory Layout

A typical run produces:

```
runs/
  bench-2025-08-28_12-34-56/       # new folder per run
    result.json                    # canonical result (per-method blocks, metadata, etc.)
    oot-replace_records.csv        # per-injection records for this method
    prewhiten_records.csv
    summary.md                     # quick human-friendly summary
```

**`result.json` highlights**

- `label` — LC label
- `basic` — stats like `n_pts`, `rms_oot`, `beta10`
- `methods` — one block per detrending method:
  - `records` — each row = one injected signal with fields like
    - `depth`, `duration`, `period`
    - `detected` (boolean)
    - `snr_injected`, `zscore`, etc.

**Per-method `*_records.csv`**

- Flattened view of `records` for spreadsheets / plotting.

**`summary.md`**

- Terse overview of TPR per method and run metadata.

---

## Interpreting Results (TPR, SNR, etc.)

- **TPR (True Positive Rate)**: fraction of injected signals that were recovered as detections.  
  *Higher is better.*

- **SNR (Signal-to-Noise Ratio)**: approximate strength of the injected transit relative to noise in the processed LC.  
  *Useful for understanding borderline cases.*

- **Profiles (`sensitive`, `balanced`, `strict`)**: preset thresholds/windows.  
  *Sensitive* may recover more weak signals but risks more false alarms; *strict* is conservative.

- **Adversarial ε**: small fractional flux perturbation to test robustness; larger ε is more challenging.

---

## Programmatic Use (import the helper)

You can call the example’s core function from your own code:

```python
from examples.bench_usecase import run_benchmark

run_dir = run_benchmark(
    path="data/raw/false_positives/.../lightcurve.csv",
    profile="balanced",
    methods=("oot-replace", "prewhiten"),
    compute_budget=None,        # or an int like 20000
    adversarial=False,
    adversarial_eps=0.0,
    runs_keep=12,
)

print("Artifacts in:", run_dir)
```

This returns the path to the new `runs/bench-*` folder.

---

## Making Plots & Aggregates

Beyond this example, the repo ships with small utilities under `tools/`:

- **`tools/smoke_all.py`** — run a tiny benchmark suite over a handful of files (quick sanity check).
- **`tools/aggregate.py`** — gather run outputs into a single CSV for analysis.
- **`tools/figures.py`** — produce robustness curves and TPR vs. depth plots from aggregated outputs.
- **`tools/robustness.py`** — run a small adversarial sweep and serialize curves for plotting.

Typical flow:

```bash
# 1) Run the smoke test over a few files
python tools/smoke_all.py --glob 'data/raw/**/*.csv' --max-files 6

# 2) Aggregate runs into a paper-ready CSV
python tools/aggregate.py --runs-root runs --out paper/records_all.csv

# 3) Make figures (robustness curves & TPR vs. depth)
python tools/figures.py
# -> figures saved under paper/figs/
```

> Tip: keep a small, public subset of `data/raw/` in the repo (e.g., 3–6 files) so others can reproduce your examples quickly.

---

## Troubleshooting

- **Auto-detection of columns fails**  
  Use `--time-col` and `--flux-col` to name your columns explicitly.

- **“unexpected keyword argument ‘inj_durations’”**  
  The current API uses `durations`, `depths`, and `periods`. Update any local scripts that still pass `inj_*` names.

- **Nothing gets detected**  
  Try `--profile sensitive`, increase depths slightly, or allow a larger compute budget (`--budget 20000`).

- **Too many old runs**  
  Use `--runs-keep N` to auto-prune old `runs/bench-*` folders.

- **Matplotlib can’t write files**  
  Ensure the `paper/figs/` directory exists (the provided scripts will create it automatically).

---

## Cite / Acknowledge

If this example (or TransitBench) helps your work, please cite the repository in your acknowledgements.  
A BibTeX stub you can adapt:

```bibtex
@misc{transitbench2025,
  title        = {TransitBench: Injection–Recovery Benchmarking for Transit Searches},
  author       = {Tripathi, Eshaan and contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-org>/transitbench}}
}
```

---

## License

This examples directory follows the project’s root license. See `LICENSE` at the repo root.
