# TransitBench

**A small, focused toolkit for stress‑testing transit detection pipelines on real light curves.**  
TransitBench loads your light curve, computes core noise metrics, runs one or more baseline detrend+search methods, and evaluates _detection performance_ by injection–recovery under fair, budgeted settings. It also supports quick adversarial robustness checks and paper‑ready figures.

---

## Table of contents

- [Why TransitBench?](#why-transitbench)
- [Features](#features)
- [Install](#install)
- [Supported inputs](#supported-inputs)
- [Quickstart](#quickstart)
  - [CLI (one‑liner)](#cli-one-liner)
  - [Python API](#python-api)
- [Core concepts](#core-concepts)
- [Profiles & thresholds](#profiles--thresholds)
- [Caching & performance](#caching--performance)
- [Reproducibility](#reproducibility)
- [Outputs & file structure](#outputs--file-structure)
- [Utilities (tools/)](#utilities-tools)
- [Figures](#figures)
- [API sketch](#api-sketch)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License & citation](#license--citation)

---

## Why TransitBench?

Most transit surveys mix heterogeneous data quality, methods and thresholds. TransitBench gives you a **consistent yardstick**:

- Standardized metrics: rms\_oot, β10, injected SNR, detection rate.
- **Budget‑fair** comparisons across methods.
- **Injection–recovery** summaries you can paste into a paper or supplement.
- Tiny **adversarial** nudges to sanity‑check robustness.

Use it to calibrate your pipeline, tune thresholds, or produce quick evidence that a candidate would be detectable (or not) given your data quality.

---

## Features

- **Robust I/O**: CSV, FITS, and TBL readers with flexible column mapping.
- **Noise metrics**: out‑of‑transit RMS, time‑correlated noise proxy (β10), point count.
- **Methods included**:  
  - `oot-replace` — out‑of‑transit replacement detrend + BLS scan  
  - `prewhiten` — simple prewhitening + BLS
- **Injection–recovery benchmarking**: depth × duration × period grids, per‑injection searches, paper‑ready textual summaries.
- **Budget fairness**: identical search grids / compute budgets per method.
- **Adversarial robustness**: small perturbations (ε) and TPR vs ε curves.
- **Threshold profiles**: `sensitive`, `balanced`, `strict` with easy overrides.
- **Caching**: memoize repeated null sampling / grid work to cut runtimes.
- **Utilities**: directory smoke tests, robustness sweeps, tau suggestion, aggregation & figures.

---

## Install

```bash
# 1) create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# 2) install TransitBench in editable mode
pip install -e .

# (optional) dev extras
# pip install -e .[dev]
```

**Requirements:** Python ≥ 3.9 and a reasonably recent NumPy/SciPy/astropy stack.

---

## Supported inputs

TransitBench can read:

- **CSV**: arbitrary column names; specify via loader/CLI flags.
- **FITS**: common LC HDU conventions.
- **TBL**: ASCII table (.tbl) with comment/header autodetection.

Typical column names (case‑insensitive) that auto‑map:

- time: `time`, `bjd`, `bjd_tdb`, `jd`
- flux: `flux`, `rel_flux`, `sap_flux`, `pdcsap_flux`
- flux error (optional): `flux_err`, `flux_error`

You can always override via parameters (see Quickstart).

---

## Quickstart

### CLI (one‑liner)

Score a single light curve and print a summary:

```bash
tb-score path/to/lightcurve.csv \
  --time-col BJD_TDB --flux-col rel_flux --flux-err-col flux_err \
  --methods oot-replace prewhiten --profile balanced
```

This will compute basic noise, run a small injection–recovery, and print detection summaries. Results are also written under `runs/`.

### Python API

```python
import transitbench as tb

# Load (CSV/FITS/TBL); override columns if needed
lc = tb.load("path/to/lightcurve.csv",
             time_col="BJD_TDB", flux_col="rel_flux", flux_err_col="flux_err")

# Quick noise readout
print(lc.summary())   # n_pts, rms_oot, beta10, etc.

# Budget-fair injection–recovery
res = lc.benchmark(
    method=("oot-replace", "prewhiten"),
    profile="balanced",
    per_injection_search=True,
    decision_metric="zscore",
    durations=(0.08, 0.12, 0.20),
    depths=(0.003, 0.005, 0.010),
    periods=(1.5, 3.0, 5.0),
    compute_budget=600,     # small/fast; increase for thorough scans
    seed=123
)

# Pretty-print paper-style summary
print(tb.format_summary(res))
```

---

## Core concepts

- **rms\_oot** — out‑of‑transit RMS of the flux series (noise floor).
- **β10** — coarse red‑noise proxy: ratio of RMS in 10‑point bins vs point RMS.
- **SNR\_injected** — matched‑filter SNR of the synthetic transit you injected
  (useful for grading detectability independent of the search).
- **TPR** — true positive rate; fraction of injections recovered by the search
  (higher is better). Reported globally and by depth/duration.
- **Adversarial ε** — small perturbation magnitude applied to the LC to probe
  method robustness; you plot TPR vs ε.

---

## Profiles & thresholds

TransitBench ships with three human‑friendly **profiles** that bundle detection thresholds and a few method knobs:

- **sensitive** — favors recall (higher TPR, more false alarms)
- **balanced** — tradeoff between TPR and specificity
- **strict** — favors precision (lower false alarms)

Internally, profiles use a base threshold (τ) per method. You can **tune** τ to your dataset:

```bash
# Suggest τ from null-sampling (fast)
python tools/quick_tau.py --root data/raw --methods oot-replace prewhiten

# Apply profile overrides produced above
python tools/tune_profiles.py --overrides profile_overrides.json
```

The overrides are merged into `recover.PROFILES` at import time so all downstream runs use your tuned thresholds.

---

## Caching & performance

Long‑running parts (null sampling, grid scans) are cached. You can:

- set a cache directory via env var `TRANSITBENCH_CACHE` (defaults under `runs/.cache/`)
- clear it by deleting that folder
- expect **order‑of‑magnitude** speedups across repeated runs with the same data/method/grid

For fast iteration, start with small grids and budgets; crank up once the wiring looks good.

---

## Reproducibility

- All randomized components accept a `seed`.
- Outputs (JSON/CSV/figs) include method, profile, grid, and seed metadata.
- `runs/` is structured to be diff‑friendly for Git.

---

## Outputs & file structure

Default locations (created on demand):

```
runs/
  smoke/                # JSONs from quick smoke tests
  robustness/           # adversarial sweep JSONs
paper/
  records_all.csv       # aggregated injection–recovery records
  figs/                 # generated figures (robustness, TPR-by-depth)
```

Each benchmark JSON contains:
- basic LC stats (`n_pts`, `rms_oot`, `beta10`)
- per‑method block with grid, threshold τ, and **records** (one per injection) including:
  `depth`, `dur`, `period_injected`, `snr_injected`, `snr_top`, `detected`, and match info.

---

## Utilities (tools/)

The repository includes small, composable scripts (run with `python tools/<name>.py ...`):

- **`smoke_all.py`** — end‑to‑end micro‑pipeline on a handful of files: benchmark → tiny robustness → aggregate → figures.  
  Example:
  ```bash
  python tools/smoke_all.py --root data/raw --limit 6 --profile balanced
  ```

- **`robustness.py`** — run a sweep of adversarial ε per file and method; writes JSONs under `runs/robustness/`.

- **`aggregate.py`** — collect `runs/*` into `paper/records_all.csv` and compute summary TPR by method.

- **`figures.py`** — generate paper‑ready plots (robustness curves, TPR vs depth) into `paper/figs/`.

- **`quick_tau.py`** — sample null distributions quickly and **suggest τ** for each method/profile.

- **`tune_profiles.py`** — apply suggested τ to your active `recover.PROFILES` via a JSON overrides file.

Each tool prints where it wrote outputs.

---

## Figures

After you have some runs:

```bash
# aggregate injection–recovery records
python tools/aggregate.py

# make figures into paper/figs/
python tools/figures.py
```

You’ll typically get:

- `robustness_<method>.png` — mean TPR vs ε with 10–90% band
- `tpr_by_depth_<method>.png` — detection rate vs injected depth

---

## API sketch

A minimal look at the API surface (subject to change):

```python
import transitbench as tb
from transitbench import recover

lc = tb.load("file.fits", time_col="TIME", flux_col="PDCSAP_FLUX", flux_err_col="PDCSAP_FLUX_ERR")

# Quick stats
info = lc.summary()  # dict: n_pts, rms_oot, beta10, ...

# Budget-fair benchmark
res = lc.benchmark(
    method=("oot-replace", "prewhiten"),
    profile="balanced",
    per_injection_search=True,
    decision_metric="zscore",
    durations=(0.08, 0.12, 0.20),
    depths=(0.003, 0.005, 0.010),
    periods=(1.5, 3.0, 5.0),
    compute_budget=600,
    seed=123
)

# Optional: small adversarial test
res_adv = lc.benchmark(method=("oot-replace",), profile="balanced",
                       adversarial=True, adversarial_eps=0.5, seed=123)
```

---

## Troubleshooting

- **`'flux' is not in list` / wrong columns**  
  Specify columns explicitly in CLI or `tb.load(..., time_col=..., flux_col=..., flux_err_col=...)`.

- **Empty or tiny robustness plots**  
  Ensure `runs/robustness/` contains JSONs. Re‑run `tools/robustness.py` first.

- **`unexpected keyword argument 'inj_durations'`**  
  You are calling an older API (or mixed code). Use `durations=...`, `depths=...`, `periods=...` as shown above.

- **`IsADirectoryError: ... is a directory` when writing a file**  
  Point file writes to a full path (including filename), not just a directory.

- **Runtime warnings (e.g., overflow in divide)**  
  Usually benign during grid scans; they occur when depth errors go to zero. We guard with `np.where` and ignore those samples in summary stats.

- **Nothing appears under `paper/`**  
  Run `python tools/aggregate.py` then `python tools/figures.py`.

---

## Contributing

Issues and PRs are welcome. Please:
- keep changes small and well‑scoped
- include a short reproduction (data snippet or steps)
- add/update a smoke test under `tools/`

A basic dev loop:

```bash
git checkout -b feature/my-change
# edit code
python tools/smoke_all.py --limit 3
pytest  # if you’ve added tests
git commit -am "feat: explain what changed"
git push origin feature/my-change
```

---

## Citation
- **Cite:** If you use TransitBench in a publication, please cite the repository and the specific release tag used.

## AI Transparency
This project was built with the help of an AI: GPT-5
