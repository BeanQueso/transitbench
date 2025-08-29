# TransitBench `src/` Developer Guide

This document gives a high-level map of the code that lives under `src/transitbench/`, what each module does, and how they fit together. It’s meant for contributors and power users who want to extend or debug the library.

> **Quick mental model:**  
> `io` loads a light curve → `detrend`/`features`/`bls` process it → `inject` adds synthetic transits (optional) → `recover` runs searches & scoring → `analysis`/`coverage`/`perf` summarize → `plot`/`artifacts`/`api` package the results.  
> `profiles` and `budget` keep runs consistent and fair; `adversarial` can stress-test the pipeline.

---

## Table of Contents

- [Data types used across the codebase](#data-types-used-across-the-codebase)
- [Module by module](#module-by-module)
  - [adversarial](#adversarial)
  - [analysis](#analysis)
  - [api](#api)
  - [artifacts](#artifacts)
  - [batch](#batch)
  - [bls](#bls)
  - [budget](#budget)
  - [cli_batch](#cli_batch)
  - [cli](#cli)
  - [core](#core)
  - [coverage](#coverage)
  - [detrend](#detrend)
  - [features](#features)
  - [inject](#inject)
  - [io](#io)
  - [perf](#perf)
  - [plot](#plot)
  - [profiles](#profiles)
  - [recover](#recover)
  - [targets](#targets)
  - [utils](#utils)
- [End-to-end flow (typical call graph)](#end-to-end-flow-typical-call-graph)
- [Extending TransitBench safely](#extending-transitbench-safely)

---

## Data types used across the codebase

While each module keeps its own classes and helpers, several shared shapes appear frequently:

- **LightCurve**  
  In-memory object (created in `core` from `io` outputs) that wraps arrays like `time`, `flux`, `flux_err`, (optional) `mask` and helpers like `.benchmark(...)`, `.copy()`, metadata (`.label`, `.source`, etc).

- **Records**  
  Per-injection results (Python dicts) with fields such as:
  - `depth`, `duration`, `period` (what we injected)
  - `snr_injected`, `snr_top` (SNRs measured)
  - `detected: bool` (decision outcome)
  - `match_period` (best period found), `zscore` (decision metric)

- **BenchmarkResult**  
  A dict like:
  ```json
  {
    "label": "...",
    "methods": {
      "oot-replace": {
        "records": [ ... ],
        "provenance": { "grid": "4000×1", "decision_metric": "zscore", ... }
      },
      "prewhiten": { ... }
    }
  }
  ```

- **Profiles / ProfileOverrides**  
  Parameter presets and overrides used by `recover` and `core` to keep experiments “budget-fair”.

- **CostModel / ComputeBudget**  
  Abstractions in `budget` used to compare methods at equal cost (e.g., “N period trials × M durations”).

---

## Module by module

### adversarial
**Purpose:** Adversarial/noise-stress utilities to sanity-check robustness of the recovery pipeline.

- Adds structured noise or perturbations (e.g., amplitude-bounded perturbations `ε`) to a light curve before detection.
- Provides small wrappers used by `tools/robustness.py` to sweep `ε` and evaluate TPR curves.
- **Typical usage (indirect):**
  ```python
  # used internally by LightCurve.benchmark(..., adversarial=True, adversarial_eps=...)
  noisy_flux = adversarial.apply_eps(flux, eps, rng=seed)
  ```

### analysis
**Purpose:** Post-processing of run outputs.

- Aggregates per-injection records into per-method statistics (TPR by depth/duration/period, SNR distributions, ROC-like curves for decision metrics).
- Produces summaries for papers and READMEs.

### api
**Purpose:** Stable, user-facing import surface.

- Re-exports friendly entry points like `load`, `LightCurve`, and convenience runners.
- Keep this minimal—only the functions you want users to rely on long-term.

### artifacts
**Purpose:** Writing out reproducible results and “paper artifacts”.

- Helper to save JSON/CSV/figures in structured folders (e.g., `runs/`, `paper/figs/`).
- Ensures filenames are slugified, includes small provenance blocks (profile name, commit hash if available).

### batch
**Purpose:** Internal batch orchestration utilities.

- Shared code behind scripts that iterate over directories of light curves, apply profiles, and emit summaries.
- Works hand-in-hand with `cli_batch` and `tools/*` scripts.

### bls
**Purpose:** Transit search primitives (Box Least Squares et al.).

- Provides a BLS search call (`bls.search(...)`) and helpers to refine peaks.
- Optimized for grid search over periods/durations with optional budget capping from `budget`.

### budget
**Purpose:** Compute-budget fairness & accounting.

- `CostModel`: estimate cost of a search (e.g., grid size × detrending passes).
- Helpers to cap searches to a budget and to compare methods at equal cost.
- Used by `core.LightCurve.benchmark(..., compute_budget=..., cost_model=...)`.

### cli_batch
**Purpose:** CLI entry points for multi-file runs.

- Powers commands like:
  ```bash
  python -m transitbench.cli_batch --root data/raw --pattern "*.csv" --profile balanced
  ```
- Thin shell around `batch` and `core`.

### cli
**Purpose:** CLI entry points for single-file operations.

- Commands like:
  ```bash
  python -m transitbench.cli benchmark path/to/file.csv --method oot-replace --profile balanced
  ```

### core
**Purpose:** The public “brain” that wires everything together.

- `load(path, ...)` → returns a `LightCurve` by delegating to `io`.
- `LightCurve.benchmark(...)`:
  - Optionally `inject` a grid of transits.
  - Run detrending (`detrend`), then `bls` search.
  - Score & decide in `recover` (z-score or other metrics).
  - Respect `profiles` and `budget`.
  - Return a `BenchmarkResult`.
- Handles labels, reproducibility metadata, and profile override application.

### coverage
**Purpose:** “Budget-fair” coverage estimates.

- Given a search grid and a budget, estimates how the period/duration grid is covered (e.g., “effective” number of trials).
- Useful for reporting and for setting apples-to-apples comparisons between methods.

### detrend
**Purpose:** Detrending & noise handling.

- Contains reference detrenders used in benchmarking:
  - **`oot-replace`**: out-of-transit replacement / masking strategy.
  - **`prewhiten`**: remove dominant systematics / periodicities before search.
- Exposes a uniform interface (`detrend.apply(method, lc, **params)`), enabling plug-and-play.

### features
**Purpose:** Feature engineering for machine-assisted vetting & QA.

- Extracts scalar features from a (detrended) light curve or from BLS outputs (e.g., depth proxies, local variance, skew/kurtosis, odd-even metrics).
- Downstream consumers can be heuristics in `recover` or external ML.

### inject
**Purpose:** Synthetic transit injection.

- Builds parametric transit signals and injects them into `LightCurve`:
  - Depth, duration, period grids.
  - Optionally multiple injections per file (per-injection search mode).
- Produces per-injection provenance that travels with `records`.

### io
**Purpose:** File I/O for real light curves.

- Transparent loader for `.csv`, `.fits`, `.tbl` (and friends), returning NumPy arrays in a common schema.
  - Looks for columns like: `time`, `flux`, `flux_err` (aliases supported).
  - Can parse typical TESS/Kepler formats and your custom CSVs.
- Normalizes metadata (`label`, source instrument, cadence).

**Example:**
```python
import transitbench as tb
lc = tb.load("data/raw/false_positives/…/lightcurve.csv")
print(lc.time.shape, lc.flux.shape, lc.label)
```

### perf
**Purpose:** Small performance helpers.

- Timing contexts, simple profilers, and counters used during runs and in `tools/smoke_all.py`.
- Non-invasive; safe to leave enabled for debug logging.

### plot
**Purpose:** Plotting utilities used by scripts and notebooks.

- Functions to draw robustness curves, TPR-by-depth, SNR distributions, and quick diagnostic light-curve views.
- High-level wrappers over `matplotlib`.

### profiles
**Purpose:** Named, documented parameter presets.

- Houses default profiles like `sensitive`, `balanced`, `strict`.
- Each profile can carry per-method parameters (e.g., `tau_base` thresholds) and global toggles.
- Supports in-repo **profile overrides** (merged at load time) used by tuning tools.

### recover
**Purpose:** Decision logic & metrics.

- Converts search outputs (e.g., BLS power, fitted depths) into decisions:
  - `detected: bool`, `zscore` (or alternative metrics), period matching.
  - Applies **thresholds** (e.g., `tau_base`) defined in `profiles`.
- Exposes helpers to:
  - Sample null distributions (for tau tuning).
  - Compute detection SNRs and confidence heuristics.
  - Apply per-profile overrides programmatically.
- This is where improvements most directly impact TPR/FPR.

### targets
**Purpose:** Target catalogs and convenience selectors.

- Helpers to fetch/normalize small target lists for experiments (TOIs, candidates, false positives).
- Returns minimal metadata for labeling and filtering.

### utils
**Purpose:** Cross-cutting utilities.

- `ensure_dir`, `slugify`, JSON/CSV writers, RNG helpers, masking utilities, safe percentiles, etc.
- Keep this clean; anything domain-specific belongs in the domain module instead.

---

## End-to-end flow (typical call graph)

```text
io.load_*  ->  core.load  ->  LightCurve
                                |
                                +-- (optional) inject.grid(...)  -> injections
                                |
                                +-- detrend.apply(method, ...)   -> cleaned LC
                                |
                                +-- bls.search(...)              -> periodogram & peaks
                                |
                                +-- recover.decide(...)          -> records (detected?, scores)
                                |
                                +-- analysis.summarize(...)      -> TPR/SNR stats
                                |
                                +-- artifacts.write_* / plot.*   -> JSON/CSV/PNGs
```

`budget` can cap the BLS grid; `profiles` select consistent parameters; `adversarial` can perturb the flux prior to search; `coverage` reports fairness; `features` computes extra vetting signals.

---

## Extending TransitBench safely

1. **Add a new detrend method**
   - Implement in `detrend/your_method.py` and register via a small dispatcher (e.g., `detrend.apply("your-method", ...)`).
   - Add defaults in `profiles` (per-method section).
   - Update `recover` thresholds only if you have null-based evidence (use `tools/quick_tau.py`).

2. **Change I/O or support a new format**
   - Extend `io` with a loader; keep the output schema the same (`time`, `flux`, `flux_err`).
   - Add an alias mapping if column names differ.
   - Wire into `core.load`.

3. **Create a new decision metric**
   - Add a metric function in `recover` and a name (e.g., `"zscore2"`).
   - Allow `LightCurve.benchmark(..., decision_metric="zscore2")`.
   - Provide a null sampler (so profiles can be tuned).

4. **Keep runs reproducible**
   - Make sure new knobs are controlled by `profiles`.
   - Respect `compute_budget` and consult `budget.CostModel` if your method changes cost.

5. **Document the change**
   - Update this README and the top-level README.
   - If there’s a script in `tools/` that exercises the feature, add an example invocation.

---

### Glossary (quick)

- **TPR**: True Positive Rate = fraction of injected transits that are recovered (higher is better).
- **FPR**: False Positive Rate = fraction of null (no transit) cases flagged as transit (lower is better).
- **SNR**: Signal-to-Noise ratio for the injected and/or recovered event.
- **Profile**: Named preset of parameters (thresholds, grid sizes, detrend settings).
- **Budget-fair**: Compare methods at equal compute cost (not equal grid sizes).

---

### Minimal code snippets

**Benchmark one file programmatically**
```python
import transitbench as tb

lc = tb.load("data/raw/false_positives/…/lightcurve.csv")
res = lc.benchmark(
    method=("oot-replace","prewhiten"),
    profile="balanced",
    decision_metric="zscore",
    per_injection_search=True,
    depths=(0.003, 0.005, 0.010),
    durations=(0.08, 0.12, 0.20),
    periods=(1.5, 3.0, 5.0),
)
print(res["methods"]["oot-replace"]["provenance"])
```

**Apply profile overrides (e.g., tuned taus)**
```python
from transitbench import recover

overrides = {
  "tau_base": {
    "oot-replace": 10.0,
    "prewhiten": 16.0
  }
}
recover.apply_profile_overrides(overrides)
```

---

If you spot drift between this guide and the code, prefer the code and open a PR to update the README. Happy benchmarking!
