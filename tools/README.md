# TransitBench Tools

Utility scripts that sit on top of the core `transitbench` API to help you **run quick experiments, aggregate results, and make paper‑ready figures**. These tools assume you are in the project root (the directory that contains `tools/` and `src/`).

> Tip: All tools are safe to run even if some inputs are missing; they try to skip problematic files and continue.

---

## Quick map

- `aggregate.py` — Collate JSON results under `runs/` into CSV/JSON summaries.
- `apply_profile_overrides.py` — Load `profile_overrides.json` and apply it to the in‑memory `recover.PROFILES`.
- `figures.py` — Produce plots from aggregated outputs (robustness curves, TPR vs depth).
- `quick_tau.py` — Sample *null* (no‑transit) scores to suggest `tau_base` thresholds per method.
- `robustness.py` — Compute per‑file **adversarial robustness** curves and write one JSON per file/epsilon.
- `run_completeness.py` — Batch **injection–recovery completeness** runs over a folder of light curves.
- `smoke_all.py` — One‑command **smoke test**: mini benchmark + (optional) robustness + aggregation + figures.
- `tune_profiles.py` — End‑to‑end tuning helper that calls `quick_tau` logic and writes `profile_overrides.json`.

---

## Conventions & folders

- **Input data**: typically under `data/raw/...` (CSV/FITS/TBL). The core loader `tb.load()` figures out the format.
- **Runs** (generated):
  - `runs/smoke/*.json` — per‑file injection–recovery summaries from smoke tests.
  - `runs/robustness/*.json` — one JSON per file per method per epsilon.
- **Paper artifacts** (generated):
  - `paper/records_all.csv` — *long* table of all injection records.
  - `paper/summary.json` — global counts/tables (e.g., TPR per method).
  - `paper/figs/*.png` — robustness curves & TPR‑by‑depth plots.
- **Tuning**:
  - `profile_overrides.json` — key:value overrides (e.g., `tau_base`) that the library will respect at runtime.

---

## Common quickstart

```bash
# 1) Run a tiny smoke benchmark on a handful of files
python tools/smoke_all.py --glob 'data/raw/**/hlsp_*lc.csv' --max-files 6

# 2) (optional) add a mini robustness sweep
python tools/smoke_all.py --glob 'data/raw/**/hlsp_*lc.csv' --max-files 6 --robustness

# 3) Aggregate + make figures (also happens at the end of smoke_all)
python tools/aggregate.py
python tools/figures.py
```

---

## `aggregate.py`

**Purpose.** Walk `runs/` and combine per‑run JSONs into:
- `paper/records_all.csv` — row per injection (depth, duration, period, SNR, detected, method, label, profile).
- `paper/summary.json` — quick stats (global TPR per method, counts).

**CLI.**
```bash
python tools/aggregate.py \
  --runs-root runs \
  --paper-root paper
```

**Notes.**
- Robust to partially written files; skips unreadable JSONs with a warning.
- Designed so `figures.py` can consume the outputs directly.

---

## `apply_profile_overrides.py`

**Purpose.** Read `profile_overrides.json` (produced by tuning) and apply values to the live `recover.PROFILES`
dictionary so subsequent runs use the new thresholds.

**CLI.**
```bash
python tools/apply_profile_overrides.py \
  --file profile_overrides.json \
  --profile balanced
```

**What it changes.**
- Keys like `tau_base` (per method) or other numeric knobs that profiles expose.
- Only affects the *current process* unless you persist the file and import on startup.

**Gotchas.**
- If the overrides file is missing keys, they are ignored; nothing is set to `None`.

---

## `figures.py`

**Purpose.** Make paper‑ready plots.

- **Robustness curves** averaged over files for each method.
- **TPR vs depth** per method using `paper/records_all.csv`.

**CLI.**
```bash
# plot robustness curves from runs/robustness/*.json
python tools/figures.py

# (figures.py also tries to plot TPR-by-depth if paper/records_all.csv exists)
```

**Outputs.**
- `paper/figs/robustness_*.png`
- `paper/figs/tpr_by_depth_*.png`

---

## `quick_tau.py`

**Purpose.** Fast estimate of `tau_base` from *null* (no‑transit) score distributions:
- Samples the null z‑score (or decision metric) on each file.
- Reports percentiles (e.g., p50, p95, p99) and suggests **sensitive/balanced/strict** `tau_base`.
- Writes `profile_overrides.json`.

**CLI.**
```bash
python tools/quick_tau.py \
  --glob 'data/raw/**/hlsp_*lc.csv' \
  --methods oot-replace prewhiten \
  --profile balanced \
  --samples-per-file 600
```

**Outputs.**
- Console table of per‑file and global percentiles.
- `profile_overrides.json` in repo root by default.

**Notes.**
- A few files may be skipped (e.g., malformed inputs); the tool continues and prints *“skipped (no null)”* or an error.

---

## `robustness.py`

**Purpose.** For each file, sweep **adversarial ε** and measure **TPR** per method at a fixed compute budget.

**Programmatic use.**
```python
from tools.robustness import run_robustness_for_list
paths = [...]  # list of LC files
written = run_robustness_for_list(paths,
                                  methods=("oot-replace","prewhiten"),
                                  profile="balanced",
                                  epsilons=(0.0, 0.25, 0.5, 1.0),
                                  durations=(0.08, 0.12, 0.20),
                                  depths=(0.003, 0.005, 0.010),
                                  periods=(1.5, 3.0, 5.0),
                                  compute_budget=600)
print("Wrote:", written)
```

**Outputs (per file × method × ε).**
- `runs/robustness/{label}__{method}__eps{ε}.json`

**Notes.**
- Uses the same benchmark grid across methods for budget‑fairness.
- Figures are produced later by `tools/figures.py`.

---

## `run_completeness.py`

**Purpose.** Batch **injection–recovery completeness** study across a folder:
- Injects a grid of (depth, duration, period).
- Runs the chosen recovery method(s).
- Computes TPR and SNR stats; writes a summary JSON per file.

**CLI.**
```bash
python tools/run_completeness.py \
  --glob 'data/raw/**/hlsp_*lc.csv' \
  --methods oot-replace prewhiten \
  --profile balanced \
  --depths 0.003 0.005 0.010 \
  --durations 0.08 0.12 0.20 \
  --periods 1.5 3.0 5.0 \
  --budget 600
```

**Outputs.**
- Per‑file JSONs in `runs/completeness/`
- These can be aggregated by `aggregate.py` (they share the same record schema).

---

## `smoke_all.py`

**Purpose.** One command that:
1) Picks a small set of files (`--glob`, `--max-files`).
2) Runs a short benchmark to `runs/smoke/`.
3) Optionally runs a tiny robustness sweep to `runs/robustness/`.
4) Aggregates everything and makes figures.

**CLI.**
```bash
python tools/smoke_all.py \
  --glob 'data/raw/**/hlsp_*lc.csv' \
  --max-files 6 \
  --robustness
```

**Outputs.**
- `runs/smoke/*.json`
- (optional) `runs/robustness/*.json`
- `paper/records_all.csv`, `paper/summary.json`
- `paper/figs/robustness_*.png`, `paper/figs/tpr_by_depth_*.png`

**Notes.**
- If robustness is skipped, you’ll still get `TPR vs depth` figures from the smoke runs.
- Printed **n_pts** is the number of cadence points in the loaded light curve.

---

## `tune_profiles.py`

**Purpose.** End‑to‑end profile tuning helper:
- Calls the *null* sampler to estimate `tau_base` (per method).
- Writes `profile_overrides.json`.
- (Optionally) applies those overrides in‑process so subsequent runs pick them up.

**CLI.**
```bash
python tools/tune_profiles.py \
  --glob 'data/raw/**/hlsp_*lc.csv' \
  --methods oot-replace prewhiten \
  --profile balanced
```

**Outputs.**
- `profile_overrides.json`
- Console log like:
  ```
  oot-replace: TOTAL N=7200, p50=6.47, p95=2383.82, p99=3285.02
    tau suggestions: {'sensitive': 2868.14, 'balanced': 3285.02, 'strict': 3305.08}
  ```

**Notes.**
- Extremely large suggested taus on a few files usually indicate strong systematics; re‑run `quick_tau.py` on a **cleaner subset** if needed and prefer p95/p99 over the maximum.

---

## Troubleshooting

- **`ModuleNotFoundError: tools.*`** — run from repo root (`python tools/smoke_all.py`), not from inside `tools/`.
- **Many “skipped (no null)” messages** — some files don’t support null sampling; the tools proceed with the rest.
- **No figures appear** — ensure `paper/records_all.csv` exists (run `aggregate.py`) or `runs/robustness/*.json` for robustness.
- **“unexpected keyword argument 'inj_durations'”** — update your scripts to use the current `LightCurve.benchmark` argument names: `durations`, `depths`, `periods` (not the older `inj_*` names).
- **Where do numbers like `n_pts` come from?** — they’re simply the length of the loaded light curve after any requested filtering.

---

## Programmatic import

All tools are importable if you prefer calling them from your own scripts:

```python
from tools.aggregate import aggregate_runs
from tools.figures import plot_robustness, plot_tpr_by_depth
from tools.robustness import run_robustness_for_list
# etc.
```

They’re thin wrappers on the public API in `src/transitbench/` and keep outputs under `runs/` and `paper/` by default.
