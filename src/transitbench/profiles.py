# transitbench/profiles.py
from __future__ import annotations
from typing import Dict, Any

# Central place for preset knobs used by recover.inject_and_recover + cleaners.
PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    # High sensitivity for curated, high-S/N LCs (e.g., TESS-alert).
    "sensitive": dict(
        decision_metric="zscore",
        rel_window=0.03,                      # ±3%
        durations=(0.06, 0.08, 0.10, 0.12, 0.16, 0.20),
        snr_thresh=5.5,
        per_injection_search=True,
        window_days=0.30,
        mask_pad=3.5,
        oot_strategy="median",                # flat-ish inside masked windows
    ),
    # Balanced defaults for general use.
    "balanced": dict(
        decision_metric="zscore",
        rel_window=0.02,
        durations=(0.08, 0.12, 0.20),
        snr_thresh=6.0,
        per_injection_search=True,
        window_days=0.30,
        mask_pad=3.0,
        oot_strategy="median",
    ),
    # FP-averse, stricter metric and tighter tolerance.
    "strict": dict(
        decision_metric="depth_over_err",
        rel_window=0.0125,
        durations=(0.08, 0.12, 0.20),
        snr_thresh=6.5,
        per_injection_search=True,
        window_days=0.30,
        mask_pad=3.0,
        oot_strategy="resample",              # preserves local variance
    ),
}

def resolve_profile(profile: str | None, overrides: dict | None = None) -> dict:
    """
    Returns a dict of parameters for recover.inject_and_recover and cleaners.
    Unknown profiles fall back to 'balanced'. 'overrides' wins on conflicts.
    """
    if not profile:
        base = dict(PROFILE_PRESETS["balanced"])
    else:
        p = str(profile).lower()
        base = dict(PROFILE_PRESETS.get(p, PROFILE_PRESETS["balanced"]))
    if overrides:
        base.update(overrides)
    return base
