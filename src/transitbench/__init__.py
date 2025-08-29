from .io import load_timeseries
from .features import basic_features
from .bls import run_bls
from .inject import inject_box_transit
from .recover import inject_and_recover, get_clean_series, CostModel
from .plot import plot_timeseries, save_raw_and_clean, cleanmode_suffix
from .detrend import rolling_median_highpass
from .coverage import phase_coverage, n_transits_observed
from .perf import (
    plot_detection_vs_coverage, plot_detection_vs_events,
    plot_completeness_heatmap, write_records_csv, write_completeness_table_csv
)
from .budget import make_budgeted_grid
from .api import score_lightcurve, score_lightcurve_metrics, to_text_report, print_report, save_report, run_adversarial
from .batch import run_batch
from . import utils, io, recover, batch
from .core import LightCurve, load

__all__ = [
    "load_timeseries","basic_features","run_bls","inject_box_transit",
    "inject_and_recover","get_clean_series","plot_timeseries",
    "save_raw_and_clean","rolling_median_highpass",
    "phase_coverage","n_transits_observed",
    "plot_detection_vs_coverage","plot_detection_vs_events",
    "plot_completeness_heatmap","write_records_csv","write_completeness_table_csv",
    "make_budgeted_grid","save_raw_and_clean","cleanmode_suffix","score_lightcurve",
    "run_batch","utils", "io", "recover", "batch", "to_text_report", "print_report", "save_report",
    "score_lightcurve_metrics", "run_adversarial", "LightCurve", "load", "CostModel"
]

__version__ = "0.1.0"