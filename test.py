import pandas as pd
df = pd.read_csv("/Users/eshaantripathi/Documents/transitbench/paper/TOI3669-01_20230906_GMUO_R_measurements.tbl", delim_whitespace=True, comment="#")
fwhm_px = df.get("FWHM_Mean")
print(f"Median FWHM(px) = {fwhm_px.median():.2f}") if fwhm_px is not None else None