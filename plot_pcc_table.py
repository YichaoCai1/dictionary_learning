import os
import pathlib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
BASE_DIR  = "results"
CHECK_DIR = "ae_20000"
CSV_STEM  = "matched_summary_exp"     # ← no “.csv” needed
METHODS = [
    "topk_24k",
    "batchtopk_24k",
    "panneal_24k",
    "panneallora_1e-1_24k",
]
LEGENDS = {
    "topk_24k"             : r"top-$k$",
    "batchtopk_24k"        : r"batch-top-$k$",
    "panneal_24k"          : r"$p$-annealing",
    "panneallora_1e-1_24k" : r"$p$-annealing-LoRa",
}
# ---------------------------------------------------------------

def locate_csv(method_folder: str) -> pathlib.Path:
    root = pathlib.Path(BASE_DIR) / method_folder / CHECK_DIR
    for suffix in (".csv", ""):
        p = root / f"{CSV_STEM}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No '{CSV_STEM}' CSV for {method_folder}")

def read_metrics(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)
    if {"concept", "matched_corr"} - set(df.columns):
        raise ValueError(f"'concept' or 'matched_corr' missing in {csv_path}")
    is_avg = df["concept"].str.contains("avg", case=False, na=False)
    concept_rows = df.loc[~is_avg].copy()
    avg_row      = df.loc[is_avg]
    concept_pcc  = dict(zip(concept_rows["concept"],
                            concept_rows["matched_corr"].astype(float)))
    avg_pcc      = (float(avg_row["matched_corr"].iloc[0])
                    if not avg_row.empty
                    else concept_rows["matched_corr"].mean())
    coverage     = (concept_rows["matched_corr"] > 0.5).mean()
    return concept_pcc, avg_pcc, coverage


# ------------------------------------------------------------------
#  Gather data
# ------------------------------------------------------------------
concept_union, method_data = set(), {}
for m in METHODS:
    csv_path = locate_csv(m)
    concept_dict, avg, cov = read_metrics(csv_path)
    concept_union.update(concept_dict.keys())
    method_data[m] = (concept_dict, avg, cov)

rows = sorted(concept_union) + ["Average", "Coverage@0.5"]
table = pd.DataFrame(index=rows, columns=[LEGENDS[m] for m in METHODS])

for m in METHODS:
    legend = LEGENDS[m]
    concept_dict, avg, cov = method_data[m]
    for c, v in concept_dict.items():
        table.at[c, legend] = v
    table.at["Average", legend]       = avg
    table.at["Coverage@0.5", legend]  = cov

# ------------------------------------------------------------------
#  Bold the row‑wise maximum and format numbers
# ------------------------------------------------------------------
fmt_table = pd.DataFrame(index=table.index, columns=table.columns)

for idx, row in table.iterrows():
    numeric = row.dropna().astype(float)
    if numeric.empty:
        continue
    max_val = numeric.max()
    for col, val in row.items():
        if pd.isna(val):
            fmt_table.at[idx, col] = "--"
        else:
            txt = f"{val:.3f}"
            if np.isclose(val, max_val):
                txt = rf"\textbf{{{txt}}}"
            fmt_table.at[idx, col] = txt

latex_code = fmt_table.to_latex(
    na_rep="--",
    escape=False,                 # keep $…$ math and \textbf
    column_format="l" + "c"*len(METHODS),
)

print(latex_code)
