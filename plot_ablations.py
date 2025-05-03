import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

# ------------------------------------------------------------------
#  Paths — edit if your folder names differ
# ------------------------------------------------------------------
RESULTS_DIR = "results"                         # where the CSVs live
METHODS = {
    "1e-3": "panneallora_1e-3_24k",
    "1e-2": "panneallora_1e-2_24k",
    "1e-1": "panneallora_1e-1_24k",
    "1"   : "panneallora_1_24k",
}
CHECK_STEPS = range(2_000, 20_001, 2_000)
# ------------------------------------------------------------------

def read_mse(csv_path: str) -> float:
    return pd.read_csv(csv_path)["l2_loss"].iloc[-1]

def read_pcc(csv_path: str):
    df = pd.read_csv(csv_path)
    avg   = df["matched_corr"].iloc[-1]
    cov50 = (df["matched_corr"].iloc[:-1] > 0.5).mean()
    return avg, cov50

records = []
for alpha, folder in METHODS.items():
    for step in CHECK_STEPS:
        ckpt = os.path.join(RESULTS_DIR, folder, f"ae_{step}")
        mse_csv  = os.path.join(ckpt, "l2_loss.csv")
        pcc_csv  = os.path.join(ckpt, "matched_summary_exp.csv")
        if not (os.path.exists(mse_csv) and os.path.exists(pcc_csv)):
            continue
        records.append(
            dict(alpha=alpha,
                 step=step,
                 mse=read_mse(mse_csv),
                 pcc_avg=read_pcc(pcc_csv)[0],
                 coverage=read_pcc(pcc_csv)[1])
        )

df = pd.DataFrame(records)

# ------------------------------------------------------------------
#  Plotting style
# ------------------------------------------------------------------
COLORS   = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # colour‑blind friendly
MARKERS  = ["o", "s", "^", "D"]                          # circle, square, tri, dia
STYLE_CY = cycler(color=COLORS, marker=MARKERS)

plt.style.use("seaborn-v0_8")
plt.rcParams.update({
    "axes.prop_cycle": STYLE_CY,
    "lines.linewidth": 1.4,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
})

def lineplot(metric: str, ylabel: str, fname: str, title: str):
    fig, ax = plt.subplots(figsize=(4, 3))
    for alpha in METHODS:
        sub = df[df["alpha"] == alpha]
        ax.plot(sub["step"], sub[metric], label=f"α={alpha}")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=.95, ncol=2)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {fname}")

lineplot("mse",      "MSE ↓",               "ablation_mse.pdf",
         "Reconstruction loss vs. α")

lineplot("pcc_avg",  r"PCC$_{\mathrm{avg}}$ ↑",
         "ablation_pcc_avg.pdf", "Average PCC vs. α")

lineplot("coverage", "Coverage @0.5 ↑",
         "ablation_coverage.pdf", "Coverage vs. α")
