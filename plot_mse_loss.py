import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler   # easy colour cycling

# ------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------
BASE_DIR = "results"      # root directory holding all method folders

METHODS = [
    "topk_24k",
    "batchtopk_24k",
    "panneal_24k",
    "panneallora_1e-1_24k",
]

LEGENDS = {
    "topk_24k":              r"top‑$k$",
    "batchtopk_24k":         r"batch‑top‑$k$",
    "panneal_24k":           r"$p$‑annealing",
    "panneallora_1e-1_24k":  r"$p$‑annealing‑LoRa",
}

# colour‑blind–friendly palette + distinctive markers
COLOURS  = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MARKERS  = ["o", "s", "^", "D"]  # circle, square, triangle, diamond
# ------------------------------------------------------------------

def extract_step(dirname: str) -> int:
    """Return integer step from directory name 'ae_<step>'."""
    m = re.search(r"ae_(\d+)$", dirname)
    if not m:
        raise ValueError(f"{dirname} does not match 'ae_<step>'")
    return int(m.group(1))

# ------------------------------------------------------------------
#  Matplotlib styling
# ------------------------------------------------------------------
plt.style.use("seaborn-v0_8")
plt.rcParams.update({
    "figure.dpi"      : 110,
    "axes.facecolor"  : "#f7f7f7",
    "axes.grid"       : True,
    "grid.linestyle"  : "--",
    "grid.alpha"      : 0.6,
    "axes.prop_cycle" : cycler(color=COLOURS),
    "axes.titlesize"  : 14,
    "axes.labelsize"  : 12,
    "legend.framealpha": 0.0,
    "lines.linewidth" : 2.0,
    "lines.markersize": 6,
})

fig, ax = plt.subplots(figsize=(7, 4))

for method, colour, marker in zip(METHODS, COLOURS, MARKERS):
    method_dir = os.path.join(BASE_DIR, method)

    # gather checkpoint directories ae_<step>
    step_dirs = sorted(
        [d for d in os.listdir(method_dir) if re.match(r"ae_\d+$", d)],
        key=extract_step,
    )

    steps, mse_vals = [], []

    for sd in step_dirs:
        step = extract_step(sd)
        csv_path = os.path.join(method_dir, sd, "l2_loss.csv")

        df = pd.read_csv(csv_path)

        # The last row already contains the averaged MSE
        mse = df["l2_loss"].iloc[-1] if "l2_loss" in df.columns else df.iloc[-1, 0]

        steps.append(step)
        mse_vals.append(mse)

    ax.plot(
        steps,
        mse_vals,
        linestyle="-",
        marker=marker,
        label=LEGENDS[method],
    )

ax.set_xlabel("Training step")
ax.set_ylabel("Average MSE loss")
ax.set_title("SAE feature‑reconstruction performance")
ax.legend(loc="best", ncol=2, handlelength=1.7)

fig.tight_layout()
fig.savefig("mse_vs_steps.pdf", bbox_inches="tight")  # save *before* showing
plt.show()
