#!/usr/bin/env python
"""
Generate “trace” plots that overlay
  • probe logits, and
  • the exponentiated matched‑dimension activations
for every SAE variant and every concept.

Output:
    trace_plots/trace_<concept>.pdf   (publication quality)
    trace_plots/trace_<concept>.png   (quick preview)
"""

# ------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------
PAIR_DIR    = "word_pairs"
PROBE_DIR   = "results/probes"
RESULTS_DIR = "results"
MODEL_ROOT  = "models"
OUT_DIR     = "trace_plots"
DEVICE      = "cuda"

METHODS = [
    "topk_24k",
    "batchtopk_24k",
    "panneal_24k",
    "panneallora_1e-1_24k",
]

def ckpt_path(method: str, step: int = 20_000) -> str:
    return f"{MODEL_ROOT}/{method}/trainer_0/checkpoints/ae_{step:05d}.pt"

CKPTS = {m: ckpt_path(m) for m in METHODS}
# ------------------------------------------------------------------


# ------------------------------------------------------------------
#  Imports
# ------------------------------------------------------------------
import importlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cycler import cycler
from transformers import AutoModelForCausalLM, AutoTokenizer

ev = importlib.import_module("evaluate_word_pairs")
load_word_pairs   = ev.load_word_pairs
prepare_probes    = ev.prepare_probes
token_activation  = ev.token_activation
load_autoencoder  = ev.load_autoencoder
# ------------------------------------------------------------------


# ------------------------------------------------------------------
#  Helper: filename → concept slug
# ------------------------------------------------------------------
def slugify(name: str) -> str:
    """
    '[verb - V + able]' → 'verb_v_+_able'
    Keeps '+', collapses other delimiters to '_'.
    """
    s = name.lower().strip("[]")
    s = s.replace('+', '_+_')
    s = re.sub(r'[^a-z0-9+]+', '_', s)
    s = re.sub(r'__+', '_', s)
    return s.strip('_')

FILE_BY_CONCEPT = {slugify(p.stem): p for p in Path(PAIR_DIR).glob("*.txt")}
# ------------------------------------------------------------------


# ------------------------------------------------------------------
#  1. Load LM + probes
# ------------------------------------------------------------------
print("Loading probes …")
model_name = "EleutherAI/pythia-70m-deduped"
tok   = AutoTokenizer.from_pretrained(model_name)
lm    = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
probes = prepare_probes(Path(PAIR_DIR), Path(PROBE_DIR), lm, tok, DEVICE)
print(f"Loaded {len(probes)} probes")

# ------------------------------------------------------------------
#  2. Read matched dimensions
# ------------------------------------------------------------------
matched_dim = {}
for m in METHODS:
    csv = Path(RESULTS_DIR) / m / "ae_20000" / "matched_summary_exp.csv"
    df  = pd.read_csv(csv)
    for _, r in df.iterrows():
        slug = r["concept"].lower()
        matched_dim.setdefault(slug, {})[m] = (
            int(r["matched_dim"]) if not pd.isna(r["matched_dim"]) else None
        )

concepts = sorted([c for c in matched_dim if c != "average"])

# ------------------------------------------------------------------
#  3. Load SAEs
# ------------------------------------------------------------------
def load_saes():
    out = {}
    for m, p in CKPTS.items():
        sae = load_autoencoder(Path(p), DEVICE)
        sae.eval()
        out[m] = sae
        print(f"✓ SAE loaded: {m}")
    return out

sae_models = load_saes()

# ------------------------------------------------------------------
#  4. Plot style
# ------------------------------------------------------------------
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
SAE_STYLE = dict(zip(METHODS, cycler(color=PALETTE)))
PROBE_STYLE = dict(color="black", linewidth=1, label="probe logit")

Path(OUT_DIR).mkdir(exist_ok=True)
plt.style.use("seaborn-v0_8")
plt.rcParams.update({"axes.grid": True, "grid.linestyle": ":", "grid.alpha": .6})

# ------------------------------------------------------------------
#  5. Trace plots
# ------------------------------------------------------------------
for slug in concepts:
    fp = FILE_BY_CONCEPT.get(slug)
    if fp is None:
        print(f"⚠ Missing word‑pair file for '{slug}', skipped.")
        continue

    pairs = load_word_pairs(fp)
    N = len(pairs)

    # full token set (neg1,pos1, …, negN,posN)
    X_list = []
    for neg, pos in pairs:
        X_list.append(token_activation(lm, tok, neg, DEVICE).cpu().numpy())
        X_list.append(token_activation(lm, tok, pos, DEVICE).cpu().numpy())
    X = np.stack(X_list)                      # (2N, 512)

    logits = probes[slug].decision_function(X)

    activ = {}
    for m, sae in sae_models.items():
        dim = matched_dim[slug].get(m)
        if dim is None:
            continue
        z_out = sae.encode(torch.tensor(X, device=DEVICE))
        if isinstance(z_out, (tuple, list)):      # LoRa returns tuple
            z_out = z_out[0]
        z_exp = torch.exp(z_out)                  # exponentiate
        activ[m] = z_exp.detach().cpu().numpy()[:, dim]  # (2N,)

    # -------- plot --------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    idx = np.arange(1, 2 * N + 1)
    ax.plot(idx, logits, **PROBE_STYLE)

    for m, y in activ.items():
        ax.plot(idx, y, label=m, linewidth=1, **SAE_STYLE[m])

    ax.axvline(N + 0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(slug.replace('_', ' '))
    ax.set_xlabel("Token index (negatives → positives)")
    ax.set_ylabel("Activation / logit")
    ax.legend(fontsize=8, framealpha=.96)

    fig.tight_layout()
    fig.savefig(Path(OUT_DIR) / f"trace_{slug}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Trace saved: {slug}")

print(f"\n✓ All plots written to '{OUT_DIR}/'")
