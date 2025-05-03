#!/usr/bin/env python
"""
Evaluate classification performance of the frozen logistic probes
on each concept’s word-pair dataset, handling bracketed filenames.
"""

import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import importlib

# ---------------------- Configuration -----------------------
PAIR_DIR   = Path("word_pairs")      # directory of your *.txt word-pair files
PROBE_DIR  = Path("results/probes")  # where prepare_probes saved <concept>.pkl
DEVICE     = "cuda"                  # or "cpu"
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
# -----------------------------------------------------------

# ---- helper functions from evaluate_word_pairs.py ----
ev = importlib.import_module("evaluate_word_pairs")
load_word_pairs   = ev.load_word_pairs
prepare_probes    = ev.prepare_probes
token_activation  = ev.token_activation
# ------------------------------------------------------

# Helper to match concept slug to filename
def slugify(name: str) -> str:
    s = name.lower().strip("[]")
    s = s.replace('+', '_+_')
    s = re.sub(r'[^a-z0-9+]+', '_', s)
    s = re.sub(r'__+', '_', s)
    return s.strip('_')

# Build mapping: slug → .txt path
FILE_BY_CONCEPT = {
    slugify(p.stem): p
    for p in PAIR_DIR.glob("*.txt")
}

# 1) Load LM + probes
print("Loading model and probes …")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
lm  = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
probes = prepare_probes(PAIR_DIR, PROBE_DIR, lm, tok, DEVICE)
print(f"Loaded {len(probes)} probes.\n")

# 2) Evaluate each concept
results = []
for concept, clf in probes.items():
    txt_path = FILE_BY_CONCEPT.get(concept)
    if txt_path is None or not txt_path.exists():
        print(f"⚠ Skipping '{concept}': no file found.")
        continue

    pairs = load_word_pairs(txt_path)
    X, y = [], []
    for neg, pos in pairs:
        xn = token_activation(lm, tok, neg, DEVICE).cpu().numpy()
        xp = token_activation(lm, tok, pos, DEVICE).cpu().numpy()
        X.append(xn); y.append(0)
        X.append(xp); y.append(1)
    X = np.stack(X)
    y = np.array(y)

    y_pred = clf.predict(X)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X)[:, 1]
    else:
        y_score = clf.decision_function(X)

    results.append({
        "Concept": concept,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_score),
    })

# 3) Summarize
df = pd.DataFrame(results).set_index("Concept")
print("=== Classification Performance per Concept ===\n")
print(df)

# # 4) Detailed reports (optional)
# print("\n=== Detailed Classification Reports ===\n")
# for concept, clf in probes.items():
#     txt_path = FILE_BY_CONCEPT.get(concept)
#     if txt_path is None:
#         continue
#     pairs = load_word_pairs(txt_path)
#     X, y = [], []
#     for neg, pos in pairs:
#         X.append(token_activation(lm, tok, neg, DEVICE).cpu().numpy())
#         X.append(token_activation(lm, tok, pos, DEVICE).cpu().numpy())
#         y.extend([0,1])
#     X = np.stack(X); y = np.array(y)
#     y_pred = clf.predict(X)
#     print(f"-- {concept} --")
#     print(classification_report(y, y_pred, digits=3))

# 5) LaTeX table
print("\n=== LaTeX Table ===\n")
print(df.to_latex(
    float_format="%.3f",
    caption="Classification performance of logistic probes for each concept",
    label="tab:probe_perf",
    escape=False
))
