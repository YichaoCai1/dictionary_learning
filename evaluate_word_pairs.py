#!/usr/bin/env python
"""Evaluate sparse-dictionary checkpoints once each, sharing globally trained
logistic‑regression probes.

Changes vs. original
--------------------
1. **Probes cached globally.**  Trained exactly once per concept.
2. **L₀ sparsity metric** replaces L₁ activation norm.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch as t
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fix_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def load_word_pairs(path: Path) -> list[Tuple[str, str]]:
    pairs = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append(tuple(parts))
    return pairs


def concept_name(path: Path) -> str:
    return (path.stem.replace("[", "").replace("]", "").replace(" - ", "_")
            .replace(" ", "_").lower())


@t.no_grad()
def token_activation(model, tokenizer, word: str, device: str):
    tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(word, return_tensors="pt", padding=True, truncation=True).to(device)
    out = model(enc.input_ids, output_hidden_states=True, return_dict=True)
    last = enc.attention_mask.sum(dim=1).item() - 1
    return out.hidden_states[-1][0, last].detach()


def build_probe_dataset(pairs, model, tok, device):
    X, y = [], []
    for s, t_ in pairs:
        X.append(token_activation(model, tok, s, device).cpu().numpy()); y.append(0)
        X.append(token_activation(model, tok, t_, device).cpu().numpy()); y.append(1)
    return np.array(X), np.array(y)


def train_or_load_probe(X, y, path: Path):
    if path.exists():
        return joblib.load(path)
    clf = LogisticRegression(max_iter=1_000).fit(X, y)
    joblib.dump(clf, path)
    return clf

# ---------------------------------------------------------------------------
# Probe preparation (run once)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Probe preparation (load‑if‑possible, train‑otherwise)
# ---------------------------------------------------------------------------

def prepare_probes(pair_dir: Path, cache_dir: Path, model, tok, device):
    """Return a mapping *concept → probe*.

    * If **cache_dir already exists** (and contains ``*.pkl`` files), we assume
      all probes have been trained previously and simply load every pickle in
      that directory—**no re‑training**.
    * If the directory does not exist (first run), we train one probe per
      concept file and save them under *cache_dir*.
    """
    probes: Dict[str, LogisticRegression] = {}

    if cache_dir.exists():                # ──► Re‑use cached probes ────────────
        print("Loading cached probes …")
        for pkl in cache_dir.glob("*.pkl"):
            probes[pkl.stem] = joblib.load(pkl)
        print(f"Loaded {len(probes)} probe(s) from {cache_dir}")
        return probes

    # ──► First run: train probes and cache them ------------------------------
    print("No cached probes found – training new probes …")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for pair_file in pair_dir.glob("*.txt"):
        concept = concept_name(pair_file)
        pairs   = load_word_pairs(pair_file)
        X, y    = build_probe_dataset(pairs, model, tok, device)
        probes[concept] = train_or_load_probe(X, y, cache_dir / f"{concept}.pkl")
    print(f"Trained {len(probes)} probe(s) and saved to {cache_dir}")
    return probes

# ---------------------------------------------------------------------------
# Dictionary loading helpers
# ---------------------------------------------------------------------------

def find_cfg_path(ckpt: Path): 
    for lvl in (ckpt.parent, ckpt.parent.parent):
        cfg = lvl / "config.json"
        if cfg.exists():
            return cfg
    raise FileNotFoundError


def load_autoencoder(ckpt: Path, device: str):
    cfg = json.load(find_cfg_path(ckpt).open())
    tr = cfg["trainer"]
    act_dim, size = tr["activation_dim"], tr["dict_size"]
    cls_name = tr.get("dict_class", "AutoEncoder")

    def _import():
        try:
            from dictionary_learning import dictionary as dm
            return getattr(dm, cls_name)
        except (ImportError, AttributeError):
            pass
        for m in ["standard", "top_k", "batch_top_k", "p_anneal", "gated_anneal"]:
            try:
                mod = importlib.import_module(f"dictionary_learning.trainers.{m}")
                if hasattr(mod, cls_name):
                    return getattr(mod, cls_name)
            except ImportError:
                continue
        raise ValueError(f"No class {cls_name}")

    DictCls = _import()
    kwargs = {k: v for k, v in dict(k=tr.get("k"), l1_penalty=tr.get("l1_penalty"),
                                    initial_sparsity_penalty=tr.get("initial_sparsity_penalty"),
                                    activation_dim=act_dim, dict_size=size, device=device).items()
              if v is not None and k in inspect.signature(DictCls.__init__).parameters}
    model = DictCls(**kwargs).to(device)
    model.load_state_dict(t.load(ckpt, map_location=device))
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@t.no_grad()
def evaluate(dict_model, model, tok, pairs, device, probe):
    out = defaultdict(float)
    z_all, logits_all = [], []

    for s, t_ in pairs:
        # Get token activations
        xs = token_activation(model, tok, s, device)
        xt = token_activation(model, tok, t_, device)

        # Encode via dictionary
        zs = dict_model.encode(xs.unsqueeze(0))[0].squeeze(0).cpu().numpy()
        zt = dict_model.encode(xt.unsqueeze(0))[0].squeeze(0).cpu().numpy()
        z_all.extend((zs, zt))

        # Logit scores for probe
        logits_all.extend((
            probe.decision_function([xs.cpu().numpy()])[0],
            probe.decision_function([xt.cpu().numpy()])[0]
        ))

        # Reconstructions
        xhat_s = dict_model(xs.unsqueeze(0)).squeeze(0)
        xhat_t = dict_model(xt.unsqueeze(0)).squeeze(0)

        # L2 loss
        out["l2_loss"] += t.linalg.norm(xs - xhat_s).item()
        out["l2_loss"] += t.linalg.norm(xt - xhat_t).item()

        # L0 sparsity
        out["l0_loss"] += (np.abs(zs) > 0).sum() + (np.abs(zt) > 0).sum()

        # Cosine similarity (optional: average over both)
        out["cossim"] += t.nn.functional.cosine_similarity(xhat_s, xt, dim=0).item()
        out["cossim"] += t.nn.functional.cosine_similarity(xhat_t, xs, dim=0).item()

        # Fraction of variance explained
        for x, xhat in [(xs, xhat_s), (xt, xhat_t)]:
            residual_var = t.var(x - xhat)
            original_var = t.var(x)
            fve = 1 - (residual_var / original_var).item()
            out["frac_variance_explained"] += fve

    n = len(pairs)
    for k in out:
        out[k] /= (2 * n)  # Because we include both source and target words

    return out, np.array(logits_all), np.stack(z_all)

# ---------------------------------------------------------------------------
# Per‑checkpoint run
# ---------------------------------------------------------------------------

def run_checkpoint(ckpt: Path, *, pair_dir: Path, out_root: Path, probes, model, tok, device):
    layer = ckpt.stem; out_dir = out_root / layer; out_dir.mkdir(parents=True, exist_ok=True)
    dmodel = load_autoencoder(ckpt, device)

    names, logits_list, feats_list, metrics = [], [], [], {}
    for p in pair_dir.glob("*.txt"):
        name = concept_name(p); names.append(name); pairs = load_word_pairs(p)
        m, logits, feats = evaluate(dmodel, model, tok, pairs, device, probes[name])
        logits_list.append(logits); feats_list.append(feats); metrics[name] = {"avg_metrics": m}

    # aggregate metrics
    aggr = defaultdict(dict)
    for c, v in metrics.items():
        for m, s in v["avg_metrics"].items():
            aggr[m][c] = s
    for m, d in aggr.items():
        df = pd.DataFrame.from_dict(d, orient="index", columns=[m]); df.index.name="concept"; df.loc["average"] = df[m].mean(); df.to_csv(out_dir/f"{m}.csv")

    # corr + hungarian
    C, D = len(logits_list), feats_list[0].shape[1]
    corr = np.zeros((C,D)); corr_exp = np.zeros_like(corr)
    for i in range(C):
        x = logits_list[i]
        for j in range(D):
            z = feats_list[i][:,j]; zexp = np.exp(z)
            corr[i,j] = np.corrcoef(x,z)[0,1] if np.std(x)>0 and np.std(z)>0 else 0
            corr_exp[i,j] = np.corrcoef(x,zexp)[0,1] if np.std(x)>0 and np.std(zexp)>0 else 0
    for mat, tag in [(corr,""),(corr_exp,"_exp")]:
        r,c = linear_sum_assignment(-mat)
        df = pd.DataFrame({"concept":[names[i] for i in r],"matched_dim":c,"matched_corr":[mat[i,d] for i,d in zip(r,c)]})
        df.loc[len(df)] = {"concept":"average","matched_dim":np.nan,"matched_corr":df.matched_corr.mean()}
        df.to_csv(out_dir / f"matched_summary{tag}.csv", index=False)
        for i,d in zip(r,c):
            metrics[names[i]][f"matched_dim{tag}"] = int(d); metrics[names[i]][f"matched_corr{tag}"] = float(mat[i,d])

    with (out_dir/"all_concepts_metrics.json").open("w") as f: json.dump(metrics,f,indent=2)
    print(f"✓ {ckpt.name} done → {out_dir}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_dir", required=True)
    ap.add_argument("--dict_path", required=True)
    ap.add_argument("--output_dir", default="results/")
    ap.add_argument("--model_name", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args(); fix_all_seeds(args.seed)
    pair_dir = Path(args.pair_dir).resolve(); ckpt_root = Path(args.dict_path).resolve()
    out_root = Path(args.output_dir).expanduser(); out_root.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name); lm = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    probes = prepare_probes(pair_dir, out_root/"probes", lm, tok, args.device)
    ckpts = sorted(p for p in ckpt_root.iterdir() if p.suffix in {".pt",".pth",".bin"})
    if not ckpts: raise FileNotFoundError("No checkpoints found")
    
    out_path = Path(os.path.join(out_root, args.dict_path.split('/')[-3])).expanduser(); out_path.mkdir(exist_ok=True)
    for c in ckpts: run_checkpoint(c, pair_dir=pair_dir, out_root=out_path, probes=probes, model=lm, tok=tok, device=args.device)

if __name__ == "__main__": 
    main()