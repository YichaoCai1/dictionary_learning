import torch as t
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import json
import importlib
import inspect
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
import random
import os

###############################################################
#                       Utility helpers                       #
###############################################################

def fix_all_seeds(seed: int = 42):
    """Make all relevant random generators deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    print(f"All seeds fixed to: {seed}")


def load_word_pairs(path: Path):
    """Read src–tgt word pairs from a txt file (space‐separated)."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append(tuple(parts))
            else:
                print(f"Skipping malformed line in {path.name}: {line.strip()}")
    return pairs


@t.no_grad()
def get_token_activation(model, tokenizer, word: str, device: str):
    """Return the final-token hidden state for *word*."""
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    output = model(encoded.input_ids, output_hidden_states=True, return_dict=True)
    hidden_states = output.hidden_states  # tuple(n_layers, batch, seq_len, d_model)
    seq_len = encoded.attention_mask.sum(dim=1).item()  # number of non-pad tokens
    last_token_index = seq_len - 1
    return hidden_states[-1][0, last_token_index, :].detach()  # (d_model,)


def build_classification_dataset(pairs, model, tokenizer, device):
    """Compute (X, y) for the logistic‐regression probe.

    Each pair contributes two samples labelled 0 (src) and 1 (tgt).
    """
    X, y = [], []
    for src, tgt in tqdm(pairs, desc="Building classification dataset", leave=False):
        X.append(get_token_activation(model, tokenizer, src, device).cpu().numpy())
        y.append(0)
        X.append(get_token_activation(model, tokenizer, tgt, device).cpu().numpy())
        y.append(1)
    return np.array(X), np.array(y)


@t.no_grad()
def evaluate(dictionary, model, tokenizer, pairs, device, clf):
    """Compute reconstruction and probe metrics for one dictionary."""
    out = defaultdict(float)
    z_all, logits_all = [], []

    for src, tgt in tqdm(pairs, desc="Evaluating pairs", leave=False):
        # Token activations
        x_src = get_token_activation(model, tokenizer, src, device)
        x_tgt = get_token_activation(model, tokenizer, tgt, device)

        # Dictionary codes (allow for methods that return tuple)
        z_src = dictionary.encode(x_src.unsqueeze(0))
        z_tgt = dictionary.encode(x_tgt.unsqueeze(0))
        if isinstance(z_src, tuple):
            z_src, z_tgt = z_src[0], z_tgt[0]
        z_src, z_tgt = z_src.squeeze(0).cpu().numpy(), z_tgt.squeeze(0).cpu().numpy()
        z_all.extend((z_src, z_tgt))

        # Probe logits
        logits_all.extend((clf.decision_function([x_src.cpu().numpy()])[0],
                           clf.decision_function([x_tgt.cpu().numpy()])[0]))

        # Reconstruction
        x_hat = dictionary(x_src.unsqueeze(0)).squeeze(0)
        l2_loss = t.linalg.norm(x_src - x_hat, dim=-1).item()
        l1_loss = x_hat.norm(p=1).item()
        cos_sim = t.nn.functional.cosine_similarity(x_hat, x_tgt, dim=0).item()
        l2_ratio = (x_hat.norm() / x_src.norm()).item()
        total_var = t.var(x_src)
        resid_var = t.var(x_src - x_hat)
        frac_var_expl = (1 - resid_var / total_var).item()
        bias = (x_hat.norm() ** 2 / (x_src * x_hat).sum()).item()

        out["l2_loss"] += l2_loss
        out["l1_loss"] += l1_loss
        out["cossim"] += cos_sim
        out["l2_ratio"] += l2_ratio
        out["frac_variance_explained"] += frac_var_expl
        out["relative_reconstruction_bias"] += bias

    n_pairs = len(pairs)
    for k in out:
        out[k] /= n_pairs

    return out, np.array(logits_all), np.stack(z_all)


def train_or_load_logistic(X, y, path: Path):
    """Train a logistic probe or load it if it already exists."""
    if path.exists():
        return joblib.load(path)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    joblib.dump(clf, path)
    return clf


###############################################################
#                    Dictionary‑loading logic                 #
###############################################################

def find_config_path(checkpoint_path: Path) -> Path:
    """Look for config.json one or two levels above *checkpoint_path*."""
    for level in (checkpoint_path.parent, checkpoint_path.parent.parent):
        cfg = level / "config.json"
        if cfg.exists():
            return cfg
    raise FileNotFoundError(f"config.json not found near {checkpoint_path}")


def load_autoencoder(checkpoint_path: Path, device: str):
    """Instantiate and load the dictionary auto‑encoder from *checkpoint_path*."""
    config_path = find_config_path(checkpoint_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)
    trainer_cfg = cfg["trainer"]

    activation_dim = trainer_cfg["activation_dim"]
    dict_size = trainer_cfg["dict_size"]
    dict_class_name = trainer_cfg.get("dict_class", "AutoEncoder")

    # Dynamically import the dictionary class
    try:
        from dictionary_learning import dictionary as dict_module  # main package location
        DictClass = getattr(dict_module, dict_class_name)
    except (ImportError, AttributeError):
        DictClass = None
    if DictClass is None:
        for mod_name in ["standard", "top_k", "batch_top_k", "p_anneal", "gated_anneal"]:
            try:
                mod = importlib.import_module(f"dictionary_learning.trainers.{mod_name}")
                if hasattr(mod, dict_class_name):
                    DictClass = getattr(mod, dict_class_name)
                    break
            except ImportError:
                continue
    if DictClass is None:
        raise ValueError(f"Could not find dictionary class '{dict_class_name}'")

    constructor_args = {
        "activation_dim": activation_dim,
        "dict_size": dict_size,
        "device": device,
        "k": trainer_cfg.get("k"),
        "l1_penalty": trainer_cfg.get("l1_penalty"),
        "initial_sparsity_penalty": trainer_cfg.get("initial_sparsity_penalty"),
    }
    constructor_args = {k: v for k, v in constructor_args.items() if v is not None and k in inspect.signature(DictClass.__init__).parameters}
    model = DictClass(**constructor_args).to(device)

    state_dict = t.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, dict_class_name

###############################################################
#                          Main loop                          #
###############################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate every checkpoint in a directory (e.g. …/checkpoints)")
    parser.add_argument("--pair_dir", type=str, required=True, help="Directory containing *.txt word‑pair files")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to the *checkpoints* directory that holds all .pt files")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/", help="Root directory for outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def process_checkpoint(checkpoint_path: Path, pair_dir: Path, output_root: Path, model, tokenizer, device):
    """Run full evaluation for *checkpoint_path* and save under output_root/<checkpoint_stem>."""
    layer_name = checkpoint_path.stem  # e.g. "ae_2000"
    output_dir = output_root / layer_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # sub‑directory only for probe pickles
    probe_dir = output_dir / "probes"
    probe_dir.mkdir(exist_ok=True)

    dictionary, _ = load_autoencoder(checkpoint_path, device)

    concept_names, all_logits, all_dict_feats, all_metrics = [], [], [], {}

    # Iterate over concept word‑pair files
    for pair_path in pair_dir.glob("*.txt"):
        name = pair_path.stem.replace("[", "").replace("]", "").replace(" - ", "_").replace(" ", "_").lower()
        concept_names.append(name)
        pairs = load_word_pairs(pair_path)

        # Build or load the probe (only depends on model, not on dictionary)
        X, y = build_classification_dataset(pairs, model, tokenizer, device)
        clf_path = probe_dir / f"{name}_logreg.pkl"
        clf = train_or_load_logistic(X, y, clf_path)

        # Evaluate
        avg_metrics, logits, dict_feats = evaluate(dictionary, model, tokenizer, pairs, device, clf)
        all_logits.append(logits)
        all_dict_feats.append(dict_feats)
        all_metrics[name] = {"avg_metrics": avg_metrics}

    ############################################
    #   Aggregate metrics and save artefacts    #
    ############################################
    # Save metric‑wise CSVs
    metrics_dict = defaultdict(dict)
    for concept, vals in all_metrics.items():
        for metric, score in vals["avg_metrics"].items():
            metrics_dict[metric][concept] = score

    for metric, concept_scores in metrics_dict.items():
        df = pd.DataFrame.from_dict(concept_scores, orient="index", columns=[metric])
        df.index.name = "concept"
        df.loc["average"] = df[metric].mean()
        df.to_csv(output_dir / f"{metric}.csv")

    # Probe‑dictionary correlations
    C = len(all_logits)  # number of concepts
    D = all_dict_feats[0].shape[1]
    corr_orig = np.zeros((C, D))
    corr_exp = np.zeros((C, D))

    for i in range(C):
        x = all_logits[i]
        for j in range(D):
            z = all_dict_feats[i][:, j]
            z_exp = np.exp(z)
            corr_orig[i, j] = np.corrcoef(x, z)[0, 1] if np.std(x) > 0 and np.std(z) > 0 else 0.0
            corr_exp[i, j] = np.corrcoef(x, z_exp)[0, 1] if np.std(x) > 0 and np.std(z_exp) > 0 else 0.0

    # Hungarian matching on original correlations
    row_ind, col_ind = linear_sum_assignment(-corr_orig)
    for i, dim in zip(row_ind, col_ind):
        all_metrics[concept_names[i]]["matched_dim"] = int(dim)
        all_metrics[concept_names[i]]["matched_corr"] = float(corr_orig[i, dim])

    # Hungarian matching on exp(z)
    row_ind_exp, col_ind_exp = linear_sum_assignment(-corr_exp)
    for i, dim in zip(row_ind_exp, col_ind_exp):
        all_metrics[concept_names[i]]["matched_dim_exp"] = int(dim)
        all_metrics[concept_names[i]]["matched_corr_exp"] = float(corr_exp[i, dim])

    # Save summaries
    def _save_summary(row_idx, col_idx, corr_mat, fname):
        df = pd.DataFrame({
            "concept": [concept_names[i] for i in row_idx],
            "matched_dim": [int(d) for d in col_idx],
            "matched_corr": [float(corr_mat[i, d]) for i, d in zip(row_idx, col_idx)],
        })
        df.loc[len(df.index)] = {"concept": "average", "matched_dim": np.nan, "matched_corr": df.matched_corr.mean()}
        df.to_csv(output_dir / fname, index=False)

    _save_summary(row_ind, col_ind, corr_orig, "matched_summary.csv")
    _save_summary(row_ind_exp, col_ind_exp, corr_exp, "matched_summary_exp.csv")

    # Save complete JSON (per checkpoint)
    with open(output_dir / "all_concepts_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"✔ Saved results for {checkpoint_path.name} to {output_dir}")


def main():
    args = parse_args()
    fix_all_seeds(args.seed)

    pair_dir = Path(args.pair_dir)
    dict_root = Path(args.dict_path).expanduser().resolve()  # e.g. …/checkpoints
    output_root = Path(os.path.join(args.output_dir, args.dict_path.split('/')[-3]))
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Language model (shared)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)

    # Gather all checkpoint files (*.pt|*.pth|*.bin) directly inside *dict_root*
    ckpt_exts = {".pt", ".pth", ".bin"}
    checkpoints = sorted([p for p in dict_root.iterdir() if p.suffix in ckpt_exts])
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found inside {dict_root}")

    print(f"Found {len(checkpoints)} checkpoint(s) – starting evaluation…\n")
    for ckpt in checkpoints:
        process_checkpoint(ckpt, pair_dir, output_root, model, tokenizer, args.device)


if __name__ == "__main__":
    main()
