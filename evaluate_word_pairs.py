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

def fix_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    print(f"All seeds fixed to: {seed}")

def load_word_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        pairs = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append(tuple(parts))
            else:
                print(f"Skipping malformed line in {path.name}: {line.strip()}")
        return pairs

@t.no_grad()
def get_token_activation(model, tokenizer, word, device):
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    output = model(input_ids, output_hidden_states=True, return_dict=True)
    hidden_states = output.hidden_states
    seq_len = attention_mask.sum(dim=1).item()
    last_token_index = seq_len - 1
    return hidden_states[-1][0, last_token_index, :].detach()

def build_classification_dataset(pairs, model, tokenizer, device):
    X, y = [], []
    for src, tgt in tqdm(pairs, desc="Building classification dataset", leave=False):
        src_vec = get_token_activation(model, tokenizer, src, device)
        tgt_vec = get_token_activation(model, tokenizer, tgt, device)
        X.append(src_vec.cpu().numpy())
        y.append(0)
        X.append(tgt_vec.cpu().numpy())
        y.append(1)
    return np.array(X), np.array(y)

@t.no_grad()
def evaluate(dictionary, model, tokenizer, pairs, device, clf):
    out = defaultdict(float)
    z_all = []
    logits_all = []
    for src, tgt in tqdm(pairs, desc="Evaluating pairs", leave=False):
        x_src = get_token_activation(model, tokenizer, src, device)
        x_tgt = get_token_activation(model, tokenizer, tgt, device)

        z_src = dictionary.encode(x_src.unsqueeze(0)).squeeze(0).cpu().numpy()
        z_tgt = dictionary.encode(x_tgt.unsqueeze(0)).squeeze(0).cpu().numpy()

        logit_src = clf.decision_function([x_src.cpu().numpy()])[0]
        logit_tgt = clf.decision_function([x_tgt.cpu().numpy()])[0]

        z_all.append(z_src)
        z_all.append(z_tgt)
        logits_all.append(logit_src)
        logits_all.append(logit_tgt)

        x_hat = dictionary.decode(dictionary.encode(x_src.unsqueeze(0))).squeeze(0)
        l2_loss = t.linalg.norm(x_src - x_hat, dim=-1).item()
        l1_loss = x_hat.norm(p=1, dim=-1).item()
        cos_sim = t.nn.functional.cosine_similarity(x_hat, x_tgt, dim=0).item()
        l2_ratio = (x_hat.norm() / x_src.norm()).item()
        total_variance = t.var(x_src, dim=0).sum()
        residual_variance = t.var(x_src - x_hat, dim=0).sum()
        frac_variance_explained = (1 - residual_variance / total_variance).item()
        bias = (x_hat.norm()**2 / (x_src * x_hat).sum()).item()

        out["l2_loss"] += l2_loss
        out["l1_loss"] += l1_loss
        out["cossim"] += cos_sim
        out["l2_ratio"] += l2_ratio
        out["frac_variance_explained"] += frac_variance_explained
        out['relative_reconstruction_bias'] += bias

    z_all = np.stack(z_all)
    logits_all = np.array(logits_all)

    n = len(pairs)
    for key in out:
        out[key] /= n

    return out, logits_all, z_all

def train_or_load_logistic(X, y, path):
    if Path(path).exists():
        clf = joblib.load(path)
    else:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        joblib.dump(clf, path)
    return clf

def load_autoencoder(dict_path: Path, device: str):
    config_path = dict_path.parent / "config.json"
    assert config_path.exists(), f"config.json not found at {config_path}"
    with open(config_path, "r") as f:
        config = json.load(f)
    trainer_cfg = config["trainer"]
    activation_dim = trainer_cfg["activation_dim"]
    dict_size = trainer_cfg["dict_size"]
    dict_class_name = trainer_cfg.get("dict_class", "AutoEncoder")
    try:
        from dictionary_learning import dictionary
        DictClass = getattr(dictionary, dict_class_name)
    except (ImportError, AttributeError):
        DictClass = None
    if DictClass is None:
        trainer_modules = ["standard", "top_k", "batch_top_k", "p_anneal", "gated_anneal"]
        for module_name in trainer_modules:
            try:
                mod = importlib.import_module(f"dictionary_learning.trainers.{module_name}")
                if hasattr(mod, dict_class_name):
                    DictClass = getattr(mod, dict_class_name)
                    break
            except ImportError:
                continue
    if DictClass is None:
        raise ValueError(f"Could not find class '{dict_class_name}'")
    constructor_params = inspect.signature(DictClass.__init__).parameters
    constructor_args = {
        "activation_dim": activation_dim,
        "dict_size": dict_size,
        "device": device,
        "k": trainer_cfg.get("k"),
        "l1_penalty": trainer_cfg.get("l1_penalty"),
        "initial_sparsity_penalty": trainer_cfg.get("initial_sparsity_penalty"),
    }
    filtered_args = {k: v for k, v in constructor_args.items() if k in constructor_params and v is not None}
    model = DictClass(**filtered_args)
    state_dict = t.load(dict_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval(), dict_class_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_dir", type=str, required=True)
    parser.add_argument("--dict_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/70m")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    fix_all_seeds(args.seed)
    pair_dir = Path(args.pair_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    dictionary, _ = load_autoencoder(Path(args.dict_path), args.device)
    output_dir = output_root / args.dict_path.split('/')[-3]
    output_dir.mkdir(parents=True, exist_ok=True)

    concept_names = []
    all_logits = []
    all_dict_feats = []
    all_metrics = {}

    for pair_path in pair_dir.glob("*.txt"):
        name = pair_path.stem.replace("[", "").replace("]", "").replace(" - ", "_").replace(" ", "_").lower()
        concept_names.append(name)
        pairs = load_word_pairs(pair_path)
        X, y = build_classification_dataset(pairs, model, tokenizer, args.device)
        clf_path = output_dir / f"{name}_logreg.pkl"
        clf = train_or_load_logistic(X, y, clf_path)
        avg_metrics, logits, dict_feats = evaluate(dictionary, model, tokenizer, pairs, args.device, clf)
        all_logits.append(logits)
        all_dict_feats.append(dict_feats)
        all_metrics[name] = {"avg_metrics": avg_metrics}

    # Save metric-wise CSVs
    metrics_dict = defaultdict(dict)
    for concept, values in all_metrics.items():
        for metric, score in values["avg_metrics"].items():
            metrics_dict[metric][concept] = score

    for metric, concept_scores in metrics_dict.items():
        df = pd.DataFrame.from_dict(concept_scores, orient="index", columns=[metric])
        df.index.name = "concept"
        df.loc["average"] = df[metric].mean()
        df.to_csv(output_dir / f"{metric}.csv")

    # Pearson correlation (original & exp)
    C = len(all_logits)
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

    # Matching based on original
    cost_matrix = -corr_orig
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i, dim in zip(row_ind, col_ind):
        all_metrics[concept_names[i]]["matched_dim"] = int(dim)
        all_metrics[concept_names[i]]["matched_corr"] = -float(cost_matrix[i, dim])

    # Matching based on exp(z)
    cost_matrix_exp = -corr_exp
    row_ind_exp, col_ind_exp = linear_sum_assignment(cost_matrix_exp)
    for i, dim in zip(row_ind_exp, col_ind_exp):
        all_metrics[concept_names[i]]["matched_dim_exp"] = int(dim)
        all_metrics[concept_names[i]]["matched_corr_exp"] = -float(cost_matrix_exp[i, dim])

    # Save matched summaries with average row
    summary_orig = pd.DataFrame({
        "concept": [concept_names[i] for i in row_ind],
        "matched_dim": [int(dim) for dim in col_ind],
        "matched_corr": [-float(cost_matrix[i, dim]) for i, dim in zip(row_ind, col_ind)],
    })
    summary_orig.loc[len(summary_orig.index)] = {
        "concept": "average",
        "matched_dim": np.nan,
        "matched_corr": summary_orig["matched_corr"].mean()
    }
    summary_orig.to_csv(output_dir / "matched_summary.csv", index=False)

    summary_exp = pd.DataFrame({
        "concept": [concept_names[i] for i in row_ind_exp],
        "matched_dim_exp": [int(dim) for dim in col_ind_exp],
        "matched_corr_exp": [-float(cost_matrix_exp[i, dim]) for i, dim in zip(row_ind_exp, col_ind_exp)],
    })
    summary_exp.loc[len(summary_exp.index)] = {
        "concept": "average",
        "matched_dim_exp": np.nan,
        "matched_corr_exp": summary_exp["matched_corr_exp"].mean()
    }
    summary_exp.to_csv(output_dir / "matched_summary_exp.csv", index=False)

    # Save full JSON
    with open(output_dir / "all_concepts_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    main()
