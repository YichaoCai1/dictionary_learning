import torch as t
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import json
import importlib
import inspect
from collections import defaultdict

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
    hidden_states = output.hidden_states  # List of [batch, seq_len, hidden_dim]
    seq_len = attention_mask.sum(dim=1).item()
    last_token_index = seq_len - 1

    return hidden_states[-1][0, last_token_index, :].detach()


@t.no_grad()
def evaluate(dictionary, model, tokenizer, pairs, device, batch_size=128):
    results = []
    out = defaultdict(float)
    
    for src, tgt in tqdm(pairs, desc="Evaluating pairs", leave=False):
        try:
            # Get token activations for the source and target words
            x = get_token_activation(model, tokenizer, src, device)
            y = get_token_activation(model, tokenizer, tgt, device)

            # Check for zero vectors before computing similarities
            if t.norm(x) == 0 or t.norm(y) == 0:
                print(f"Skipping due to zero vector norm: {src}, {tgt}")
                continue

            # Pass the activations through the dictionary (autoencoder)
            x_hat = dictionary(x.unsqueeze(0))[0].squeeze(0)

            # Compute L2 Loss
            l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()

            # Compute L1 Loss
            l1_loss = x_hat.norm(p=1, dim=-1).mean()

            # Compute Cosine Similarity
            cos_sim = t.nn.functional.cosine_similarity(x_hat, y, dim=0).item()

            # Compute the L2 Ratio
            l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean()

            # Compute variance explained
            total_variance = t.var(x, dim=0).sum()
            residual_variance = t.var(x - x_hat, dim=0).sum()
            frac_variance_explained = (1 - residual_variance / total_variance)

            # Relative Reconstruction Bias
            x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2)**2
            x_dot_x_hat = (x * x_hat).sum(dim=-1)
            relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()

            # Store the metrics in the results dictionary
            results.append({
                "source": src,
                "target": tgt,
                "cos_sim": cos_sim,
                "l2_loss": l2_loss.item(),
                "l1_loss": l1_loss.item(),
                "l2_ratio": l2_ratio.item(),
                "frac_variance_explained": frac_variance_explained.item(),
                "relative_reconstruction_bias": relative_reconstruction_bias.item()
            })

            # Accumulate metrics for averaging later
            out["l2_loss"] += l2_loss.item()
            out["l1_loss"] += l1_loss.item()
            out["cossim"] += cos_sim
            out["l2_ratio"] += l2_ratio.item()
            out["frac_variance_explained"] += frac_variance_explained.item()
            out['relative_reconstruction_bias'] += relative_reconstruction_bias.item()

        except Exception as e:
            print(f"Skipping ({src}, {tgt}): {e}")

    # Average the results over the number of pairs evaluated
    n_results = len(results)
    for key in out:
        out[key] /= n_results

    return results, out


def print_summary(name, results, out):
    if not results:
        print(f"No valid results for {name}")
        return
    
    # Compute averages of the metrics
    cos_avg = sum(r["cos_sim"] for r in results) / len(results)
    l2_avg = sum(r["l2_loss"] for r in results) / len(results)
    l1_avg = sum(r["l1_loss"] for r in results) / len(results)
    l2_ratio_avg = sum(r["l2_ratio"] for r in results) / len(results)
    variance_avg = sum(r["frac_variance_explained"] for r in results) / len(results)
    bias_avg = sum(r["relative_reconstruction_bias"] for r in results) / len(results)

    print(f"[{name}] Avg CosSim: {cos_avg:.4f}, Avg L2 Loss: {l2_avg:.4f}, Avg L1 Loss: {l1_avg:.4f}")
    print(f"Avg L2 Ratio: {l2_ratio_avg:.4f}, Avg Variance Explained: {variance_avg:.4f}, Avg Relative Reconstruction Bias: {bias_avg:.4f}")

    # Save average results to the file
    avg_results = {
        "cos_sim": cos_avg,
        "l2_loss": l2_avg,
        "l1_loss": l1_avg,
        "l2_ratio": l2_ratio_avg,
        "frac_variance_explained": variance_avg,
        "relative_reconstruction_bias": bias_avg
    }

    return avg_results


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
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    pair_dir = Path(args.pair_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)


    print(f"Loading word pair files from {pair_dir}...")
    pair_files = list(pair_dir.glob("*.txt"))
    assert pair_files, f"No .txt files found in {pair_dir}"

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)

    print(f"Loading dictionary from: {args.dict_path}")
    dictionary, _ = load_autoencoder(Path(args.dict_path), args.device)

    output_dir = output_root / args.dict_path.split('/')[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    for pair_path in pair_files:
        name = pair_path.stem.replace("[", "").replace("]", "").replace(" - ", "_").replace(" ", "_").lower()
        output_path = output_dir / f"{name}.json"

        print(f"\nEvaluating {pair_path.name}...")
        pairs = load_word_pairs(pair_path)
        results, out = evaluate(dictionary, model, tokenizer, pairs, args.device)

        # Print and save the summary
        avg_results = print_summary(pair_path.name, results, out)
        
        # Save individual results and averages to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "results": results,
                "average_results": avg_results,
                "summary": out
            }, f, indent=2)
        
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

"""
Cosine Similarity (cos_sim): Measures the cosine of the angle between the source word's reconstructed activation and the target word's activation.

L2 Loss (l2_loss): Measures the Euclidean distance between the reconstructed and original activation vectors.

L1 Loss (l1_loss): Measures the absolute difference between the reconstructed and original activation vectors.

L2 Ratio (l2_ratio): Measures the ratio of the norms (lengths) of the reconstructed and original activation vectors.

Variance Explained (frac_variance_explained): Measures how much variance in the original activation is captured by the reconstruction.

Relative Reconstruction Bias (relative_reconstruction_bias): Measures how much bias exists in the reconstruction compared to the original.
"""