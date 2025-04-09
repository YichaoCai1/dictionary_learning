import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def trace_final_token_embeddings(model, tokenizer, words, device):
    """
    For each word, trace the final token embedding using model hidden states.
    Returns a dict: {'last': {word: activation vector or error}}, and a dict of skipped words with errors.
    """
    tokenizer.pad_token = tokenizer.eos_token
    results = {"last": {}}
    skipped = {}

    for word in tqdm(words, desc="Tracing words"):
        try:
            # Tokenize single word as a batch of size 1
            encoded = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)

            # Forward pass with hidden states
            with torch.no_grad():
                output = model(input_ids, output_hidden_states=True, return_dict=True)
                hidden_states = output.hidden_states  # List of [batch, seq_len, hidden_dim]

            # Find the last token index for the word
            seq_len = attention_mask.sum(dim=1).item()
            last_token_index = seq_len - 1

            # Extract from last layer only
            vec = hidden_states[-1][0, last_token_index, :].cpu().tolist()
            results["last"][word] = vec

        except Exception as e:
            skipped[word] = str(e)

    return results, skipped


def load_words_from_directory(pair_dir):
    """
    Load all words from all txt files in a directory containing tab-separated word pairs.
    Returns a deduplicated list of words.
    """
    all_words = set()
    for file in os.listdir(pair_dir):
        if file.endswith(".txt"):
            with open(os.path.join(pair_dir, file), "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        all_words.update(parts)
    return sorted(list(all_words))


def main(pair_dir, model_name, output_json, device):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    words = load_words_from_directory(pair_dir)
    print(f"Loaded {len(words)} unique words.")

    traced, skipped = trace_final_token_embeddings(model, tokenizer, words, device)

    print("\nSummary:")
    for layer in traced:
        print(f"Layer {layer}: {len(traced[layer])} words")
    print(f"Skipped: {len(skipped)} words")
    if skipped:
        print("Example skipped word:")
        k = list(skipped.keys())[0]
        print(f"  {k}: {skipped[k]}")

    # Save
    with open(output_json, "w") as f:
        json.dump({"traced": traced, "skipped": skipped}, f, indent=2)
    print(f"Saved results to: {output_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_dir", type=str, required=True, help="Directory with word pair txt files")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--output_json", type=str, default="layer_activation_coverage.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args.pair_dir, args.model_name, args.output_json, args.device)