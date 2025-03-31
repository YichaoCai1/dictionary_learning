import torch as t
from nnsight import LanguageModel
from datasets import load_dataset
from tqdm import tqdm
import os

# === Config ===
model_name = "EleutherAI/pythia-70m-deduped"
ctx_len = 128
batch_size = 64
activation_dim = 512
chunk_size = 100_000  # save every 100k activations
save_dir = "saved_activations"
os.makedirs(save_dir, exist_ok=True)

# === Auto-detect and use all GPUs ===
device = "cuda" if t.cuda.is_available() else "cpu"
device_map = "auto" if t.cuda.device_count() > 1 else {"": 0}

# === Load model and submodule ===
print("Loading model...")
model = LanguageModel(model_name, device_map=device_map)
submodule = model.gpt_neox.final_layer_norm
tokenizer = model.tokenizer

# === Load dataset (streaming) ===
dataset = load_dataset(
    "/lts/ycai/Projects/Datasets/the_pile_deduplicated",
    split="train",
    streaming=True
)

# === Text generator ===
def text_generator():
    for example in dataset:
        yield example["text"]

gen = text_generator()
buffer = []
file_idx = 0
total_activations = 0

# === Inference loop ===
with t.no_grad():
    pbar = tqdm(desc="Total activations saved")
    while True:
        try:
            texts = [next(gen) for _ in range(batch_size)]
        except StopIteration:
            break  # End of dataset

        # Tokenize inputs
        tokens = tokenizer(
            texts,
            return_tensors='pt',
            max_length=ctx_len,
            padding=True,
            truncation=True
        ).to(device)

        # Trace and extract activations
        with model.trace() as tracer:
            reps = submodule.output.save()
            model.inputs.save()

            _ = model(tokens, max_length=ctx_len, truncation=True)

            reps = reps.value
            input_data = model.inputs.value
            attn_mask = input_data[1]["attention_mask"]

            submodule.output.stop()

        if isinstance(reps, tuple):
            reps = reps[0]

        # Remove padded tokens
        reps = reps[attn_mask != 0]

        buffer.append(reps.cpu())
        total_activations += reps.shape[0]
        pbar.update(reps.shape[0])

        # Save when full
        if sum(x.shape[0] for x in buffer) >= chunk_size:
            chunk = t.cat(buffer, dim=0)
            t.save(chunk, os.path.join(save_dir, f"activations_{file_idx:05d}.pt"))
            buffer = []
            file_idx += 1

    # Final flush
    if buffer:
        chunk = t.cat(buffer, dim=0)
        t.save(chunk, os.path.join(save_dir, f"activations_{file_idx:05d}.pt"))
        pbar.update(chunk.shape[0])

pbar.close()
print(f"✅ Done. Total activations saved: {total_activations}")
