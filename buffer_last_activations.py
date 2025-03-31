# save_activations.py
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
chunk_size = 100_000
save_dir = "saved_activations"
os.makedirs(save_dir, exist_ok=True)

# === Multi-GPU Support ===
device = "cuda" if t.cuda.is_available() else "cpu"
device_map = "auto" if t.cuda.device_count() > 1 else {"": 0}

# === Load Model ===
print("Loading model...")
model = LanguageModel(model_name, device_map=device_map)
submodule = model.gpt_neox.final_layer_norm
tokenizer = model.tokenizer

# === Load Your Dataset ===
dataset_path = "/lts/ycai/Projects/Datasets/the_pile_deduplicated"

print(f"Loading dataset from: {dataset_path}")
dataset = load_dataset(
    dataset_path,
    split="train",
    streaming=True
)

# Peek a few samples to verify loading
print("✅ Sample data:")
for i, example in enumerate(dataset):
    print(f"[{i}] {example['text'][:80]}...")
    if i >= 2: break

# === Create Text Generator ===
def text_generator():
    for example in dataset:
        yield example["text"]

gen = text_generator()
buffer = []
file_idx = 0
total_activations = 0

# === Main Loop ===
with t.no_grad():
    pbar = tqdm(desc="Total activations saved")

    while True:
        try:
            texts = [next(gen) for _ in range(batch_size)]
        except StopIteration:
            break

        tokens = tokenizer(
            texts,
            return_tensors='pt',
            max_length=ctx_len,
            padding=True,
            truncation=True
        ).to(device)

        # Trace and extract
        with model.trace() as tracer:
            hook = submodule.output.save_hook()
            _ = model(**tokens)

        reps = hook.value
        attn_mask = tokens["attention_mask"]

        if isinstance(reps, tuple):
            reps = reps[0]

        reps = reps[attn_mask != 0]

        if reps.shape[0] == 0:
            continue

        buffer.append(reps.cpu())
        total_activations += reps.shape[0]
        pbar.update(reps.shape[0])

        if sum(x.shape[0] for x in buffer) >= chunk_size:
            chunk = t.cat(buffer, dim=0)
            t.save(chunk, os.path.join(save_dir, f"activations_{file_idx:05d}.pt"))
            buffer = []
            file_idx += 1

    if buffer:
        chunk = t.cat(buffer, dim=0)
        t.save(chunk, os.path.join(save_dir, f"activations_{file_idx:05d}.pt"))
        pbar.update(chunk.shape[0])

pbar.close()
print(f"✅ Done. Total activations saved: {total_activations}")
