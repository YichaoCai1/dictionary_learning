import torch as t
from nnsight import LanguageModel
from datasets import load_dataset
from tqdm import tqdm
import os

# === Config ===
# model_name = "EleutherAI/pythia-70m-deduped"
# save_dir = "saved_activations_70m"

model_name = "EleutherAI/pythia-1b-deduped"
save_dir = "saved_activations_1b"

# model_name = "EleutherAI/pythia-6.9b-deduped"
# save_dir = "saved_activations_6.9b"

ctx_len = 128
batch_size = 64
activation_dim = 512
chunk_size = 100_000
os.makedirs(save_dir, exist_ok=True)

# === Device config ===
device = "cuda" if t.cuda.is_available() else "cpu"
device_map = "auto" if t.cuda.device_count() > 1 else {"": 0}

# === Load model ===
print("Loading model...")
model = LanguageModel(model_name, device_map=device_map)
submodule = model.gpt_neox.final_layer_norm

# === Load dataset ===
dataset_path = "/lts/ycai/Projects/Datasets/the_pile_deduplicated"
print(f"Loading dataset from: {dataset_path}")
dataset = load_dataset(dataset_path, split="train", streaming=True)

def text_generator():
    for example in dataset:
        yield example["text"]

gen = text_generator()
buffer = []
file_idx = 0
total_activations = 0

# === Helper function to get activations ===
def get_activations_from_batch(text_batch):
    with t.no_grad():
        with model.trace(text_batch, invoker_args={"truncation": True, "max_length": ctx_len}):
            hidden_states = submodule.output.save()
            input = model.inputs.save()
            submodule.output.stop()

        attn_mask = input.value[1]["attention_mask"]
        hidden_states = hidden_states.value
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # ✅ Move attention mask to the same device as hidden_states
        attn_mask = attn_mask.to(hidden_states.device)

        hidden_states = hidden_states[attn_mask != 0]
        return hidden_states.contiguous().cpu()
    
# === Main loop ===
print("Beginning activation extraction...")
pbar = tqdm(desc="Total activations saved")
while True:
    try:
        texts = [next(gen) for _ in range(batch_size)]
    except StopIteration:
        break

    reps = get_activations_from_batch(texts)
    if reps.shape[0] == 0:
        continue

    buffer.append(reps)
    total_activations += reps.shape[0]
    pbar.update(reps.shape[0])

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
