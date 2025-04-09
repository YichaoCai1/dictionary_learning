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

# === Helper function to get activations for batch ===
def get_activations_from_batch(text_batch):
    # === Use original working logic like in ActivationBuffer ===
    with t.no_grad():
        # Trace through the model for the batch
        with model.trace(text_batch, invoker_args={"truncation": True, "max_length": ctx_len}):
            hidden_states = submodule.output.save()
            input = model.inputs.save()
            submodule.output.stop()

        attn_mask = input.value[1]["attention_mask"]
        hidden_states = hidden_states.value
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # Extract activations for the last token in each sequence of the batch
        batch_activations = []
        for i in range(attn_mask.shape[0]):  # iterate over batch size
            seq_len = attn_mask[i].sum().item()
            last_token_index = seq_len - 1

            # Get the activation for the last token of this sequence
            batch_activations.append(hidden_states[i, last_token_index, :])

        return t.stack(batch_activations, dim=0).contiguous().cpu()  # Stack activations into a batch tensor

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
