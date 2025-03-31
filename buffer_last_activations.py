import torch as t
from nnsight import LanguageModel
from datasets import load_dataset
from tqdm import tqdm
import os

# === Config ===
device = "cuda:0"
model_name = "EleutherAI/pythia-70m-deduped"
ctx_len = 128
batch_size = 64
activation_dim = 512
chunk_size = 100_000  # save every 100k activations
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

# === Load model and submodule ===
model = LanguageModel(model_name, device_map=device)
submodule = model.gpt_neox.final_layer_norm
tokenizer = model.tokenizer

# === Load dataset (streaming mode) ===
dataset = load_dataset(
    "path/to/the_pile_depulicated",  # ← replace with your actual path
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
            break

        tokens = tokenizer(
            texts,
            return_tensors='pt',
            max_length=ctx_len,
            padding=True,
            truncation=True
        ).to(device)

        with model.trace(tokens, invoker_args={"max_length": ctx_len, "truncation": True}):
            reps = submodule.output.save()
            model.inputs.save()
            submodule.output.stop()

        reps = reps.value
        if isinstance(reps, tuple):
            reps = reps[0]

        attn_mask = model.inputs.value[1]["attention_mask"]
        reps = reps[attn_mask != 0]

        buffer.append(reps.cpu())
        total_activations += reps.shape[0]
        pbar.update(reps.shape[0])

        if sum([x.shape[0] for x in buffer]) >= chunk_size:
            chunk = t.cat(buffer, dim=0)
            t.save(chunk, os.path.join(save_dir, f"activations_{file_idx:05d}.pt"))
            buffer = []
            file_idx += 1

    if buffer:
        chunk = t.cat(buffer, dim=0)
        t.save(chunk, os.path.join(save_dir, f"activations_{file_idx:05d}.pt"))
        pbar.update(chunk.shape[0])

pbar.close()
print(f"✅ Done. Total activations: {total_activations}")
