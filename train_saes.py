# train_sae.py
import torch
import glob
from torch.utils.data import IterableDataset, DataLoader
from dictionary_learning.trainers import StandardTrainer, TopKTrainer, PAnnealTrainer, GatedSAETrainer
from dictionary_learning import AutoEncoder
from dictionary_learning.training import trainSAE

# === StreamingActivationDataset: Streams one .pt file at a time ===
class StreamingActivationDataset(IterableDataset):
    def __init__(self, path_pattern, shuffle_each_chunk=True):
        self.paths = sorted(glob.glob(path_pattern))
        self.shuffle_each_chunk = shuffle_each_chunk

    def __iter__(self):
        for path in self.paths:
            print(f"📥 Streaming from: {path}")
            chunk = torch.load(path)  # shape: [N, 512]
            if self.shuffle_each_chunk:
                indices = torch.randperm(len(chunk))
                chunk = chunk[indices]
            for row in chunk:
                yield row

# === TensorBuffer: Feeds batches of activations into trainSAE ===
class TensorBuffer:
    def __init__(self, data: IterableDataset, out_batch_size: int = 8192, device: str = "cpu"):
        self.data_loader = DataLoader(
            data,
            batch_size=out_batch_size,
            drop_last=True,
            num_workers=0,  # You can increase if I/O is your bottleneck
        )
        self.device = device
        self.iterator = iter(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator).to(self.device)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            raise StopIteration

    def close(self):
        pass  # Optional compatibility with ActivationBuffer API

# === Load streaming dataset ===
dataset = StreamingActivationDataset("saved_activations/activations_*.pt")

# === Wrap in TensorBuffer for training ===
buffer = TensorBuffer(
    data=dataset,
    out_batch_size=16384,
    device="cuda",
)

# === SAE training configuration ===
trainer_cfg = {
    "trainer": StandardTrainer,
    "activation_dim": 512,
    "dict_size": 32768,
    "l1_penalty": 0.1,
    "lr": 1e-4,
    "steps": 120000,
    "resample_steps": 25000,
    "warmup_steps": 1000,
    "device": "cuda",
    "layer": -1,
    "lm_name": "model.gpt_neox.final_layer_norm"
}

# === Train Sparse Autoencoder ===
ae = trainSAE(
    data=buffer,
    trainer_configs=[trainer_cfg],
    steps=120000,
    save_steps=[20000, 40000, 60000, 80000, 100000, 120000],
    log_steps=10000,
    verbose=True,
    save_dir="models"
)
