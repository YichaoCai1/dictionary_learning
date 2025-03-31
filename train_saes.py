# train_sae.py
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning import AutoEncoder
from dictionary_learning.training import trainSAE

# === Custom Dataset for Precomputed Activations ===
class PrecomputedActivationDataset(Dataset):
    def __init__(self, path_pattern):
        paths = sorted(glob.glob(path_pattern))
        self.data = []
        for path in paths:
            print(f"Loading {path}")
            self.data.append(torch.load(path))  # Each file = [N, 512]
        self.data = torch.cat(self.data, dim=0)  # [Total, 512]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# === TensorBuffer: Iterable wrapper for training ===
class TensorBuffer:
    def __init__(self, data: Dataset, out_batch_size: int = 8192, device: str = "cpu"):
        self.data_loader = DataLoader(data, batch_size=out_batch_size, shuffle=True, drop_last=True)
        self.device = device
        self.iterator = iter(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator).to(self.device)
        except StopIteration:
            # Optional: reset for next epoch-like behavior
            self.iterator = iter(self.data_loader)
            raise StopIteration

    def close(self):
        pass  # API-compatible with ActivationBuffer

# === Load dataset ===
dataset = PrecomputedActivationDataset("data/activations_*.pt")
buffer = TensorBuffer(
    data=dataset,
    out_batch_size=16384,
    device="cuda:0"
)

# === SAE training config ===
trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "activation_dim": 512,
    "dict_size": 32768,
    "out_batch_size": 16384,
    "entropy": False,
    "io": "out",
    "sparsity_penalty": 0.1,
    "lr": 1e-4,
    "steps": 120000,
    "resample_steps": 25000,
    "ghost_threshold": None,
    "warmup_steps": 1000,
    "device": "cuda:0",
}

# === Train the SAE ===
ae = trainSAE(
    data=buffer,
    trainer_configs=[trainer_cfg],
    save_dir="models"
)
