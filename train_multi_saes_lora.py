# train_multi_sae.py

import torch
import glob
import random
import threading
import queue
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from dictionary_learning.trainers import PAnnealTrainerLoRa
from dictionary_learning import AutoEncoder
from dictionary_learning.training import trainSAE

# === Asynchronous Streaming Dataset ===
class AsyncStreamingDataset(IterableDataset):
    def __init__(
        self,
        path_pattern,
        prefetch_depth=4,
        shuffle_files=True,
        shuffle_each_chunk=True,
        device="cpu",
    ):
        self.paths = sorted(glob.glob(path_pattern))
        self.prefetch_depth = prefetch_depth
        self.shuffle_files = shuffle_files
        self.shuffle_each_chunk = shuffle_each_chunk
        self.device = device

    def __iter__(self):
        file_queue = queue.Queue(maxsize=self.prefetch_depth)
        stop_token = object()

        def file_loader_thread():
            paths = self.paths[:]
            if self.shuffle_files:
                random.shuffle(paths)
            for path in paths:
                print(f"📅 Prefetching from: {path}")
                chunk = torch.load(path, map_location=self.device)
                if self.shuffle_each_chunk:
                    chunk = chunk[torch.randperm(len(chunk))]
                file_queue.put(chunk)
            file_queue.put(stop_token)

        threading.Thread(target=file_loader_thread, daemon=True).start()

        while True:
            chunk = file_queue.get()
            if chunk is stop_token:
                break
            for row in chunk:
                yield row

# === TensorBuffer: Feeds batches of activations into trainSAE ===
class TensorBuffer:
    def __init__(self, data: IterableDataset, out_batch_size: int = 8192, device: str = "cpu"):
        self.data_loader = DataLoader(
            data,
            batch_size=out_batch_size,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False
        )
        self.device = device
        self.iterator = iter(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.iterator)
        return batch.to(self.device, non_blocking=True)

    def close(self):
        pass

    
def fix_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"🔒 All random seeds fixed to {seed}")

fix_all_seeds()


# === Define trainer classes and target GPUs ===
trainer_gpu_pairs = [
    ("panneallora_1e-3_test", PAnnealTrainerLoRa, "cuda:0"),
    ("panneallora_1e-2_test", PAnnealTrainerLoRa, "cuda:1"),
    ("panneallora_1e-1_test", PAnnealTrainerLoRa, "cuda:2")
]


# === Shared async dataset ===
dataset = AsyncStreamingDataset(
    path_pattern="saved_activations_70m/activations_*.pt",
    prefetch_depth=8,
    shuffle_files=True,
    shuffle_each_chunk=True,
    device="cpu",
)


# === Function to train a model on one GPU ===
def train_on_gpu(name, trainer_class, device):
    try:
        print(f"🚀 Starting {name} on {device}")
        buffer = TensorBuffer(data=dataset, out_batch_size=16384, device=device)

        trainer_cfg = {
            "trainer": trainer_class,
            "activation_dim": 512,
            "dict_size": 32768,
            "lr": 1e-4,
            "steps": 2000,            # 120000
            "warmup_steps": 1000,       # 1000
            "device": device,
            "layer": -1,
            "lm_name": "model.gpt_neox.final_layer_norm",
            "initial_sparsity_penalty": 0.1,
            "resample_steps": 25000,
        }
        
        if name == "panneallora_1e-3_test":
            trainer_cfg.update({
                "lora_coeff_scale": 0.001,
            })
        elif name == "panneallora_1e-2_test":
            trainer_cfg.update({
                "lora_coeff_scale": 0.01,
            })
        elif name == "panneallora_1e-1_test":
            trainer_cfg.update({
                "lora_coeff_scale": 0.1,
            })
        else: 
            raise NotImplementedError()
        
        ae = trainSAE(
            data=buffer,
            trainer_configs=[trainer_cfg],
            steps=2000,          # 120000,
            save_steps=1000,     # [20000, 40000, 60000, 80000, 100000, 120000],
            log_steps=100,          # 10000,
            verbose=True,
            save_dir=f"models/{name}"
        )
        print(f"✅ Finished {name} on {device}")
        
        if hasattr(ae, 'trainer') and hasattr(ae.trainer, 'svd_fallback_count'):
            fallback_rate = 100 * ae.trainer.svd_fallback_count / max(ae.trainer.svd_total_calls, 1)
            print(f"[Diagnostics] {name} on {device}: SVD fallback triggered {ae.trainer.svd_fallback_count} \
                times over {ae.trainer.svd_total_calls} nuclear norm calls. (Fallback rate: {fallback_rate:.2f}%)")

    except Exception as e:
        print(f"❌ ERROR in {name} on {device}: {str(e)}")
        import traceback
        traceback.print_exc()


# === Launch threads for each trainer + GPU ===
threads = []
for name, trainer_class, device in trainer_gpu_pairs:
    thread = threading.Thread(target=train_on_gpu, args=(name, trainer_class, device), daemon=True)
    thread.start()
    threads.append(thread)

# === Wait for all to complete ===
for t in threads:
    t.join()
