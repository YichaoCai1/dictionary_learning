import argparse
import random
import torch
import numpy as np
import glob
import threading
import queue
from torch.utils.data import IterableDataset, DataLoader
from dictionary_learning.trainers import PAnnealTrainerLoRa
from dictionary_learning.training import trainSAE

# === Asynchronous Streaming Dataset ===
class AsyncStreamingDataset(IterableDataset):
    def __init__(self, path_pattern, prefetch_depth=4, shuffle_files=True, shuffle_each_chunk=True, device="cpu"):
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

# === TensorBuffer ===
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

# === Fix random seeds ===
def fix_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"🔒 All random seeds fixed to {seed}")

# === Main Training Function ===
def main(args):
    fix_all_seeds()

    print(f"🚀 Starting experiment: {args.experiment_name} on device: {args.device}")

    dataset = AsyncStreamingDataset(
        path_pattern=args.data_path,
        prefetch_depth=args.prefetch_depth,
        shuffle_files=True,
        shuffle_each_chunk=True,
        device="cpu",
    )

    buffer = TensorBuffer(data=dataset, out_batch_size=args.batch_size, device=args.device)

    trainer_cfg = {
        "trainer": PAnnealTrainerLoRa,
        "activation_dim": args.activation_dim,
        "dict_size": args.dict_size,
        "lr": args.lr,
        "steps": args.train_steps,
        "warmup_steps": args.warmup_steps,
        "device": args.device,
        "layer": args.layer,
        "lm_name": args.lm_name,
        "initial_sparsity_penalty": args.initial_sparsity_penalty,
        "resample_steps": args.resample_steps,
        "sparsity_warmup_steps": args.sparsity_warmup_steps,
        "lora_coeff_scale": args.lora_coeff_scale,
    }

    ae = trainSAE(
        data=buffer,
        trainer_configs=[trainer_cfg],
        steps=args.train_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        verbose=True,
        save_dir=f"models/{args.experiment_name}"
    )

    print(f"✅ Finished training {args.experiment_name}")

    if hasattr(ae, 'trainer') and hasattr(ae.trainer, 'svd_fallback_count'):
        fallback_rate = 100 * ae.trainer.svd_fallback_count / max(ae.trainer.svd_total_calls, 1)
        print(f"[Diagnostics] SVD fallback triggered {ae.trainer.svd_fallback_count} times "
              f"over {ae.trainer.svd_total_calls} nuclear norm calls. (Fallback rate: {fallback_rate:.2f}%)")

# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single SAE model with LoRA")

    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (e.g., cuda:0)")
    parser.add_argument("--data_path", type=str, default="saved_activations_70m/activations_*.pt", help="Path pattern for activation files")
    parser.add_argument("--prefetch_depth", type=int, default=8, help="Prefetch depth for async loading")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size")
    parser.add_argument("--activation_dim", type=int, default=512, help="Activation dimension")
    parser.add_argument("--dict_size", type=int, default=32768, help="Dictionary size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_steps", type=int, default=12000, help="Number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for LR")
    parser.add_argument("--resample_steps", type=int, default=2500, help="Resample dead neurons every N steps")
    parser.add_argument("--sparsity_warmup_steps", type=int, default=200, help="Steps to warm up sparsity penalty")
    parser.add_argument("--initial_sparsity_penalty", type=float, default=0.1, help="Initial sparsity penalty")
    parser.add_argument("--lora_coeff_scale", type=float, required=True, help="LoRA loss scaling factor")
    parser.add_argument("--layer", type=int, default=-1, help="LLM layer index")
    parser.add_argument("--lm_name", type=str, default="model.gpt_neox.final_layer_norm", help="Language model layer name")
    parser.add_argument("--save_steps", type=int, nargs="+", default=[2000, 4000, 6000, 8000, 10000, 12000], help="Save checkpoints at these steps")
    parser.add_argument("--log_steps", type=int, default=1000, help="Log progress every N steps")

    args = parser.parse_args()

    main(args)


# python train_sae_lora.py --experiment_name panneallora_1e-3_12k --device cuda --lora_coeff_scale 0.001
