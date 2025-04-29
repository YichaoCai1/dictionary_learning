# train_sae_lora.py — revised to match your original structure but with the
# performance fixes we discussed: multi-process streaming, batch-level yielding,
# optional GPU prefetch, and safer memory use.
# ---------------------------------------------------------------------------

import argparse
import glob
import random
import queue
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from dictionary_learning.trainers import PAnnealTrainerLoRa
from dictionary_learning.training import trainSAE

# ════════════════════════════════════════════════════════════════════════════
# 1. Batch-level Async Streaming Dataset
# ════════════════════════════════════════════════════════════════════════════
class AsyncStreamingDataset(IterableDataset):
    """Streams activation shards (saved with torch.save) and yields *mini-batches*.

    Each DataLoader worker is assigned a distinct slice of the file list so we
    scale I/O throughput as you increase --num_workers.  In-chunk shuffling keeps
    sample order stochastic without a heavy global shuffle.
    """

    def __init__(
        self,
        path_pattern: str,
        batch_size: int = 8192,
        shuffle_files: bool = True,
        shuffle_each_chunk: bool = True,
        use_mmap: bool = True,
    ):
        super().__init__()
        self.path_pattern = path_pattern
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.shuffle_each_chunk = shuffle_each_chunk
        self.use_mmap = use_mmap
        self.paths = sorted(glob.glob(path_pattern))
        if not self.paths:
            raise FileNotFoundError(f"No files matched {path_pattern}")

    # ------------------------------------------------------------------
    def _rng(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        return random.Random(42 + worker_id)

    # ------------------------------------------------------------------
    def _worker_paths(self):
        info = get_worker_info()
        if info is None:
            return self.paths
        return self.paths[info.id :: info.num_workers]

    # ------------------------------------------------------------------
    def __iter__(self):
        rng = self._rng()
        paths = self._worker_paths()
        if self.shuffle_files:
            rng.shuffle(paths)

        for path in paths:
            # torch.load mmap=True needs torch ≥2.3; fall back silently if older
            try:
                chunk = torch.load(path, mmap=self.use_mmap, map_location="cpu")
            except TypeError:
                chunk = torch.load(path, map_location="cpu")

            if self.shuffle_each_chunk:
                g = torch.Generator().manual_seed(rng.randint(0, 2 ** 31))
                chunk = chunk[torch.randperm(len(chunk), generator=g)]

            # yield contiguous mini-batches, drop the last incomplete one
            full_batches = len(chunk) // self.batch_size
            if full_batches == 0:
                continue
            chunk = chunk[: full_batches * self.batch_size]
            chunk = chunk.view(full_batches, self.batch_size, *chunk.shape[1:])
            for batch in chunk:
                yield batch


# ════════════════════════════════════════════════════════════════════════════
# 2. Optional host→device prefetch wrapper
# ════════════════════════════════════════════════════════════════════════════
class PrefetchLoader:
    """Overlaps host-to-device copies with compute (CUDA only)."""

    def __init__(self, loader: DataLoader, device: str):
        self.loader = loader
        self.device = torch.device(device)
        self.stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

    def __iter__(self):
        self.it = iter(self.loader)
        if self.stream is not None:
            self._preload()
        return self

    def __next__(self):
        if self.stream is None:
            return next(self.it)
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    def _preload(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            raise
        with torch.cuda.stream(self.stream):
            self.next_batch = batch.to(self.device, non_blocking=True)


# ════════════════════════════════════════════════════════════════════════════
# 3. Utilities
# ════════════════════════════════════════════════════════════════════════════

def fix_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"🔒  All random seeds fixed to {seed}")


# ════════════════════════════════════════════════════════════════════════════
# 4. Training harness
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    fix_all_seeds()
    device = args.device

    print(f"🚀  Starting experiment: {args.experiment_name} on {device}")

    fix_all_seeds()
    device = args.device

    print(f"🚀  Starting experiment: {args.experiment_name} on {device}")

    # Build streaming dataset → DataLoader
    dataset = AsyncStreamingDataset(
        path_pattern=args.data_path,
        batch_size=args.batch_size,
        shuffle_files=True,
        shuffle_each_chunk=True,
        use_mmap=not args.disable_mmap,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,            # dataset already returns ready-made tensors
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=lambda x: x[0],  # DataLoader wraps yield in a list
    )
    
    sample_batch = next(iter(loader))          # CPU tensor
    print("Batch shape:", sample_batch.shape)  # should be (batch_size, activation_dim)
    assert sample_batch.shape == (
        args.batch_size,
        args.activation_dim,
    ), "Shape mismatch with trainSAE expectations!"


    data_iter = PrefetchLoader(loader, device) if "cuda" in device else loader

    trainer_cfg = {
        "trainer": PAnnealTrainerLoRa,
        "activation_dim": args.activation_dim,
        "dict_size": args.dict_size,
        "lr": args.lr,
        "steps": args.train_steps,
        "warmup_steps": args.warmup_steps,
        "device": device,
        "layer": args.layer,
        "lm_name": args.lm_name,
        "initial_sparsity_penalty": args.initial_sparsity_penalty,
        "resample_steps": args.resample_steps,
        "sparsity_warmup_steps": args.sparsity_warmup_steps,
        "lora_coeff_scale": args.lora_coeff_scale,
    }

    ae = trainSAE(
        data=data_iter,
        trainer_configs=[trainer_cfg],
        steps=args.train_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        verbose=True,
        save_dir=f"models/{args.experiment_name}",
    )

    print(f"✅  Finished training {args.experiment_name}")

    if hasattr(ae, "trainer") and hasattr(ae.trainer, "svd_fallback_count"):
        fb = ae.trainer.svd_fallback_count
        tot = max(ae.trainer.svd_total_calls, 1)
        print(f"[Diagnostics] SVD fallback triggered {fb}/{tot} times ({100*fb/tot:.2f}%).")


# ════════════════════════════════════════════════════════════════════════════
# 5. CLI entry-point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single SAE model with LoRA (fast loader)")

    # Essential args
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_path", default="saved_activations_70m/activations_*.pt")

    # Data loader params
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_mmap", action="store_true")

    # Model / optimisation hyper-params (unchanged from your original)
    parser.add_argument("--activation_dim", type=int, default=512)
    parser.add_argument("--dict_size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_steps", type=int, default=12000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--resample_steps", type=int, default=2500)
    parser.add_argument("--sparsity_warmup_steps", type=int, default=200)
    parser.add_argument("--initial_sparsity_penalty", type=float, default=0.1)
    parser.add_argument("--lora_coeff_scale", type=float, required=True)

    # Layer selection
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--lm_name", default="model.gpt_neox.final_layer_norm")

    # Logging & checkpoints
    parser.add_argument("--save_steps", type=int, nargs="+", default=[2000,4000,6000,8000,10000,12000])
    parser.add_argument("--log_steps", type=int, default=1000)

    args = parser.parse_args()
    main(args)
