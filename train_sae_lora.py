import argparse
import random
import torch
import numpy as np
import glob
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from dictionary_learning.trainers import PAnnealTrainerLoRa
from dictionary_learning.training import trainSAE

# === Async *batch* streaming Dataset ===
class BatchStreamingDataset(IterableDataset):
    """Streams activation tensors from *.pt shards and yields mini-batches.

    Each worker process gets its own slice of the file list, so we scale I/O
    bandwidth roughly linearly with num_workers.  Shuffling happens at two
    levels:
        1.  Per-process file list is shuffled once per epoch.
        2.  Within each shard we randperm and then cut contiguous batches.
    """

    def __init__(
        self,
        path_pattern: str,
        batch_rows: int = 8192,
        shuffle_files: bool = True,
        shuffle_each_chunk: bool = True,
        mmap: bool = True,
    ) -> None:
        super().__init__()
        self.paths = sorted(glob.glob(path_pattern))
        if not self.paths:
            raise FileNotFoundError(f"No files match pattern: {path_pattern}")
        self.batch_rows = batch_rows
        self.shuffle_files = shuffle_files
        self.shuffle_each_chunk = shuffle_each_chunk
        self.mmap = mmap

    # ------------------------------------------------------------------
    def _rng(self, base_seed: int = 0):
        """Per-worker RNG helper so shuffling is independent across workers."""
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        return random.Random(base_seed + worker_id)

    # ------------------------------------------------------------------
    def _shard_paths(self):
        """Split the global file list so each worker reads its own slice."""
        info = get_worker_info()
        if info is None:  # single-process DataLoader
            return self.paths
        # Round-robin assignment: worker 0 gets paths[0::num_workers], etc.
        return self.paths[info.id :: info.num_workers]

    # ------------------------------------------------------------------
    def __iter__(self):
        rng = self._rng(base_seed=42)
        paths = self._shard_paths()
        if self.shuffle_files:
            rng.shuffle(paths)

        for path in paths:
            # torch.load supports mmap since PyTorch 2.3
            try:
                chunk = torch.load(path, mmap=self.mmap, map_location="cpu")
            except TypeError:  # older torch; fall back to standard load
                chunk = torch.load(path, map_location="cpu")

            if self.shuffle_each_chunk:
                # Derive a reproducible seed *per chunk* so each worker’s
                # shuffle differs even when paths overlap.
                local_gen = torch.Generator()
                local_gen.manual_seed(rng.randint(0, 2 ** 31))
                idx = torch.randperm(len(chunk), generator=local_gen)
                chunk = chunk[idx]

            # ── Yield fixed-size mini-batches ───────────────────────────
            for i in range(0, len(chunk), self.batch_rows):
                batch = chunk[i : i + self.batch_rows]
                if len(batch) == self.batch_rows:  # drop last partial batch
                    yield batch


# === Optional host→device pre-fetcher =========================================
class DevicePrefetchLoader:
    """Wraps a DataLoader so that the *next* batch is already on the GPU."""

    def __init__(self, loader: DataLoader, device: str = "cuda") -> None:
        self.loader = loader
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

    def __iter__(self):
        self.iter = iter(self.loader)
        if self.stream is not None:
            self._preload()
        return self

    def __next__(self):
        if self.stream is None:
            # CPU training or no CUDA available
            return next(self.iter)

        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    # ------------------------------------------------------------------
    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            raise
        with torch.cuda.stream(self.stream):
            self.next_batch = batch.to(self.device, non_blocking=True)


# === Fix random seeds so results are reproducible ============================

def fix_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"🔒  All random seeds fixed to {seed}")


# === Main Training Function ==================================================

def main(args):
    fix_all_seeds()

    print(f"🚀  Starting experiment: {args.experiment_name} on device: {args.device}")

    # 1. Build dataset → DataLoader with multi-processing workers
    dataset = BatchStreamingDataset(
        path_pattern=args.data_path,
        batch_rows=args.batch_size,
        shuffle_files=True,
        shuffle_each_chunk=True,
        mmap=not args.disable_mmap,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,          # dataset already returns full mini-batch tensors
        shuffle=False,            # shuffling handled inside the dataset
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=lambda x: x[0],  # unwrap list([tensor]) → tensor
    )

    # 2. Optional GPU pre-fetcher (only makes sense when training on CUDA)
    data_iter = DevicePrefetchLoader(loader, device=args.device) if "cuda" in args.device else loader

    # 3. Configure trainer
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

    # 4. Kick off training
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

    # 5. Diagnostics
    if hasattr(ae, "trainer") and hasattr(ae.trainer, "svd_fallback_count"):
        fb = ae.trainer.svd_fallback_count
        total = max(ae.trainer.svd_total_calls, 1)
        print(f"[Diagnostics] SVD fallback triggered {fb} / {total} times ({100*fb/total:.2f}%).")


# === CLI entry-point ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single SAE model with LoRA (fast data loader)")

    # --- General experiment params ------------------------------------
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g. cuda:0 or cpu")

    # --- Dataset & loader ---------------------------------------------
    parser.add_argument("--data_path", type=str, default="saved_activations_70m/activations_*.pt", help="Glob for activation shards")
    parser.add_argument("--batch_size", type=int, default=16384, help="Rows per training batch")
    parser.add_argument("--num_workers", type=int, default=4, help="Multiprocessing workers for DataLoader (0 ⇒ single-process)")
    parser.add_argument("--disable_mmap", action="store_true", help="Fallback to regular torch.load if mmap causes issues")

    # --- Model / trainer hyper-params ----------------------------------
    parser.add_argument("--activation_dim", type=int, default=512, help="Activation dimension")
    parser.add_argument("--dict_size", type=int, default=32768, help="Dictionary size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_steps", type=int, default=12000, help="Training steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR warm-up steps")
    parser.add_argument("--resample_steps", type=int, default=2500, help="Resample dead neurons every N steps")
    parser.add_argument("--sparsity_warmup_steps", type=int, default=200, help="Steps to warm-up sparsity penalty")
    parser.add_argument("--initial_sparsity_penalty", type=float, default=0.1, help="Initial sparsity penalty weight")
    parser.add_argument("--lora_coeff_scale", type=float, required=True, help="LoRA loss scaling factor")

    # --- LLM layer selection ------------------------------------------
    parser.add_argument("--layer", type=int, default=-1, help="LLM layer index (-1 = final)" )
    parser.add_argument("--lm_name", type=str, default="model.gpt_neox.final_layer_norm", help="Layer name in the checkpoint")

    # --- Logging / checkpointing --------------------------------------
    parser.add_argument("--save_steps", type=int, nargs="+", default=[2000, 4000, 6000, 8000, 10000, 12000], help="Checkpoint every N steps")
    parser.add_argument("--log_steps", type=int, default=1000, help="Print training metrics every N steps")

    parsed_args = parser.parse_args()
    main(parsed_args)
    
    

# python train_sae_lora.py --experiment_name panneallora_1e-3_12k --device cuda --lora_coeff_scale 0.001