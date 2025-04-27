import torch as t
from typing import Optional
"""
Implements the standard SAE training scheme.
"""

from ..dictionary import AutoEncoderLoRa
from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG

class PAnnealTrainerLoRa(SAETrainer):
    """
    SAE training scheme with the option to anneal the sparsity parameter p.
    You can further choose to use Lp or Lp^p sparsity.
    """
    def __init__(self, 
                 steps: int, # total number of steps to train for
                 activation_dim: int,       # the dimension of input (LLM's) activations
                 dict_size: int,            # the dimension of the feature of SAE
                 layer: int,                # an index for the LLM activations' layer
                 lm_name: str,              # language model name
                 dict_class: type = AutoEncoderLoRa,        # the SAE strcuture
                 lr: float = 1e-3,
                 warmup_steps: int = 1000, # lr warmup period at start of training and after each resample
                 decay_start: Optional[int] = None, # step at which to start decaying lr
                 sparsity_warmup_steps: Optional[int] = 2000, # number of steps to warm up sparsity penalty
                 sparsity_function: str = 'Lp', # Lp or Lp^p
                 initial_sparsity_penalty: float = 1e-1, # equal to l1 penalty in standard trainer
                 lora_coeff_scale: float = 1e-1,     # the value of lora_coeff/sparsity_coeef
                 anneal_start: int = 15000, # step at which to start annealing p
                 anneal_end: Optional[int] = None, # step at which to stop annealing, defaults to steps-1
                 p_start: float = 1, # starting value of p (constant throughout warmup)
                 p_end: float = 0, # annealing p_start to p_end linearly after warmup_steps, exact endpoint excluded
                 n_sparsity_updates: int | str = 10, # number of times to update the sparsity penalty, at most steps-anneal_start times
                 sparsity_queue_length: int = 10, # number of recent sparsity loss terms, onle needed for adaptive_sparsity_penalty
                 resample_steps: Optional[int] = None, # number of steps after which to resample dead neurons
                 device: Optional[str] = None,
                 seed: int = 42,
                 wandb_name: str = 'PAnnealLoRaTrainer',
                 submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        if device is None:
            self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # initialize dictionary
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.ae = dict_class(activation_dim, dict_size, device=device)
        self.ae.to(self.device)
        
        self.lr = lr
        self.sparsity_function = sparsity_function
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end if anneal_end is not None else steps
        self.p_start = p_start
        self.p_end = p_end
        self.p = p_start
        self.next_p = None
        if n_sparsity_updates == "continuous":
            self.n_sparsity_updates = self.anneal_end - anneal_start +1
        else:
            self.n_sparsity_updates = n_sparsity_updates
        self.sparsity_update_steps = t.linspace(anneal_start, self.anneal_end, self.n_sparsity_updates, dtype=int)
        self.p_values = t.linspace(p_start, p_end, self.n_sparsity_updates)
        self.p_step_count = 0
        self.sparsity_coeff = initial_sparsity_penalty # alpha
        
        self.lora_coeff_scale = lora_coeff_scale
        
        self.sparsity_queue_length = sparsity_queue_length
        self.sparsity_queue = []

        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.decay_start = decay_start
        self.steps = steps
        self.logging_parameters = ['p', 'next_p', 'lp_loss', 'scaled_lp_loss', 'sparsity_coeff']
        self.seed = seed
        self.wandb_name = wandb_name

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(self.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        if (self.sparsity_update_steps.unique(return_counts=True)[1] >1).any():
            print("Warning! Duplicates om self.sparsity_update_steps detected!")

    def resample_neurons(self, deads, activations):
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # reset encoder/decoder weights for dead neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()
            self.ae.encoder.weight[deads][:n_resample] = sampled_vecs * alive_norm * 0.2
            
            alive_norm_lora = self.ae.lora_encoder.weight[~deads].norm(dim=-1).mean()
            self.ae.lora_encoder.weight[deads][:n_resample] = sampled_vecs * alive_norm_lora * 0.2
            
            self.ae.decoder.weight[:,deads][:,:n_resample] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.encoder.bias[deads][:n_resample] = 0.
            self.ae.lora_encoder.bias[deads][:n_resample] = 0.
        
            # reset Adam parameters for dead neurons
            state = self.optimizer.state
            for name, param in self.ae.named_parameters():
                if "encoder.weight" in name or "lora_encoder.weight" in name or "decoder.weight" in name or "bias" in name:
                    if param in state:
                        if "exp_avg" in state[param]:
                            state[param]["exp_avg"][deads] = 0.0
                        if "exp_avg_sq" in state[param]:
                            state[param]["exp_avg_sq"][deads] = 0.0

    def lp_norm(self, f, p):
        norm_sq = f.pow(p).sum(dim=-1)
        if self.sparsity_function == 'Lp^p':
            return norm_sq.mean()
        elif self.sparsity_function == 'Lp':
            return norm_sq.pow(1/p).mean()
        else:
            raise ValueError("Sparsity function must be 'Lp' or 'Lp^p'")
    
    
    def nuclear_norm(
            self, f_lora, step=None, mode="randomized",
            k=64, n_iter=2, compute_every=1
        ):
        if step is not None and (step % compute_every):
            return t.zeros((), device=f_lora.device, dtype=f_lora.dtype,
                            requires_grad=True)

        self.svd_total_calls += 1
        k = min(k, *f_lora.shape)          # keep k legal

        try:
            if mode == "exact":
                _, S, _ = t.linalg.svd(f_lora, full_matrices=False)
                return S.sum()

            if mode == "randomized":
                with t.no_grad():
                    f_lora = f_lora + 1e-6 * t.randn_like(f_lora)

                B = t.randn(f_lora.shape[1], k, device=f_lora.device)
                Y = f_lora @ B
                for _ in range(n_iter):
                    Y = f_lora @ (f_lora.T @ Y)
                    Y, _ = t.linalg.qr(Y, mode='reduced')   # stabilise

                Q, _ = t.linalg.qr(Y, mode='reduced')
                smaller = Q.T @ f_lora
                try:
                    _, S, _ = t.linalg.svd(smaller, full_matrices=False)
                except RuntimeError:
                    self.svd_fallback_count += 1
                    with t.no_grad():
                        _, S, _ = t.svd_lowrank(smaller, q=k)
                return S.sum()

            if mode == "subsampled":
                m = min(512, f_lora.shape[0])
                idx = t.randint(0, f_lora.shape[0], (m,),
                                    device=f_lora.device)
                f_sub = f_lora[idx]
                try:
                    _, S, _ = t.linalg.svd(f_sub, full_matrices=False)
                except RuntimeError:
                    self.svd_fallback_count += 1
                    with t.no_grad():
                        _, S, _ = t.svd_lowrank(
                            f_sub, q=min(32, *f_sub.shape))
                return S.sum()

            raise ValueError(f"Unknown mode: {mode}")

        except Exception as e:
            print(f"Nuclear norm failed at step {step}: {e}")
            self.svd_fallback_count += 1
            return t.zeros((), device=f_lora.device, dtype=f_lora.dtype,
                            requires_grad=True)

    def loss(self, x: t.Tensor, step:int, logging=False):
        sparsity_scale = self.sparsity_warmup_fn(step)

        # Compute loss terms
        x_hat, f, f_main, f_lora = self.ae(x, output_features=True)
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        lp_loss = self.lp_norm(f_main, self.p)
        scaled_lp_loss = lp_loss * self.sparsity_coeff * sparsity_scale
        
        lora_loss = self.nuclear_norm(
            f_lora,
            step=step,
            mode="randomized",
            k=64,
            n_iter=1,   # ← power iteration设小一点更安全
            compute_every=10
        )
        scaled_lora_loss = lora_loss * self.lora_coeff_scale * self.sparsity_coeff * sparsity_scale
        
        self.lp_loss = lp_loss
        self.scaled_lp_loss = scaled_lp_loss

        if self.next_p is not None:
            lp_loss_next = self.lp_norm(f, self.next_p)
            self.sparsity_queue.append([self.lp_loss.item(), lp_loss_next.item()])
            self.sparsity_queue = self.sparsity_queue[-self.sparsity_queue_length:]
    
        if step in self.sparsity_update_steps:
            # check to make sure we don't update on repeat step:
            if step >= self.sparsity_update_steps[self.p_step_count]:
                # Adapt sparsity penalty alpha
                if self.next_p is not None:
                    local_sparsity_new = t.tensor([i[0] for i in self.sparsity_queue]).mean()
                    local_sparsity_old = t.tensor([i[1] for i in self.sparsity_queue]).mean()
                    self.sparsity_coeff = self.sparsity_coeff * (local_sparsity_new / local_sparsity_old).item()
                # Update p
                self.p = self.p_values[self.p_step_count].item()
                if self.p_step_count < self.n_sparsity_updates-1:
                    self.next_p = self.p_values[self.p_step_count+1].item()
                else:
                    self.next_p = self.p_end
                self.p_step_count += 1

        # Update dead feature count
        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0        
    
        if logging is False:
            return recon_loss + scaled_lp_loss + scaled_lora_loss
        else: 
            loss_log = {
                'p' : self.p,
                'next_p' : self.next_p,
                'lp_loss' : lp_loss.item(),
                'scaled_lp_loss' : scaled_lp_loss.item(),
                'sparsity_coeff' : self.sparsity_coeff,
                'lora_loss': lora_loss.item(),
                'scaled_lora_loss': scaled_lora_loss.item(),
            }
            return x, x_hat, f, loss_log
    
        
    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step, logging=False)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == self.resample_steps - 1:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'trainer_class' : "PAnnealTrainer",
            'dict_class' : "AutoEncoder",
            'activation_dim' : self.activation_dim,
            'dict_size' : self.dict_size,
            'lr' : self.lr,
            'sparsity_function' : self.sparsity_function,
            'sparsity_penalty' : self.sparsity_coeff,
            'p_start' : self.p_start,
            'p_end' : self.p_end,
            'anneal_start' : self.anneal_start,
            'sparsity_queue_length' : self.sparsity_queue_length,
            'n_sparsity_updates' : self.n_sparsity_updates,
            'warmup_steps' : self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'decay_start': self.decay_start,
            'resample_steps' : self.resample_steps,
            'steps' : self.steps,
            'seed' : self.seed,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name' : self.wandb_name,
            'submodule_name' : self.submodule_name,
            'sparsity_coeff' : self.sparsity_coeff,
        }
