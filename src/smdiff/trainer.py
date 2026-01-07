"""
Training loop for symbolic music diffusion models.
Integrates with the unified registry/config system while preserving
compatibility with legacy hparams infrastructure.
"""
import copy
import time
import os
import numpy as np
import torch
from tqdm import tqdm

from .data import SimpleNpyDataset
from .utils.log_utils import (
    log, save_model, save_samples, save_stats,
    load_model, load_stats, log_stats
)
from .utils.sampler_utils import get_sampler, get_samples
from .utils.train_utils import EMA, optim_warmup, augment_note_tensor
from .data.base import cycle


def main(H):
    """
    Main training loop.
    
    Args:
        H: Hyperparameters object with all training config
    """

    data_np = np.load(H.dataset_path, allow_pickle=True)
    midi_data = SimpleNpyDataset(data_np, H.NOTES, tokenizer_id=getattr(H, 'tokenizer_id', None))

    # Split train/val
    val_idx = int(len(midi_data) * H.validation_set_size)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(midi_data, range(val_idx, len(midi_data))),
        batch_size=H.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(midi_data, range(val_idx)),
        batch_size=H.batch_size,
    )

    log(f'Total train batches: {len(train_loader)}, eval: {len(val_loader)}')
    if H.epochs:
        H.train_steps = int(H.epochs * len(train_loader))
        log(f'Training for {H.epochs} epochs = {H.train_steps} steps')

    # Initialize model (via registry-aware get_sampler)
    sampler = get_sampler(H).cuda()
    optim = torch.optim.Adam(sampler.parameters(), lr=H.lr)

    # EMA setup
    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    # Training state
    losses = np.array([])
    val_losses = np.array([])
    elbo = np.array([])
    val_elbos = np.array([])
    mean_losses = np.array([])
    cons_var = np.array([]), np.array([]), np.array([]), np.array([])
    start_step = 0

    # Resume from checkpoint
    if H.load_step > 0:
        start_step = H.load_step + 1

        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda()
        log("Loaded model checkpoint")
        
        if H.ema:
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir
                )
            except Exception:
                ema_sampler = copy.deepcopy(sampler)
                
        if H.load_optim:
            optim = load_model(
                optim, f'{H.sampler}_optim', H.load_step, H.load_dir
            )
            for param_group in optim.param_groups:
                param_group['lr'] = H.lr

        try:
            train_stats = load_stats(H, H.load_step)
        except Exception:
            train_stats = None

        if train_stats is not None:
            losses = train_stats["losses"]
            mean_losses = train_stats["mean_losses"]
            val_losses = train_stats["val_losses"]
            val_elbos = train_stats["val_elbos"]
            elbo = train_stats["elbo"]
            cons_var = train_stats["cons_var"]
            H.steps_per_log = train_stats["steps_per_log"]
            log(f"Resumed from step {H.load_step}: mean_loss={mean_losses[-1]:.6f}, val_loss={val_losses[-1]:.6f}")
        else:
            log('No stats file found for loaded model, displaying stats from load step only.')

    sampler = sampler.cuda()

    scaler = torch.cuda.amp.GradScaler()
    train_iterator = cycle(train_loader)

    log(f"Sampler params total: {sum(p.numel() for p in sampler.parameters())}")
    
    # Log tokenizer and masking info if available
    if hasattr(H, 'tokenizer_id'):
        log(f"Tokenizer: {H.tokenizer_id}")
    if hasattr(H, 'masking_strategy') and H.masking_strategy:
        log(f"Masking strategy: {H.masking_strategy}")

    # Training loop
    for step in range(start_step, H.train_steps):
        sampler.train()
        if H.ema:
            ema_sampler.train()
        step_start_time = time.time()
        
        # LR warmup
        if H.warmup_iters:
            if step <= H.warmup_iters:
                optim_warmup(H, step, optim)

        x = augment_note_tensor(H, next(train_iterator))
        x = x.cuda(non_blocking=True)

        if H.amp:
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                stats = sampler.train_iter(x)

            scaler.scale(stats['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            stats = sampler.train_iter(x)

            if torch.isnan(stats['loss']).any():
                log(f'Skipping step {step} with NaN loss')
                continue
            optim.zero_grad()
            stats['loss'].backward()
            optim.step()

        losses = np.append(losses, stats['loss'].item())

        sampler.eval()
        if H.ema:
            ema_sampler.eval()

        # Logging
        if step % H.steps_per_log == 0:
            step_time_taken = time.time() - step_start_time
            stats['step_time'] = step_time_taken
            mean_loss = np.mean(losses)
            stats['mean_loss'] = mean_loss
            mean_losses = np.append(mean_losses, mean_loss)
            losses = np.array([])

            log_stats(step, stats)

            if H.sampler == 'absorbing':
                elbo = np.append(elbo, stats['vb_loss'].item())

        # EMA update
        if H.ema and step % H.steps_per_update_ema == 0 and step:
            ema.update_model_average(ema_sampler, sampler)

        # Sampling
        if step % H.steps_per_sample == 0 and step:
            log(f"Sampling step {step}")
            samples = get_samples(
                ema_sampler if H.ema else sampler,
                H.sample_steps,
                b=H.show_samples
            )
            save_samples(samples, step, H.log_dir)

        # Validation
        if H.steps_per_eval and step % H.steps_per_eval == 0 and step:
            min_step = H.steps_per_eval
            valid_loss, valid_elbo, num_batches = 0.0, 0.0, 0
            log(f"Evaluating step {step}")

            for x in tqdm(val_loader):
                with torch.no_grad():
                    stats = sampler.train_iter(x.cuda())
                    valid_loss += stats['loss'].item()
                    if H.sampler == 'absorbing':
                        valid_elbo += stats['vb_loss'].item()
                    num_batches += 1
            valid_loss = valid_loss / num_batches
            if H.sampler == 'absorbing':
                valid_elbo = valid_elbo / num_batches

            val_losses = np.append(val_losses, valid_loss)
            val_elbos = np.append(val_elbos, valid_elbo)
            log(f"Validation at step {step}: loss={valid_loss:.6f}" + 
                (f", elbo={valid_elbo:.6f}" if H.sampler == 'absorbing' else ""))

        # Checkpointing
        if step % H.steps_per_checkpoint == 0 and step > H.load_step:
            save_model(sampler, H.sampler, step, H.log_dir)
            save_model(optim, f'{H.sampler}_optim', step, H.log_dir)

            if H.ema:
                save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)

            train_stats = {
                'losses': losses,
                'mean_losses': mean_losses,
                'val_losses': val_losses,
                'elbo': elbo,
                'val_elbos': val_elbos,
                'cons_var': cons_var,
                'steps_per_log': H.steps_per_log,
                'steps_per_eval': H.steps_per_eval,
            }
            save_stats(H, train_stats, step)
