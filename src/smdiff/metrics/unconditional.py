"""Metrics for unconditional generation evaluation."""
import numpy as np
from .common import (
    kl_divergence,
    pitch_class_histogram,
    duration_histogram,
    velocity_histogram,
    note_density_per_bar,
    compute_self_similarity,
    compute_pitch_range,
    compute_sample_diversity,
    is_valid_sample
)


def evaluate_unconditional(generated_samples, train_samples, is_octuple=True):
    """
    Evaluate unconditional generation against training data.
    
    Args:
        generated_samples: List of (T, C) generated token arrays (variable length)
        train_samples: List of (T, C) training token arrays (variable length)
        is_octuple: Whether samples are Octuple encoded (Defaults to True for this script)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Octuple indices: Bar=0, Pos=1, Prog=2, Pitch=3, Dur=4, Vel=5
    # Enforce Octuple defaults strictly
    pitch_idx = 3
    duration_idx = 4
    velocity_idx = 5
    bar_idx = 0

    # Compute distributions on lists (functions handle variable-length sequences)
    # Ensure inputs are lists of arrays
    if not isinstance(generated_samples, list):
         generated_samples = [s for s in generated_samples]
    if not isinstance(train_samples, list):
         train_samples = [s for s in train_samples]

    gen_pch = pitch_class_histogram(generated_samples, pitch_idx=pitch_idx)
    train_pch = pitch_class_histogram(train_samples, pitch_idx=pitch_idx)
    metrics['pch_kl'] = kl_divergence(train_pch, gen_pch)
    
    # Duration KL
    # Increase max_bins for Octuple (which has typically >32 duration tokens)
    gen_dur = duration_histogram(generated_samples, duration_idx=duration_idx, max_bins=128)
    train_dur = duration_histogram(train_samples, duration_idx=duration_idx, max_bins=128)
    metrics['duration_kl'] = kl_divergence(train_dur, gen_dur)
    
    # Velocity KL
    gen_vel = velocity_histogram(generated_samples, velocity_idx=velocity_idx)
    train_vel = velocity_histogram(train_samples, velocity_idx=velocity_idx)
    metrics['velocity_kl'] = kl_divergence(train_vel, gen_vel)
    
    # Note Density KL
    gen_density = note_density_per_bar(generated_samples, bar_idx=bar_idx)
    train_density = note_density_per_bar(train_samples, bar_idx=bar_idx)
    
    # Create histograms for note density
    # Handle empty arrays if any
    max_d_gen = gen_density.max() if len(gen_density) > 0 else 0
    max_d_train = train_density.max() if len(train_density) > 0 else 0
    max_density = int(max(max_d_gen, max_d_train) + 1)
    
    gen_density_hist = np.bincount(gen_density.astype(int), minlength=max_density)
    train_density_hist = np.bincount(train_density.astype(int), minlength=max_density)
    metrics['note_density_kl'] = kl_divergence(train_density_hist, gen_density_hist)
    
    # Musical coherence metrics (per-sample averages)
    self_sims = []
    pitch_ranges = []
    
    for sample in generated_samples:
        if len(sample) == 0:
            continue
        self_sims.append(compute_self_similarity(sample, pitch_idx=pitch_idx, duration_idx=duration_idx))
        pitch_ranges.append(compute_pitch_range(sample, pitch_idx=pitch_idx))
    
    metrics['self_similarity'] = np.mean(self_sims) if self_sims else 0.0
    metrics['self_similarity_std'] = np.std(self_sims) if self_sims else 0.0
    
    metrics['pitch_range_mean'] = np.mean(pitch_ranges) if pitch_ranges else 0.0
    metrics['pitch_range_std'] = np.std(pitch_ranges) if pitch_ranges else 0.0
    
    # Diversity metric
    metrics['sample_diversity'] = compute_sample_diversity(generated_samples, 
                                                         pitch_idx=pitch_idx, 
                                                         duration_idx=duration_idx)
    
    # Validity metric
    valid_count = sum([is_valid_sample(s, 
                                      pitch_idx=pitch_idx, 
                                      duration_idx=duration_idx) for s in generated_samples])
    metrics['valid_samples_pct'] = 100.0 * valid_count / len(generated_samples) if len(generated_samples) > 0 else 0.0
    
    return metrics
