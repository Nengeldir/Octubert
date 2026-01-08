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


def evaluate_unconditional(generated_samples, train_samples):
    """
    Evaluate unconditional generation against training data.
    
    Args:
        generated_samples: List of (T, C) generated token arrays
        train_samples: List of (T, C) training token arrays
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Convert to numpy arrays
    gen_array = np.array([s for s in generated_samples])
    train_array = np.array([s for s in train_samples])
    
    # Distribution similarity metrics
    gen_pch = pitch_class_histogram(gen_array)
    train_pch = pitch_class_histogram(train_array)
    metrics['pch_kl'] = kl_divergence(train_pch, gen_pch)
    
    gen_dur = duration_histogram(gen_array)
    train_dur = duration_histogram(train_array)
    metrics['duration_kl'] = kl_divergence(train_dur, gen_dur)
    
    gen_vel = velocity_histogram(gen_array)
    train_vel = velocity_histogram(train_array)
    metrics['velocity_kl'] = kl_divergence(train_vel, gen_vel)
    
    gen_density = note_density_per_bar(gen_array)
    train_density = note_density_per_bar(train_array)
    
    # Create histograms for note density
    max_density = max(gen_density.max(), train_density.max()) + 1
    gen_density_hist = np.bincount(gen_density.astype(int), minlength=max_density)
    train_density_hist = np.bincount(train_density.astype(int), minlength=max_density)
    metrics['note_density_kl'] = kl_divergence(train_density_hist, gen_density_hist)
    
    # Musical coherence metrics (per-sample averages)
    self_sims = []
    pitch_ranges = []
    polyphonies = []
    
    for sample in generated_samples:
        self_sims.append(compute_self_similarity(sample))
        pitch_ranges.append(compute_pitch_range(sample))
        # Polyphony: average notes per timestep (for trio, multiple tracks can be active)
        # Approximate by counting non-zero/non-padding tokens per timestep
        polyphonies.append(_compute_polyphony(sample))
    
    metrics['self_similarity'] = np.mean(self_sims)
    metrics['self_similarity_std'] = np.std(self_sims)
    metrics['pitch_range_mean'] = np.mean(pitch_ranges)
    metrics['pitch_range_std'] = np.std(pitch_ranges)
    metrics['avg_polyphony'] = np.mean(polyphonies)
    
    # Diversity metric
    metrics['sample_diversity'] = compute_sample_diversity(generated_samples)
    
    # Validity metric
    valid_count = sum([is_valid_sample(s) for s in generated_samples])
    metrics['valid_samples_pct'] = 100.0 * valid_count / len(generated_samples)
    
    return metrics


def _compute_polyphony(tokens):
    """
    Estimate polyphony (average simultaneous notes).
    For octuple encoding, this is approximated by active tracks per timestep.
    """
    # Simple heuristic: count non-padding tokens per timestep
    # In octuple, if pitch > 0, it's an active note
    pitches = tokens[:, 2]
    active_per_step = (pitches > 0).sum() / len(pitches)
    return active_per_step
