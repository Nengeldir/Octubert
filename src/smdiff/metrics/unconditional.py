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


def evaluate_unconditional(generated_samples, train_samples, is_octuple=False):
    """
    Evaluate unconditional generation against training data.
    
    Args:
        generated_samples: List of (T, C) generated token arrays (variable length)
        train_samples: List of (T, C) training token arrays (variable length)
        is_octuple: Whether samples are Octuple encoded
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Define indices based on encoding
    if is_octuple:
        pitch_idx = 3
        duration_idx = 4
        velocity_idx = 5
        bar_idx = 0
    else:
        # Grid/Event-List (Melody/Trio)
        pitch_idx = 0
        duration_idx = None
        velocity_idx = None
        bar_idx = None

    # Compute distributions on lists (functions handle variable-length sequences)
    gen_pch = pitch_class_histogram(generated_samples, pitch_idx=pitch_idx)
    train_pch = pitch_class_histogram(train_samples, pitch_idx=pitch_idx)
    metrics['pch_kl'] = kl_divergence(train_pch, gen_pch)
    
    if duration_idx is not None:
        # Increase max_bins for Octuple (which has typically >32 duration tokens)
        gen_dur = duration_histogram(generated_samples, duration_idx=duration_idx, max_bins=128)
        train_dur = duration_histogram(train_samples, duration_idx=duration_idx, max_bins=128)
        metrics['duration_kl'] = kl_divergence(train_dur, gen_dur)
    else:
        metrics['duration_kl'] = 0.0
    
    if velocity_idx is not None:
        gen_vel = velocity_histogram(generated_samples, velocity_idx=velocity_idx)
        train_vel = velocity_histogram(train_samples, velocity_idx=velocity_idx)
        metrics['velocity_kl'] = kl_divergence(train_vel, gen_vel)
    else:
        metrics['velocity_kl'] = 0.0
    
    gen_density = note_density_per_bar(generated_samples, bar_idx=bar_idx)
    train_density = note_density_per_bar(train_samples, bar_idx=bar_idx)
    
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
        if duration_idx is not None:
            self_sims.append(compute_self_similarity(sample, pitch_idx=pitch_idx, duration_idx=duration_idx))
        else:
            self_sims.append(0.0) # TODO: support pitch-only self-similarity
        
        pitch_ranges.append(compute_pitch_range(sample, pitch_idx=pitch_idx))
        # Polyphony: average notes per timestep (for trio, multiple tracks can be active)
        # Approximate by counting non-zero/non-padding tokens per timestep
        polyphonies.append(_compute_polyphony(sample, pitch_idx=pitch_idx))
    
    metrics['self_similarity'] = np.mean(self_sims)
    metrics['self_similarity_std'] = np.std(self_sims)
    metrics['pitch_range_mean'] = np.mean(pitch_ranges)
    metrics['pitch_range_std'] = np.std(pitch_ranges)
    metrics['avg_polyphony'] = np.mean(polyphonies)
    
    # Diversity metric
    metrics['sample_diversity'] = compute_sample_diversity(generated_samples, 
                                                         pitch_idx=pitch_idx, 
                                                         duration_idx=duration_idx if duration_idx is not None else 0)
    
    # Validity metric
    valid_count = sum([is_valid_sample(s, 
                                      pitch_idx=pitch_idx, 
                                      duration_idx=duration_idx if duration_idx is not None else 0) for s in generated_samples])
    metrics['valid_samples_pct'] = 100.0 * valid_count / len(generated_samples)
    
    return metrics


def _compute_polyphony(tokens, pitch_idx=2):
    """
    Estimate polyphony (average simultaneous notes).
    For octuple encoding, this is approximated by active tracks per timestep.
    """
    # Simple heuristic: count non-padding tokens per timestep
    # If pitch > 0, it's an active note
    if tokens.ndim == 2:
        # If input is (T, C) [e.g. Trio]
        # Count cells > 0 across channels for each step
        # But wait, pitch_idx=0 is default for Melody.
        # If it's Trio, we assume all channels are pitches.
        # If pitch_idx > 0 (Octuple), we look at that column only.
        if pitch_idx > 0:
             # Octuple case (column specific)
             if tokens.shape[1] > pitch_idx:
                pitches = tokens[:, pitch_idx]
                active_per_step = (pitches > 0).sum() / len(pitches) if len(pitches) > 0 else 0
                return active_per_step
        else:
            # Grid case (Melody/Trio) - all cols might be pitches
            # Count how many columns > 0 per step -> average
            active_notes = (tokens > 0).sum(axis=1)
            return active_notes.mean()
    
    elif tokens.ndim == 1:
        # (T,) flattened
        pitches = tokens
        active_per_step = (pitches > 0).sum() / len(pitches) if len(pitches) > 0 else 0
        return active_per_step

    return 0

