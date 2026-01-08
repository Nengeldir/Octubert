"""Metrics for infilling evaluation."""
import numpy as np
from .common import (
    kl_divergence,
    pitch_class_histogram,
    duration_histogram,
    compute_self_similarity,
    compute_pitch_range,
    compute_sample_diversity,
    is_valid_sample
)


def evaluate_infilling(generated_samples, original_samples, mask_start_step, mask_end_step):
    """
    Evaluate infilling quality with reconstruction and boundary metrics.
    
    Args:
        generated_samples: List of (T, C) generated token arrays
        original_samples: List of (T, C) original ground truth arrays
        mask_start_step: Start timestep of masked region
        mask_end_step: End timestep of masked region
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Extract masked regions
    gen_masked = [s[mask_start_step:mask_end_step] for s in generated_samples]
    orig_masked = [s[mask_start_step:mask_end_step] for s in original_samples]
    
    # Reconstruction accuracy metrics (in masked region)
    pitch_accs = []
    duration_accs = []
    token_accs = []
    
    for gen, orig in zip(gen_masked, orig_masked):
        pitch_accs.append(_pitch_accuracy(gen, orig))
        duration_accs.append(_duration_accuracy(gen, orig))
        token_accs.append(_token_accuracy(gen, orig))
    
    metrics['pitch_accuracy'] = np.mean(pitch_accs)
    metrics['pitch_accuracy_std'] = np.std(pitch_accs)
    metrics['duration_accuracy'] = np.mean(duration_accs)
    metrics['duration_accuracy_std'] = np.std(duration_accs)
    metrics['token_accuracy'] = np.mean(token_accs)
    metrics['token_accuracy_std'] = np.std(token_accs)
    
    # Musical quality in masked region
    gen_pch = pitch_class_histogram(np.array(gen_masked))
    orig_pch = pitch_class_histogram(np.array(orig_masked))
    metrics['infilled_pch_kl'] = kl_divergence(orig_pch, gen_pch)
    
    # Note density error
    gen_densities = [len(g[g[:, 2] > 0]) for g in gen_masked]
    orig_densities = [len(o[o[:, 2] > 0]) for o in orig_masked]
    density_errors = [abs(g - o) / (mask_end_step - mask_start_step) 
                      for g, o in zip(gen_densities, orig_densities)]
    metrics['infilled_density_error'] = np.mean(density_errors)
    
    # Boundary coherence metrics
    pitch_smoothness = []
    rhythm_smoothness = []
    
    for gen_full, orig_full in zip(generated_samples, original_samples):
        # Check continuity at start boundary
        if mask_start_step > 0:
            pitch_diff_start = abs(
                gen_full[mask_start_step, 2] - gen_full[mask_start_step - 1, 2]
            )
            pitch_smoothness.append(pitch_diff_start)
            
            rhythm_diff_start = abs(
                gen_full[mask_start_step, 3] - gen_full[mask_start_step - 1, 3]
            )
            rhythm_smoothness.append(rhythm_diff_start)
        
        # Check continuity at end boundary
        if mask_end_step < len(gen_full):
            pitch_diff_end = abs(
                gen_full[mask_end_step - 1, 2] - gen_full[mask_end_step, 2]
            )
            pitch_smoothness.append(pitch_diff_end)
            
            rhythm_diff_end = abs(
                gen_full[mask_end_step - 1, 3] - gen_full[mask_end_step, 3]
            )
            rhythm_smoothness.append(rhythm_diff_end)
    
    metrics['boundary_pitch_smoothness'] = np.mean(pitch_smoothness) if pitch_smoothness else 0.0
    metrics['boundary_rhythm_smoothness'] = np.mean(rhythm_smoothness) if rhythm_smoothness else 0.0
    
    # General quality metrics (full samples)
    self_sims = [compute_self_similarity(s) for s in generated_samples]
    metrics['self_similarity'] = np.mean(self_sims)
    
    pitch_ranges = [compute_pitch_range(s) for s in generated_samples]
    metrics['pitch_range_mean'] = np.mean(pitch_ranges)
    
    metrics['sample_diversity'] = compute_sample_diversity(generated_samples)
    
    valid_count = sum([is_valid_sample(s) for s in generated_samples])
    metrics['valid_samples_pct'] = 100.0 * valid_count / len(generated_samples)
    
    return metrics


def _pitch_accuracy(generated, original, pitch_idx=2):
    """Compute percentage of matching pitches."""
    gen_pitches = generated[:, pitch_idx]
    orig_pitches = original[:, pitch_idx]
    
    matches = (gen_pitches == orig_pitches).sum()
    total = len(gen_pitches)
    
    return 100.0 * matches / total if total > 0 else 0.0


def _duration_accuracy(generated, original, duration_idx=3):
    """Compute percentage of matching durations."""
    gen_durations = generated[:, duration_idx]
    orig_durations = original[:, duration_idx]
    
    matches = (gen_durations == orig_durations).sum()
    total = len(gen_durations)
    
    return 100.0 * matches / total if total > 0 else 0.0


def _token_accuracy(generated, original):
    """Compute percentage of exactly matching tokens (all 8 attributes)."""
    matches = (generated == original).all(axis=1).sum()
    total = len(generated)
    
    return 100.0 * matches / total if total > 0 else 0.0
