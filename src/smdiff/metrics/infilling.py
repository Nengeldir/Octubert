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


def evaluate_infilling(generated_samples, original_samples, mask_start_step, mask_end_step, is_octuple=True):
    """
    Evaluate infilling quality with reconstruction and boundary metrics.
    
    Args:
        generated_samples: List of (T, C) generated token arrays
        original_samples: List of (T, C) original ground truth arrays
        mask_start_step: Start timestep of masked region (OR start step index for grid)
        mask_end_step: End timestep of masked region (OR end step index for grid)
        is_octuple: Whether samples are Octuple encoded (uses Bar column for masking)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Octuple indices: Bar=0, Pos=1, Prog=2, Pitch=3, Dur=4, Vel=5
    pitch_idx = 3
    duration_idx = 4
    velocity_idx = 5
    bar_idx = 0

    # Extract masked regions
    # Octuple: mask_start_step is treated as BAR index if is_octuple is True
    # User script passes 16 * start_bar. We need to handle that carefully.
    # The caller typically passes raw step count.
    # Let's assume input is in STEPS (standard interface), but for Octuple check Bars.
    # Assuming 16 steps per bar (standard in this repo).
    
    start_bar = mask_start_step // 16
    end_bar = mask_end_step // 16
    
    def extract_region(sample):
        # Column 0 is Bar index
        if sample.ndim != 2 or sample.shape[1] != 8:
            return sample # fallback or empty
        bars = sample[:, 0]
        mask = (bars >= start_bar) & (bars < end_bar)
        
        # Validation: ensure we have columns
        if mask.any():
            return sample[mask]
        else:
            return np.zeros((0, 8), dtype=sample.dtype)
        
    gen_masked = [extract_region(s) for s in generated_samples]
    orig_masked = [extract_region(s) for s in original_samples]
    
    # Reconstruction accuracy metrics (in masked region)
    pitch_accs = []
    duration_accs = []
    token_accs = []
    
    for gen, orig in zip(gen_masked, orig_masked):
        # SKIP if original region is empty (no ground truth)
        if len(orig) == 0:
            continue

        if len(gen) == 0:
            pitch_accs.append(0.0)
            duration_accs.append(0.0)
            token_accs.append(0.0)
        else:
            # Truncate to min length for comparison
            min_len = min(len(gen), len(orig))
            g_trunc = gen[:min_len]
            o_trunc = orig[:min_len]
            
            p_acc = _pitch_accuracy(g_trunc, o_trunc, pitch_idx=pitch_idx)
            pitch_accs.append(p_acc)
            
            d_acc = _duration_accuracy(g_trunc, o_trunc, duration_idx=duration_idx)
            duration_accs.append(d_acc)
            
            t_acc = _token_accuracy(g_trunc, o_trunc)
            token_accs.append(t_acc)
    
    metrics['pitch_accuracy'] = np.mean(pitch_accs) if pitch_accs else 0.0
    metrics['duration_accuracy'] = np.mean(duration_accs) if duration_accs else 0.0
    metrics['token_accuracy'] = np.mean(token_accs) if token_accs else 0.0
    
    # Musical quality in masked region
    # Filter out empty arrays for histogram calculation
    gen_valid = [g for g in gen_masked if len(g) > 0]
    orig_valid = [o for o in orig_masked if len(o) > 0]
    
    if gen_valid and orig_valid:
        gen_pch = pitch_class_histogram(gen_valid, pitch_idx=pitch_idx)
        orig_pch = pitch_class_histogram(orig_valid, pitch_idx=pitch_idx)
        metrics['infilled_pch_kl'] = kl_divergence(orig_pch, gen_pch)
    else:
        metrics['infilled_pch_kl'] = 0.0
    
    # Note density error (notes/bar)
    mask_duration_bars = max(1, end_bar - start_bar)
        
    def _count_notes(x):
        return x.shape[0] # Just count events for now as proxy for notes

    # Compute density per bar
    gen_densities = [_count_notes(g) for g in gen_masked]
    orig_densities = [_count_notes(o) for o in orig_masked]
    
    # Error: difference in n_notes / mask_duration
    density_errors = [abs(g - o) / mask_duration_bars 
                      for g, o in zip(gen_densities, orig_densities)]
    
    metrics['infilled_density_error'] = np.mean(density_errors) if density_errors else 0.0
    
    # Boundary coherence metrics
    # simplified: check pitch distance between last pre-mask event and first mask event
    pitch_smoothness = []
    
    for gen_full in generated_samples:
        if gen_full.ndim != 2 or gen_full.shape[1] != 8:
            continue
            
        bars = gen_full[:, 0]
        # Events just before mask start
        pre_mask = gen_full[bars < start_bar]
        # Events at mask start
        at_mask = gen_full[bars == start_bar]
        
        if len(pre_mask) > 0 and len(at_mask) > 0:
            last_pre = pre_mask[-1]
            first_at = at_mask[0]
            
            p_pre = last_pre[pitch_idx]
            p_at = first_at[pitch_idx]
            
            # Simple absolute pitch difference
            pitch_smoothness.append(abs(p_pre - p_at))
            
    metrics['boundary_pitch_smoothness'] = np.mean(pitch_smoothness) if pitch_smoothness else 0.0
    metrics['boundary_rhythm_smoothness'] = 0.0 # Placeholder
    
    # General quality metrics (full samples)
    if generated_samples:
        pitch_ranges = [compute_pitch_range(s, pitch_idx=pitch_idx) for s in generated_samples]
        metrics['pitch_range_mean'] = np.mean(pitch_ranges) if pitch_ranges else 0.0
        
        metrics['sample_diversity'] = compute_sample_diversity(generated_samples, 
                                                             pitch_idx=pitch_idx, 
                                                             duration_idx=duration_idx)
        
        valid_count = sum([is_valid_sample(s, 
                                          pitch_idx=pitch_idx, 
                                          duration_idx=duration_idx) for s in generated_samples])
        metrics['valid_samples_pct'] = 100.0 * valid_count / len(generated_samples)
    else:
        metrics['pitch_range_mean'] = 0.0
        metrics['sample_diversity'] = 0.0
        metrics['valid_samples_pct'] = 0.0
    
    return metrics


def _pitch_accuracy(generated, original, pitch_idx=3):
    """Compute percentage of matching pitches."""
    gen_pitches = generated[:, pitch_idx]
    orig_pitches = original[:, pitch_idx]
    
    matches = (gen_pitches == orig_pitches).sum()
    total = len(gen_pitches)
    
    return 100.0 * matches / total if total > 0 else 0.0


def _duration_accuracy(generated, original, duration_idx=4):
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
