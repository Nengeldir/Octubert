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


def evaluate_infilling(generated_samples, original_samples, mask_start_step, mask_end_step, is_octuple=False):
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
    
    # Extract masked regions
    if is_octuple:
        # Convert step indices back to bars (assuming 16 steps/bar)
        start_bar = mask_start_step // 16
        end_bar = mask_end_step // 16
        
        def extract_region(sample):
            # Column 0 is Bar index
            if sample.ndim != 2 or sample.shape[1] != 8:
                return sample # fallback
            bars = sample[:, 0]
            mask = (bars >= start_bar) & (bars < end_bar)
            return sample[mask]
            
        gen_masked = [extract_region(s) for s in generated_samples]
        orig_masked = [extract_region(s) for s in original_samples]
    else:
        # Standard Grid/Image slicing
        gen_masked = [s[mask_start_step:mask_end_step] for s in generated_samples]
        orig_masked = [s[mask_start_step:mask_end_step] for s in original_samples]
    
    # Reconstruction accuracy metrics (in masked region)
    pitch_accs = []
    duration_accs = []
    token_accs = []
    
    for gen, orig in zip(gen_masked, orig_masked):
        # Handle empty regions (if no notes in masked bars)
        if len(gen) == 0 and len(orig) == 0:
            # Perfect match if both empty
            pitch_accs.append(100.0)
            duration_accs.append(100.0)
            token_accs.append(100.0)
        elif len(gen) == 0 or len(orig) == 0:
            # Mismatch if one is empty
            pitch_accs.append(0.0)
            duration_accs.append(0.0)
            token_accs.append(0.0)
        else:
            # For Octuple, lengths might differ due to generation. 
            # We can only compute accuracy on the min length (or treat length diff as error).
            # Here we follow standard practice: truncation to min length for strict comparison.
            min_len = min(len(gen), len(orig))
            g_trunc = gen[:min_len]
            o_trunc = orig[:min_len]
            
            pitch_accs.append(_pitch_accuracy(g_trunc, o_trunc))
            duration_accs.append(_duration_accuracy(g_trunc, o_trunc))
            token_accs.append(_token_accuracy(g_trunc, o_trunc))
    
    metrics['pitch_accuracy'] = np.mean(pitch_accs)
    metrics['pitch_accuracy_std'] = np.std(pitch_accs)
    metrics['duration_accuracy'] = np.mean(duration_accs)
    metrics['duration_accuracy_std'] = np.std(duration_accs)
    metrics['token_accuracy'] = np.mean(token_accs)
    metrics['token_accuracy_std'] = np.std(token_accs)
    
    # Musical quality in masked region
    # For Octuple, histograms handle variable length arrays fine
    if gen_masked:
        # Filter out empty arrays for histogram calculation to avoid errors
        gen_valid = [g for g in gen_masked if len(g) > 0]
        orig_valid = [o for o in orig_masked if len(o) > 0]
        
        if gen_valid and orig_valid:
            gen_pch = pitch_class_histogram(np.array(gen_valid, dtype=object) if is_octuple else np.array(gen_valid))
            orig_pch = pitch_class_histogram(np.array(orig_valid, dtype=object) if is_octuple else np.array(orig_valid))
            metrics['infilled_pch_kl'] = kl_divergence(orig_pch, gen_pch)
        else:
             metrics['infilled_pch_kl'] = 0.0
    else:
        metrics['infilled_pch_kl'] = 0.0
    
    # Note density error (notes/bar)
    # Mask duration in bars:
    if is_octuple:
        mask_duration_bars = max(1, end_bar - start_bar)
    else:
        # Assuming 16 steps/bar for simple normalization, or just use raw count
        mask_duration_bars = (mask_end_step - mask_start_step) / 16.0
        
    gen_densities = [len(g[g[:, 2] > 0]) for g in gen_masked]
    orig_densities = [len(o[o[:, 2] > 0]) for o in orig_masked]
    density_errors = [abs(g - o) / mask_duration_bars 
                      for g, o in zip(gen_densities, orig_densities)]
    metrics['infilled_density_error'] = np.mean(density_errors)
    
    # Boundary coherence metrics
    pitch_smoothness = []
    rhythm_smoothness = []
    
    for gen_full, orig_full in zip(generated_samples, original_samples):
        if is_octuple:
            # Find boundary events by bar/time, not index
            start_events = gen_full[gen_full[:, 0] == start_bar]
            # Look for events right before start_bar
            pre_events = gen_full[gen_full[:, 0] < start_bar]
            
            # Simple heuristic: compare last event before mask and first event in mask
            if len(start_events) > 0 and len(pre_events) > 0:
                s_ev = start_events[0]
                p_ev = pre_events[-1]
                pitch_smoothness.append(abs(s_ev[3] - p_ev[3])) # Pitch col is 3 in Octuple [bar, pos, prog, pitch...]
                # Rhythm smoothness for Octuple is tricky (pos diff?), skipping for now.
        else:
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
