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
    Evaluate infilling quality with reconstruction and boundary metrics using token index masking.
    
    Args:
        generated_samples: List of (T, C) generated token arrays
        original_samples: List of (T, C) original ground truth arrays
        mask_start_step: Start token index of masked region
        mask_end_step: End token index of masked region
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Octuple indices: Bar=0, Pos=1, Prog=2, Pitch=3, Dur=4, Vel=5
    prog_idx = 2
    pitch_idx = 3
    duration_idx = 4
    velocity_idx = 5
    
    def extract_region(sample):
        # Token-based slicing
        if mask_start_step >= len(sample):
            return np.zeros((0, sample.shape[1]), dtype=sample.dtype) if sample.ndim > 1 else np.zeros((0,), dtype=sample.dtype)
        
        end = min(mask_end_step, len(sample))
        if end <= mask_start_step:
            return np.zeros((0, sample.shape[1]), dtype=sample.dtype) if sample.ndim > 1 else np.zeros((0,), dtype=sample.dtype)
            
        return sample[mask_start_step:end]
        
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
    
    # Note count error (Total events difference in masked region)
    def _count_notes(x):
        return x.shape[0]

    gen_counts = [_count_notes(g) for g in gen_masked]
    orig_counts = [_count_notes(o) for o in orig_masked]
    
    count_errors = [abs(g - o) for g, o in zip(gen_counts, orig_counts)]
    metrics['infilled_count_error'] = np.mean(count_errors) if count_errors else 0.0
    
    # Boundary coherence metrics
    # Improved: check pitch/duration distance between last pre-mask event and first mask event
    pitch_smoothness = []
    rhythm_smoothness = []
    
    boundary_checks = 0
    boundary_matches = 0
    
    for gen_full in generated_samples:
        if gen_full.ndim != 2:
            continue
            
        at_idx = mask_start_step
        
        if at_idx < len(gen_full) and at_idx > 0:
            boundary_checks += 1
            # Instrument at the start of the mask
            inst_at = gen_full[at_idx, prog_idx]
            
            # Search backwards for the most recent event of the same instrument
            # Scan from mask_start_step - 1 down to 0
            pre_idx = -1
            for i in range(at_idx - 1, -1, -1):
                if gen_full[i, prog_idx] == inst_at:
                    pre_idx = i
                    break
            
            if pre_idx != -1:
                boundary_matches += 1
                p_pre = gen_full[pre_idx, pitch_idx]
                p_at = gen_full[at_idx, pitch_idx]
                
                d_pre = gen_full[pre_idx, duration_idx]
                d_at = gen_full[at_idx, duration_idx]
                
                pitch_smoothness.append(abs(p_pre - p_at))
                rhythm_smoothness.append(abs(d_pre - d_at))
            
    metrics['boundary_pitch_smoothness'] = np.mean(pitch_smoothness) if pitch_smoothness else 0.0
    metrics['boundary_rhythm_smoothness'] = np.mean(rhythm_smoothness) if rhythm_smoothness else 0.0
    metrics['boundary_matches_pct'] = 100.0 * boundary_matches / boundary_checks if boundary_checks > 0 else 0.0
    
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
