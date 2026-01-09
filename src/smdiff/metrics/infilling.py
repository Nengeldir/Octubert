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
    
    # Define indices based on encoding
    if is_octuple:
        # Octuple indices: Bar=0, Pos=1, Prog=2, Pitch=3, Dur=4, Vel=5
        pitch_idx = 3
        duration_idx = 4
        velocity_idx = 5
        bar_idx = 0
    else:
        # Grid/Event-List (Melody/Trio)
        # Shape is (T, 1) or (T, 3). Values are Pitches.
        # No explicit Duration, Velocity, or Bar columns.
        pitch_idx = 0 
        duration_idx = None # Not available
        velocity_idx = None # Not available
        bar_idx = None # Implicit

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
        # SKIP if original region is empty (no ground truth to compare against)
        # This prevents "100% accuracy" when both are empty due to the song ending.
        if len(orig) == 0:
            continue

        # Handle empty generation but non-empty original
        if len(gen) == 0:
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
            
            p_acc = _pitch_accuracy(g_trunc, o_trunc, pitch_idx=pitch_idx)
            pitch_accs.append(p_acc)
            
            # DEBUG: Print first sample mismatch details if accuracy is 0 but we have content
            if len(pitch_accs) == 1:
                print(f"[DEBUG INFILL] Sample 0 comparison:")
                print(f"  Gen len: {len(gen)}, Orig len: {len(orig)}, Min len: {min_len}")
                if min_len > 0:
                     gp = g_trunc[:, pitch_idx]
                     op = o_trunc[:, pitch_idx]
                     match_count = (gp == op).sum()
                     print(f"  Pitch Acc: {p_acc:.2f}% ({match_count}/{min_len})")
                     print(f"  Gen Pitches (first 10): {gp[:10]}")
                     print(f"  Orig Pitches (first 10): {op[:10]}")
                     print(f"  Gen Bar IDs (first 10): {g_trunc[:10, 0]}")
                     print(f"  Orig Bar IDs (first 10): {o_trunc[:10, 0]}")

            if duration_idx is not None:
                duration_accs.append(_duration_accuracy(g_trunc, o_trunc, duration_idx=duration_idx))
            else:
                duration_accs.append(0.0) # Not applicable
            token_accs.append(_token_accuracy(g_trunc, o_trunc))
    
    if pitch_accs:
        metrics['pitch_accuracy'] = np.mean(pitch_accs)
        metrics['pitch_accuracy_std'] = np.std(pitch_accs)
    else:
        metrics['pitch_accuracy'] = 0.0
        metrics['pitch_accuracy_std'] = 0.0

    if duration_accs:
        metrics['duration_accuracy'] = np.mean(duration_accs)
        metrics['duration_accuracy_std'] = np.std(duration_accs)
    else:
        metrics['duration_accuracy'] = 0.0
        metrics['duration_accuracy_std'] = 0.0

    if token_accs:
        metrics['token_accuracy'] = np.mean(token_accs)
        metrics['token_accuracy_std'] = np.std(token_accs)
    else:
        metrics['token_accuracy'] = 0.0
        metrics['token_accuracy_std'] = 0.0
    
    # Musical quality in masked region
    # For Octuple, histograms handle variable length arrays fine
    if gen_masked:
        # Filter out empty arrays for histogram calculation to avoid errors
        gen_valid = [g for g in gen_masked if len(g) > 0]
        orig_valid = [o for o in orig_masked if len(o) > 0]
        
        if gen_valid and orig_valid:
            # Fix: Pass list directly to avoid object array creation issues
            gen_pch = pitch_class_histogram(gen_valid, pitch_idx=pitch_idx)
            orig_pch = pitch_class_histogram(orig_valid, pitch_idx=pitch_idx)
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
        
    # Use pitch_idx for density if available, else use flatten check (all non-zeros)
    def _count_notes(x, p_idx):
        if p_idx is None: return np.count_nonzero(x) # Count all non-zero entries
        if x.ndim > 1 and x.shape[1] > p_idx:
            return len(x[x[:, p_idx] > 0])
        return np.count_nonzero(x)

    gen_densities = [_count_notes(g, pitch_idx) for g in gen_masked]
    orig_densities = [_count_notes(o, pitch_idx) for o in orig_masked]
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
                pitch_smoothness.append(abs(s_ev[pitch_idx] - p_ev[pitch_idx])) 
                # Rhythm smoothness for Octuple is tricky (pos diff?), skipping for now.
        else:
            # Check continuity at start boundary
            if mask_start_step > 0:
                # Use index 0 if pitch_idx is 0 or None (flattened)
                p_idx_eff = pitch_idx if pitch_idx is not None else 0
                
                # Check bounds
                if gen_full.ndim == 1: # (T,)
                    val_curr = gen_full[mask_start_step]
                    val_prev = gen_full[mask_start_step - 1]
                else: # (T, C)
                    val_curr = gen_full[mask_start_step, p_idx_eff]
                    val_prev = gen_full[mask_start_step - 1, p_idx_eff]
                
                pitch_diff_start = abs(val_curr - val_prev)
                pitch_smoothness.append(pitch_diff_start)
                
                if duration_idx is not None and gen_full.ndim > 1:
                    rhythm_diff_start = abs(
                        gen_full[mask_start_step, duration_idx] - gen_full[mask_start_step - 1, duration_idx]
                    )
                    rhythm_smoothness.append(rhythm_diff_start)
            
            # Check continuity at end boundary
            if mask_end_step < len(gen_full):
                p_idx_eff = pitch_idx if pitch_idx is not None else 0
                if gen_full.ndim == 1:
                   val_curr = gen_full[mask_end_step]
                   val_prev = gen_full[mask_end_step - 1]
                else:
                   val_curr = gen_full[mask_end_step, p_idx_eff]
                   val_prev = gen_full[mask_end_step - 1, p_idx_eff]

                pitch_diff_end = abs(val_prev - val_curr)
                pitch_smoothness.append(pitch_diff_end)
                
                if duration_idx is not None and gen_full.ndim > 1:
                    rhythm_diff_end = abs(
                        gen_full[mask_end_step - 1, duration_idx] - gen_full[mask_end_step, duration_idx]
                    )
                    rhythm_smoothness.append(rhythm_diff_end)
    
    metrics['boundary_pitch_smoothness'] = np.mean(pitch_smoothness) if pitch_smoothness else 0.0
    metrics['boundary_rhythm_smoothness'] = np.mean(rhythm_smoothness) if rhythm_smoothness else 0.0
    
    # General quality metrics (full samples)
    if duration_idx is not None:
        self_sims = [compute_self_similarity(s, pitch_idx=pitch_idx, duration_idx=duration_idx) for s in generated_samples]
        metrics['self_similarity'] = np.mean(self_sims)
    else:
        metrics['self_similarity'] = 0.0 # TODO: support pitch-only self-similarity
    
    pitch_ranges = [compute_pitch_range(s, pitch_idx=pitch_idx if pitch_idx is not None else 0) for s in generated_samples]
    metrics['pitch_range_mean'] = np.mean(pitch_ranges)
    
    metrics['sample_diversity'] = compute_sample_diversity(generated_samples, 
                                                         pitch_idx=pitch_idx if pitch_idx is not None else 0, 
                                                         duration_idx=duration_idx if duration_idx is not None else 0)
    
    valid_count = sum([is_valid_sample(s, 
                                      pitch_idx=pitch_idx if pitch_idx is not None else 0, 
                                      duration_idx=duration_idx if duration_idx is not None else 0) for s in generated_samples])
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
