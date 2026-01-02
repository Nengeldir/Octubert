"""Structure-aware evaluation runner for unconditional generation and infilling.

Metrics (saved as JSON) and why they matter:
    - pitch_class_js: Jensen-Shannon (JS) divergence between pitch-class histograms; lower is closer to the reference pitch distribution.
    - duration_js: JS divergence between duration bins; lower means rhythm/duration distribution matches reference.
    - consistency/variance: framewise overlap of pitch/duration distributions (OA metric) to capture local stability and variety.
    - self_similarity_mean: bar-to-bar repetition/coherence proxy via cosine similarity; higher suggests structural repetition.
    - self_similarity_gap_to_ref: difference vs. reference to show over/under repetition.
    - key_consistency / key_agreement_with_ref: dominant pitch-class per bar; higher means stable key and agreement with reference set.
    - bar_pitch_var_mean / bar_onset_density_mean: quick bar-level structure summary (pitch spread and density of onsets).
    - masked_token_accuracy (infilling): exact token match on the masked span; higher is better reconstruction.
    - structural_fad (optional): Frechet distance (FAD) over simple structural embeddings; lower indicates closer structural statistics.

Supports masking ablation (random vs. bar-aligned) to see if structure-aware masking helps infilling.
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch

from hparams import get_sampler_hparams
from utils import get_sampler, load_model, log
from utils.eval_utils import evaluate_consistency_variance
from utils.sampler_utils import get_samples, np_to_ns, ns_to_np
from utils.data_utils import SubseqSampler
from utils.split_utils import load_split_ids, load_processed_subset


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """Jensen-Shannon divergence: symmetric, finite KL-style distance between two histograms."""
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def pitch_class_hist(tensors: np.ndarray) -> np.ndarray:
    """Count pitch classes (mod 12). Captures harmony/tonal balance."""
    pcs = np.zeros(12, dtype=np.float64)
    pitches = tensors[..., 0].ravel()
    pitches = pitches[(pitches > 1) & (pitches < 90)]
    if pitches.size:
        pcs += np.bincount(pitches % 12, minlength=12)
    return pcs


def duration_hist(tensors: np.ndarray) -> np.ndarray:
    """Histogram of durations (quantized). Captures rhythm distribution."""
    # durations are stored in channel 1 for trio; for melody only channel 0 exists.
    if tensors.shape[-1] > 1:
        durs = tensors[..., 1].ravel()
    else:
        durs = tensors[..., 0].ravel() * 0  # no duration channel; fallback zeros
    durs = durs[(durs > 0) & (durs < 32)]
    return np.bincount(durs, minlength=32)


def masked_token_accuracy(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """Exact-match accuracy on masked span; basic infilling fidelity metric."""
    masked_ref = ref[mask]
    masked_pred = pred[mask]
    if masked_ref.size == 0:
        return float('nan')
    return float((masked_ref == masked_pred).mean())


def get_reference_subset(H, dataset: np.ndarray, eval_batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a reference subset for metrics; returns subset and full dataset for shape info."""
    midi_data = SubseqSampler(dataset, H.NOTES)
    # Cap eval_batch_size at available data
    eval_batch_size = min(eval_batch_size, midi_data.dataset.shape[0])
    idx = np.random.choice(midi_data.dataset.shape[0], eval_batch_size)
    refs = midi_data[idx]
    return refs, midi_data.dataset


def apply_octuple_mask_np(x_0: np.ndarray, strategy: str) -> np.ndarray:
    """Apply partial-masking strategies on octuple tensors (B, L, 8); returns boolean mask."""
    b, seq_len, _ = x_0.shape
    mask = np.zeros_like(x_0, dtype=bool)

    bar_indices = x_0[:, :, 0]
    max_bars = bar_indices.max(axis=1)

    rng = np.random.default_rng()

    def sample_bars(count):
        return (rng.random(b) * (max_bars.astype(np.float64) + 1)).astype(int)

    if strategy == '1_bar_all':
        target_bars = sample_bars(1)
        target_attr = np.array([3, 4, 5, 7])
        for i in range(b):
            bar_mask = bar_indices[i] == target_bars[i]
            mask[i, bar_mask, :][:, target_attr] = True
    elif strategy == '2_bar_all':
        r1, r2 = sample_bars(1), sample_bars(1)
        target_attr = np.array([3, 4, 5, 7])
        for i in range(b):
            bar_mask = (bar_indices[i] == r1[i]) | (bar_indices[i] == r2[i])
            mask[i, bar_mask, :][:, target_attr] = True
    elif strategy == '1_bar_attribute':
        target_bars = sample_bars(1)
        avail_attrs = np.array([3, 4, 5, 7])
        sel_attr = avail_attrs[(rng.random(b) * 4).astype(int)]
        for i in range(b):
            bar_mask = bar_indices[i] == target_bars[i]
            mask[i, bar_mask, sel_attr[i]] = True
    elif strategy == '2_bar_attribute':
        r1, r2 = sample_bars(1), sample_bars(1)
        avail_attrs = np.array([3, 4, 5, 7])
        sel_attr = avail_attrs[(rng.random(b) * 4).astype(int)]
        for i in range(b):
            bar_mask = (bar_indices[i] == r1[i]) | (bar_indices[i] == r2[i])
            mask[i, bar_mask, sel_attr[i]] = True
    elif strategy == 'rand_attribute':
        avail_attrs = np.array([3, 4, 5, 7])
        sel_attr = avail_attrs[(rng.random(b) * 4).astype(int)]
        for i in range(b):
            mask[i, :, sel_attr[i]] = True
    else:
        raise ValueError(f"Unknown octuple masking strategy: {strategy}")

    return mask


def run_generation(H, sampler, dataset: np.ndarray, n_samples: int, mode: str, gap: Tuple[int, int], mask_tracks) -> Tuple[np.ndarray, np.ndarray]:
    """Generate samples (unconditional or infilling) alongside reference slices."""
    # Default eval_batch_size if not set (for sample mode)
    eval_batch_size = getattr(H, 'eval_batch_size', None) or max(64, n_samples)
    refs, full_dataset = get_reference_subset(H, dataset, max(eval_batch_size, n_samples))
    batch_size = H.sampling_batch_size

    log(f"[gen] mode={mode} n_samples={n_samples} batch_size={batch_size} sample_steps={H.sample_steps}")

    if mode == "unconditional":
        outs = []
        total = 0
        while total < n_samples:
            b = min(batch_size, n_samples - total)
            sampler.sampling_batch_size = b
            sa = get_samples(sampler, sample_steps=H.sample_steps, temp=H.temp)
            outs.append(sa)
            total += b
        samples = np.concatenate(outs, axis=0)
    elif mode == "infilling":
        # start from refs, apply mask, then sample
        samples = refs.copy()

        if getattr(H, 'model', None) == 'octuple' and getattr(H, 'masking_strategy', None):
            log(f"[gen] applying octuple masking_strategy={H.masking_strategy}")
            mask = apply_octuple_mask_np(samples, H.masking_strategy)
            mask_id = getattr(sampler, 'mask_id', None)
            if mask_id is None:
                raise ValueError("sampler.mask_id not set for octuple model")
            mask_id_np = np.array(mask_id.cpu().numpy() if hasattr(mask_id, 'cpu') else mask_id)
            for i in range(samples.shape[2]):
                samples[:, :, i][mask[:, :, i]] = int(mask_id_np[i])
        elif mask_tracks:
            for t in mask_tracks:
                samples[:, gap[0]:gap[1], t] = H.codebook_size[t]
        else:
            # For non-octuple models: mask the gap span with codebook_size
            # codebook_size is a tuple like (90,) for melody or (90, 90, 512) for trio
            if len(H.codebook_size) == 1:
                # Single-channel (melody): fill gap with codebook_size[0]
                samples[:, gap[0]:gap[1]] = H.codebook_size[0]
            else:
                # Multi-channel (trio): fill each channel with its codebook_size
                for c in range(len(H.codebook_size)):
                    samples[:, gap[0]:gap[1], c] = H.codebook_size[c]
        outs = []
        total = 0
        while total < n_samples:
            l = total
            u = min(samples.shape[0], l + batch_size)
            outs.append(get_samples(sampler, sample_steps=H.sample_steps, x_T=samples[l:u], temp=H.temp))
            total = u
        samples = np.concatenate(outs, axis=0)
    else:
        raise ValueError("mode must be unconditional or infilling")

    refs = refs[:n_samples]
    samples = samples[:n_samples]
    log(f"[gen] done: samples={samples.shape} refs={refs.shape}")
    return samples, refs


# --- Structure-focused helpers (defined before compute_metrics) ---


def split_bars(tensors: np.ndarray, steps_per_bar: int = 16) -> np.ndarray:
    """Reshape sequences into [B, bars, steps_per_bar, C] for bar-level analysis."""
    # tensors: [B, T, C]
    T = tensors.shape[1]
    n_bars = T // steps_per_bar
    trimmed = tensors[:, : n_bars * steps_per_bar]
    return trimmed.reshape(trimmed.shape[0], n_bars, steps_per_bar, tensors.shape[-1])


def bar_pitch_vectors(tensors: np.ndarray) -> np.ndarray:
    """Bar-level pitch-class vectors; basis for repetition and key proxies."""
    bars = split_bars(tensors)
    B, NB, _, C = bars.shape
    vecs = np.zeros((B, NB, 12), dtype=np.float64)
    pitches = bars[..., 0]
    mask = (pitches > 1) & (pitches < 90)
    for b in range(B):
        for j in range(NB):
            pj = pitches[b, j][mask[b, j]]
            if pj.size:
                vecs[b, j] = np.bincount(pj % 12, minlength=12)
    return vecs


def self_similarity_score(tensors: np.ndarray) -> float:
    """Average cosine similarity of consecutive bars (higher = more repetition/structure)."""
    vecs = bar_pitch_vectors(tensors)
    sims = []
    for v in vecs:
        if v.shape[0] < 2:
            continue
        # cosine similarities between consecutive bars (repetition/coherence proxy)
        a = v[:-1]
        b = v[1:]
        denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
        s = (a * b).sum(axis=1) / denom
        sims.append(s.mean())
    if not sims:
        return float('nan')
    return float(np.mean(sims))


def key_consistency(tensors: np.ndarray) -> Tuple[float, int]:
    """Dominant pitch class per bar; returns stability score and global mode key."""
    vecs = bar_pitch_vectors(tensors)
    keys = np.argmax(vecs, axis=2)  # dominant pitch class per bar
    cons = []
    mode_keys = []
    for ks in keys:
        values, counts = np.unique(ks, return_counts=True)
        mode = values[np.argmax(counts)]
        mode_keys.append(mode)
        cons.append((ks == mode).mean())
    if not cons:
        return float('nan'), -1
    return float(np.mean(cons)), int(np.bincount(mode_keys, minlength=12).argmax())


def bar_level_summaries(tensors: np.ndarray) -> Tuple[float, float]:
    """Mean pitch variance and onset density per bar; simple structure indicators."""
    bars = split_bars(tensors)
    pitches = bars[..., 0]
    onset = (pitches > 1).astype(np.float32)
    pitch_var = pitches.astype(np.float32)
    pitch_var[pitch_var <= 1] = np.nan

    # Only compute variance for bars with at least two finite pitches to avoid deg-of-freedom warnings.
    flat = pitch_var.reshape(-1, pitch_var.shape[-1])
    valid = np.sum(np.isfinite(flat), axis=1) >= 2
    if valid.any():
        bar_var = np.full(flat.shape[0], np.nan, dtype=np.float32)
        with np.errstate(invalid='ignore'):
            bar_var[valid] = np.nanvar(flat[valid], axis=1)
        bar_var = bar_var.reshape(pitch_var.shape[:2])
        pitch_var_mean = float(np.nanmean(bar_var))
    else:
        pitch_var_mean = float('nan')

    onset_density = onset.mean(axis=2).mean()
    return float(pitch_var_mean), float(onset_density)


def phrase_level_self_similarity(tensors: np.ndarray, phrase_bars: int = 4) -> float:
    """Cosine similarity between consecutive phrases (multi-bar chunks) to capture long-range structure.
    
    Higher values indicate repeated phrase-level motifs (e.g., verse/chorus structure).
    """
    vecs = bar_pitch_vectors(tensors)
    sims = []
    
    for v in vecs:
        n_bars = v.shape[0]
        if n_bars < phrase_bars * 2:
            continue
        
        # Aggregate bars into phrases
        n_phrases = n_bars // phrase_bars
        phrases = []
        for p in range(n_phrases):
            phrase_vec = v[p * phrase_bars: (p + 1) * phrase_bars].sum(axis=0)
            phrases.append(phrase_vec)
        
        phrases = np.array(phrases)
        if len(phrases) < 2:
            continue
        
        # Consecutive phrase similarity
        a = phrases[:-1]
        b = phrases[1:]
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
        s = (a * b).sum(axis=1) / norms
        sims.append(s.mean())
    
    return float(np.mean(sims)) if sims else float('nan')


def harmonic_flux(tensors: np.ndarray) -> float:
    """Measure harmonic change rate by tracking pitch-class centroid shifts per bar.
    
    Higher flux = more harmonic movement; lower = stable harmony. Typical range 0.2–0.6.
    """
    vecs = bar_pitch_vectors(tensors)
    fluxes = []
    
    for v in vecs:
        if v.shape[0] < 2:
            continue
        
        # Compute centroids (weighted pitch-class mean)
        centroids = []
        for bar_vec in v:
            total = bar_vec.sum()
            if total > 0:
                centroid = np.dot(bar_vec, np.arange(12)) / total
                centroids.append(centroid)
            else:
                centroids.append(0.0)
        
        centroids = np.array(centroids)
        if len(centroids) < 2:
            continue
        
        # Flux = average absolute difference between consecutive centroids (normalized)
        diffs = np.abs(np.diff(centroids))
        # Normalize by wrapping around octave (12 semitones)
        diffs = np.minimum(diffs, 12 - diffs)
        fluxes.append(diffs.mean() / 12.0)  # normalize to [0, 1]
    
    return float(np.mean(fluxes)) if fluxes else float('nan')


def infill_rhythm_correlation(samples: np.ndarray, gap: Tuple[int, int]) -> float:
    """Compute correlation of onset density between masked span and surrounding context.
    
    Higher correlation suggests better rhythmic continuity between infill and context.
    """
    bars = split_bars(samples)
    steps_per_bar = 16
    gap_bar_start = gap[0] // steps_per_bar
    gap_bar_end = gap[1] // steps_per_bar
    
    correlations = []
    for b in range(bars.shape[0]):
        pitches = bars[b, :, :, 0]
        onset = (pitches > 1).astype(np.float32).mean(axis=1)  # onset density per bar
        
        if gap_bar_start > 0 and gap_bar_end < len(onset):
            context = np.concatenate([onset[:gap_bar_start], onset[gap_bar_end:]])
            infill = onset[gap_bar_start:gap_bar_end]
            
            if len(context) > 1 and len(infill) > 1:
                # Correlate infill bars with the first |infill| context bars for rhythmic continuity
                ctx = context[:len(infill)]
                if ctx.shape[0] == infill.shape[0]:
                    infill_std = np.std(infill)
                    ctx_std = np.std(ctx)
                    if infill_std > 0 and ctx_std > 0:
                        corr = np.corrcoef(infill, ctx)[0, 1]
                        if np.isfinite(corr):
                            correlations.append(corr)
    
    return float(np.mean(correlations)) if correlations else float('nan')


def masked_span_embedding_similarity(refs: np.ndarray, samples: np.ndarray, gap: Tuple[int, int]) -> float:
    """Cosine similarity of bar pitch vectors between reference and generated masked span.
    
    Higher similarity indicates better reconstruction of the masked region's harmonic content.
    """
    steps_per_bar = 16
    gap_bar_start = gap[0] // steps_per_bar
    gap_bar_end = gap[1] // steps_per_bar
    
    ref_vecs = bar_pitch_vectors(refs)
    sample_vecs = bar_pitch_vectors(samples)
    
    similarities = []
    for b in range(ref_vecs.shape[0]):
        ref_span = ref_vecs[b, gap_bar_start:gap_bar_end]
        sample_span = sample_vecs[b, gap_bar_start:gap_bar_end]
        
        for i in range(len(ref_span)):
            r = ref_span[i]
            s = sample_span[i]
            norm = np.linalg.norm(r) * np.linalg.norm(s)
            if norm > 1e-8:
                sim = (r @ s) / norm
                similarities.append(sim)
    
    return float(np.mean(similarities)) if similarities else float('nan')


def structural_embedding(tensors: np.ndarray) -> np.ndarray:
    """Cheap structural embedding: normalized pitch-class histogram + onset stats per sample."""
    # Cheap structural embedding: bar pitch class histogram + onset density per bar, averaged and flattened
    bars = split_bars(tensors)
    B, NB, _, _ = bars.shape
    vecs = []
    for b in range(B):
        pitch_vec = np.zeros(12)
        onset_vec = []
        for j in range(NB):
            pj = bars[b, j, :, 0]
            mask = (pj > 1) & (pj < 90)
            pitch_vec += np.bincount(pj[mask] % 12, minlength=12)
            onset_vec.append(mask.mean())
        onset_vec = np.array(onset_vec)
        pitch_vec = pitch_vec / (pitch_vec.sum() + 1e-8)
        vecs.append(np.concatenate([pitch_vec, onset_vec.mean(keepdims=True), onset_vec.var(keepdims=True)]))
    return np.stack(vecs)


def structural_fad(samples: np.ndarray, refs: np.ndarray) -> float:
    """Frechet distance (FAD) over structural embeddings; lower means closer structure.

    This is a lightweight stand-in for MusicBERT-style embedding FAD when heavy encoders
    are unavailable on the evaluation machine.
    """
    def stats(x):
        mu = x.mean(axis=0)
        sigma = np.cov(x, rowvar=False)
        return mu, sigma

    def frechet(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        # simple trace approximation to avoid sqrt of matrices for speed
        trace = np.trace(sigma1 + sigma2 - 2 * np.sqrt((sigma1 @ sigma2) + 1e-8))
        return float(diff @ diff + trace)

    emb_s = structural_embedding(samples)
    emb_r = structural_embedding(refs)
    mu_s, sigma_s = stats(emb_s)
    mu_r, sigma_r = stats(emb_r)
    return frechet(mu_s, sigma_s, mu_r, sigma_r)


def compute_metrics(samples: np.ndarray, refs: np.ndarray, mode: str, gap: Tuple[int, int], samples_ns=None, refs_ns=None) -> Dict:
    """Compute distributional + structural metrics and infill accuracy.

    JS divergence: distribution match; lower is better.
    Consistency/variance: local stability/variety (OA overlap).
    Self-similarity: repetition/structure; closer to reference is desirable.
    Key consistency: tonal stability and agreement.
    Bar summaries: quick view of pitch spread and density (structure proxy).
    Masked token accuracy: exact reconstruction of masked span.
    
    Args:
        samples_ns, refs_ns: Optional precomputed NoteSequences to avoid repeated conversion (for bootstrap speed).
    """
    metrics: Dict[str, float] = {}

    pcs_s = pitch_class_hist(samples)
    pcs_r = pitch_class_hist(refs)
    metrics["pitch_class_js"] = js_divergence(pcs_s, pcs_r)

    dur_s = duration_hist(samples)
    dur_r = duration_hist(refs)
    metrics["duration_js"] = js_divergence(dur_s, dur_r)

    # Use precomputed NoteSequences if provided (bootstrap speedup)
    if samples_ns is None:
        samples_ns = np_to_ns(samples)
    if refs_ns is None:
        refs_ns = np_to_ns(refs)
    cons, var = evaluate_consistency_variance(refs_ns, samples_ns)
    metrics["consistency_pitch"] = float(cons[0])
    metrics["consistency_duration"] = float(cons[1])
    metrics["variance_pitch"] = float(var[0])
    metrics["variance_duration"] = float(var[1])

    # Structure: bar self-similarity and repetition
    sim_s = self_similarity_score(samples)
    sim_r = self_similarity_score(refs)
    metrics["self_similarity_mean"] = sim_s
    metrics["self_similarity_gap_to_ref"] = sim_s - sim_r

    # Key / tonal consistency (very lightweight estimator using pitch-class mode per bar)
    key_cons_s, key_mode_s = key_consistency(samples)
    key_cons_r, key_mode_r = key_consistency(refs)
    metrics["key_consistency"] = key_cons_s
    metrics["key_consistency_gap_to_ref"] = key_cons_s - key_cons_r
    metrics["key_agreement_with_ref"] = 1.0 if key_mode_s == key_mode_r else 0.0

    # Bar-level rhythm/pitch summaries for quick inspection
    metrics["bar_pitch_var_mean"], metrics["bar_onset_density_mean"] = bar_level_summaries(samples)
    
    # Long-range structure metrics
    phrase_sim_s = phrase_level_self_similarity(samples)
    phrase_sim_r = phrase_level_self_similarity(refs)
    metrics["phrase_similarity_mean"] = phrase_sim_s
    metrics["phrase_similarity_gap_to_ref"] = phrase_sim_s - phrase_sim_r
    
    flux_s = harmonic_flux(samples)
    flux_r = harmonic_flux(refs)
    metrics["harmonic_flux_mean"] = flux_s
    metrics["harmonic_flux_gap_to_ref"] = flux_s - flux_r

    if mode == "infilling":
        mask = np.zeros_like(refs, dtype=bool)
        mask[:, gap[0]:gap[1]] = True
        acc = masked_token_accuracy(refs, samples, mask)
        metrics["masked_token_accuracy"] = acc
        
        # Infill continuity: rhythm correlation between masked span and context
        rhythm_corr = infill_rhythm_correlation(samples, gap)
        metrics["infill_rhythm_correlation"] = rhythm_corr
        
        # Masked-span embedding similarity (bar pitch vectors cosine)
        span_sim = masked_span_embedding_similarity(refs, samples, gap)
        metrics["masked_span_embedding_similarity"] = span_sim

    return metrics


def bootstrap_confidence_intervals(samples: np.ndarray, refs: np.ndarray, mode: str, gap: Tuple[int, int], n_bootstrap: int = 1000) -> Dict:
    """Compute bootstrap 95% confidence intervals for all metrics.
    
    Returns dict with keys like 'pitch_class_js_ci' containing (lower, upper) bounds.
    """
    n_samples = samples.shape[0]
    
    # Precompute NoteSequences once to avoid repeated conversion (huge speedup for octuple)
    log("[bootstrap] precomputing NoteSequences for consistency/variance metrics...")
    all_samples_ns = np_to_ns(samples)
    all_refs_ns = np_to_ns(refs)
    log(f"[bootstrap] NoteSequence conversion complete; starting {n_bootstrap} iterations")
    
    bootstrap_metrics = []
    
    log_interval = max(1, n_bootstrap // 10)
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)
        boot_samples = samples[idx]
        boot_refs = refs[idx]
        boot_samples_ns = [all_samples_ns[j] for j in idx]
        boot_refs_ns = [all_refs_ns[j] for j in idx]
        
        boot_m = compute_metrics(boot_samples, boot_refs, mode, gap, boot_samples_ns, boot_refs_ns)
        bootstrap_metrics.append(boot_m)

        if (i + 1) % log_interval == 0:
            log(f"[bootstrap] {i + 1}/{n_bootstrap} completed")
    
    # Compute 95% CIs (2.5th and 97.5th percentiles)
    ci_dict = {}
    all_keys = bootstrap_metrics[0].keys()
    
    for key in all_keys:
        values = [m[key] for m in bootstrap_metrics if key in m and np.isfinite(m[key])]
        if values:
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            ci_dict[f"{key}_ci"] = [float(lower), float(upper)]
            ci_dict[f"{key}_std"] = float(np.std(values))
    
    return ci_dict


def save_samples_midi(samples: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, ns in enumerate(np_to_ns(samples)):
        from note_seq import note_sequence_to_midi_file

        path = os.path.join(out_dir, f"sample_{i}.mid")
        note_sequence_to_midi_file(ns, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["unconditional", "infilling"], default="unconditional")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--gap_start", type=int, default=-1)
    parser.add_argument("--gap_end", type=int, default=-1)
    parser.add_argument("--mask_tracks", nargs="*", type=int, default=[])
    parser.add_argument("--masking_strategy", type=str, default=None, help="Octuple partial-masking strategy (e.g., 1_bar_all, 2_bar_all, 1_bar_attribute, 2_bar_attribute, rand_attribute, mixed)")
    parser.add_argument("--save_midis", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--compute_fad", action="store_true", help="Compute structural FAD (cheap embedding)")
    parser.add_argument("--ablate_masking", action="store_true", help="Run bar-aligned masking as second eval for infilling")
    parser.add_argument("--bootstrap_ci", action="store_true", help="Compute bootstrap 95%% confidence intervals (slow)")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap samples for CIs")
    parser.add_argument("--bootstrap_only", action="store_true", help="Skip generation; load existing samples/refs and compute bootstrap CIs only")
    parser.add_argument("--split_path", type=str, default="data/splits/pop909_split.json")
    parser.add_argument("--split_partition", choices=["train", "val", "test"], default="test")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--no_split", action="store_true", help="Ignore split file and use raw dataset_path")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sampling subsets")

    # hparams overrides
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tracks", type=str, default=None)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    args = parser.parse_args()

    # Manually instantiate H based on model type to avoid double-parsing
    from hparams.default_hparams import HparamsAbsorbing, HparamsAbsorbingConv, HparamsHierarchTransformer, HparamsUTransformer, HparamsOctuple
    
    model_type = args.model or "transformer"
    if model_type == 'transformer':
        H = HparamsAbsorbing(args)
    elif model_type == 'octuple':
        H = HparamsOctuple(args)
    elif model_type == 'hierarch_transformer':
        H = HparamsHierarchTransformer(args)
    elif model_type == 'U_transformer':
        H = HparamsUTransformer(args)
    else:
        H = HparamsAbsorbingConv(args)
    
    # Override with any explicitly set args
    for key in ["load_dir", "load_step", "model", "tracks", "bars", "dataset_path", "sample_steps", "masking_strategy"]:
        val = getattr(args, key, None)
        if val is not None and val != 0:
            H[key] = val
    
    # Set default dataset_path if still None
    if H.dataset_path is None:
        if H.tracks == 'melody':
            H.dataset_path = 'data/POP909_melody.npy'
        elif H.tracks == 'trio':
            H.dataset_path = 'data/POP909_trio.npy'
        else:
            H.dataset_path = 'data/POP909_octuple.npy'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.gap_start < 0 or args.gap_end < 0:
        args.gap_start = H.NOTES // 4
        args.gap_end = (H.NOTES * 3) // 4
    gap = (args.gap_start, args.gap_end)

    log("[main] loading dataset")

    # Load dataset respecting split if present
    if not args.no_split and os.path.isfile(args.split_path):
        split_ids = load_split_ids(args.split_path, args.split_partition)
        dataset = load_processed_subset(args.processed_dir, split_ids)
    else:
        # For octuple models, try to load from processed dir if it exists; fall back to stacked npy
        if H.model == 'octuple':
            log(f"[main] octuple model detected; attempting to load from processed_dir={args.processed_dir}")
            try:
                # Load all available processed files
                from pathlib import Path
                processed_files = sorted(Path(args.processed_dir).glob('*.npy'))
                all_ids = [f.stem for f in processed_files]  # e.g., '001', '002', ...
                dataset = load_processed_subset(args.processed_dir, all_ids)
                log(f"[main] loaded {len(all_ids)} octuple files from processed_dir")
            except Exception as e:
                log(f"[main] failed to load from processed_dir: {e}; falling back to {H.dataset_path}")
                dataset = np.load(H.dataset_path, allow_pickle=True)
        else:
            dataset = np.load(H.dataset_path, allow_pickle=True)

    log(f"[main] dataset loaded: {dataset.shape}")

    if args.bootstrap_only:
        # Load cached samples/refs from previous run
        out_dir = args.save_dir or H.load_dir
        if not os.path.isabs(out_dir):
            out_dir = os.path.join("logs", out_dir)
        cache_dir = os.path.join(out_dir, "cache")
        samples_path = os.path.join(cache_dir, "samples.npy")
        refs_path = os.path.join(cache_dir, "refs.npy")
        if not os.path.exists(samples_path) or not os.path.exists(refs_path):
            raise FileNotFoundError(f"Cached samples/refs not found in {cache_dir}. Run without --bootstrap_only first.")
        samples = np.load(samples_path)
        refs = np.load(refs_path)
        log(f"[main] loaded cached samples {samples.shape} and refs {refs.shape} from {cache_dir}")
        metrics_path = os.path.join(out_dir, "metrics", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            log(f"[main] loaded existing metrics from {metrics_path}")
        else:
            log("[main] no existing metrics.json; will compute fresh metrics")
            metrics = compute_metrics(samples, refs, args.mode, gap)
    else:
        log("[main] building sampler")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sampler = get_sampler(H).to(device)
        log("[main] loading weights")
        sampler = load_model(sampler, f"{H.sampler}_ema", H.load_step, H.load_dir)

        log("[main] starting generation")
        samples, refs = run_generation(H, sampler, dataset, args.n_samples, args.mode, gap, args.mask_tracks)
        log("[main] computing metrics")
        metrics = compute_metrics(samples, refs, args.mode, gap)

    out_dir = args.save_dir or H.load_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join("logs", out_dir)
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, "metrics.json")
    if not args.bootstrap_only:
        # Cache samples/refs for potential bootstrap_only reruns
        cache_dir = os.path.join(out_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, "samples.npy"), samples)
        np.save(os.path.join(cache_dir, "refs.npy"), refs)
        log(f"[main] cached samples/refs to {cache_dir}")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log(f"[main] saved metrics to {metrics_path}")
    
    if args.bootstrap_ci:
        log(f"[main] computing bootstrap CIs with {args.n_bootstrap} samples")
        ci_dict = bootstrap_confidence_intervals(samples, refs, args.mode, gap, args.n_bootstrap)
        metrics.update(ci_dict)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log(f"[main] added bootstrap CIs to {metrics_path}")

    if args.compute_fad:
        fad = structural_fad(samples, refs)
        metrics["structural_fad"] = fad
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log(f"Added structural FAD to {metrics_path}")

    if args.ablate_masking and args.mode == "infilling":
        # rerun with bar-aligned masking (mask full bars instead of raw indices)
        bar = 16
        b_gap = (args.gap_start // bar * bar, args.gap_end // bar * bar)
        samples_bar, refs_bar = run_generation(H, sampler, dataset, args.n_samples, args.mode, b_gap, args.mask_tracks)
        metrics_bar = compute_metrics(samples_bar, refs_bar, args.mode, b_gap)
        if args.compute_fad:
            metrics_bar["structural_fad"] = structural_fad(samples_bar, refs_bar)
        metrics_bar_path = os.path.join(metrics_dir, "metrics_bar.json")
        with open(metrics_bar_path, "w", encoding="utf-8") as f:
            json.dump(metrics_bar, f, indent=2)
        log(f"[main] saved bar-aligned masking metrics to {metrics_bar_path}")

    if args.save_midis:
        midi_dir = os.path.join(metrics_dir, "midis")
        os.makedirs(midi_dir, exist_ok=True)
        for i, ns in enumerate(np_to_ns(samples)):
            # note_seq NoteSequence has .SaveToString; easiest is note_sequence_to_midi_file
            from note_seq import note_sequence_to_midi_file
            note_sequence_to_midi_file(ns, os.path.join(midi_dir, f"sample_{i}.mid"))
        log(f"[main] saved MIDI samples to {midi_dir}")


if __name__ == "__main__":
    main()