# Evaluation: structure-aware metrics

This repo ships a headless evaluator `evaluate_metrics.py` for unconditional generation and infilling. It loads your checkpoints, generates samples, and computes distributional + structure-sensitive metrics. Outputs are JSON files under `logs/<log_dir>/metrics/`.

## Requirements

- A trained checkpoint under `logs/<log_dir>/saved_models/` (use EMA weights if available, e.g., `*_ema_<step>.th`).
- Processed POP909 per-song files in `data/processed/` named `001.npy` ... `909.npy`.
- Split file `data/splits/pop909_split.json` (committed) that defines train/val/test ids; by default we evaluate on `test`.
- `soundfont.sf2` is NOT required for metrics (only for audio playback elsewhere).

## Key arguments (common)

- `--load_dir`, `--load_step`, `--model`, `--tracks`, `--bars`: checkpoint selection (mirrors training args).
- `--n_samples`: number of generations/evals.
- `--split_path`, `--split_partition`: which split file/partition to draw references from (`test` by default).
- `--processed_dir`: where per-song processed `.npy` live (default `data/processed`).
- `--sample_steps`: diffusion steps (defaults from hparams if omitted).
- `--save_midis`: optionally write generated MIDI to `logs/<log_dir>/metrics/midis/`.
- `--compute_fad`: adds structural FAD (lightweight embedding) to metrics.
- `--ablate_masking`: for infilling, also run bar-aligned masking and save `metrics_bar.json`.

## Unconditional evaluation

Generates fresh samples, compares them to the reference distribution (test split) with JS, structural metrics, etc.

Example:

```bash
python evaluate_metrics.py \
  --mode unconditional \
  --n_samples 128 \
  --load_dir log_conv_transformer_trio_1024 \
  --load_step 250000 \
  --model conv_transformer --tracks trio --bars 64 \
  --split_partition test \
  --compute_fad --bootstrap_ci --save_midis
```

Outputs:

- `logs/log_conv_transformer_trio_1024/metrics/metrics.json` with pitch/duration JS, self-similarity, key consistency, bar summaries, phrase similarity, harmonic flux, optional structural FAD, and bootstrap 95% CIs (if `--bootstrap_ci` used).
- Optional MIDI under `.../metrics/midis/`.

## Infilling evaluation

Masks a span and fills it; measures masked token accuracy plus the same structure-sensitive metrics. Default mask is middle 50% if not provided.

Example:

```bash
python evaluate_metrics.py \
  --mode infilling \
  --n_samples 64 \
  --gap_start 256 --gap_end 768 \
  --load_dir log_conv_transformer_trio_1024 \
  --load_step 250000 \
  --model conv_transformer --tracks trio --bars 64 \
  --split_partition test \
  --compute_fad --ablate_masking --save_midis
```

Outputs:

- `metrics.json` with masked token accuracy, distributional and structural metrics.
- `metrics_bar.json` (if `--ablate_masking`) for bar-aligned masking comparison.
- Optional MIDI under `.../metrics/midis/`.

## Notes on splits and seeds

- The evaluator loads only the ids from the chosen partition of `data/splits/pop909_split.json`. To bypass, add `--no_split --dataset_path <stacked.npy>`, but reproducibility is best with the split.
- `--seed` controls NumPy/torch seeds for subset sampling; the split itself is fixed. If you want zero randomness in subset choice, set `--n_samples` large enough to cover the partition or adjust `eval_batch_size` in hparams.

## Metric glossary and interpretation

### Distributional metrics (unconditional + infilling)

- **pitch_class_js**: Jensen-Shannon divergence between pitch-class (mod 12) histograms.  
  **Lower is better** (0 = identical distribution). Typical good: < 0.05; acceptable: < 0.15.
- **duration_js**: JS divergence between duration bin histograms.  
  **Lower is better** (0 = identical). Good: < 0.10; acceptable: < 0.25.
- **consistency_pitch / consistency_duration**: Framewise overlap (OA) of consecutive pitch/duration distributions.  
  **Higher is better** (1.0 = perfect local stability). Good: > 0.75; shows stable local structure.
- **variance_pitch / variance_duration**: Variance in OA overlap across frames.  
  **Moderate is best** (~0.7–0.9); too low = monotonous; too high = erratic.

### Structural metrics (unconditional + infilling)

- **self_similarity_mean**: Average cosine similarity between consecutive bar pitch-class vectors.  
  **Higher suggests repetition/coherence** (0–1). Music typically 0.3–0.7; higher = more formulaic.
- **self_similarity_gap_to_ref**: Difference vs. reference self-similarity.  
  **Close to 0 is ideal** (±0.05 acceptable); large negative = under-repetitive; large positive = over-repetitive.
- **key_consistency**: Fraction of bars matching the dominant pitch class.  
  **Higher is better** (0–1). Good: > 0.5; indicates stable tonal center.
- **key_consistency_gap_to_ref / key_agreement_with_ref**: Match with reference tonal behavior.  
  **Close to 0 / 1.0 is ideal**.
- **bar_pitch_var_mean**: Average pitch variance per bar.  
  Moderate values (50–200) indicate melodic movement; very low = monotone; very high = chaotic.
- **bar_onset_density_mean**: Average note density per bar (0–1).  
  Typical range 0.2–0.6 for pop; < 0.1 = sparse; > 0.7 = dense.
- **phrase_similarity_mean**: Cosine similarity between consecutive 4-bar phrases (captures verse/chorus structure).  
  **Higher indicates phrase repetition** (0–1). Pop music typically 0.4–0.7; very high (> 0.8) = formulaic.
- **phrase_similarity_gap_to_ref**: Difference from reference phrase repetition.  
  **Close to 0 is ideal** (±0.05 acceptable).
- **harmonic_flux_mean**: Rate of harmonic change (pitch-class centroid shifts per bar, normalized 0–1).  
  **Moderate is typical** (0.2–0.4 for pop). Very low (< 0.1) = static harmony; high (> 0.5) = restless.
- **harmonic_flux_gap_to_ref**: Difference from reference harmonic movement rate.  
  **Close to 0 is ideal**.
- **structural_fad**: Frechet distance over structural embeddings (pitch-class + onset stats).  
  **Lower is better** (0 = identical). Typical range 5–50; < 10 is very close; > 100 suggests large structural drift.

### Infilling-specific metrics

- **masked_token_accuracy**: Exact token match on masked span (0–1).  
  **Higher is better**. Good reconstruction: > 0.4; excellent: > 0.6. Note this is strict; even good infills may score ~0.3–0.5 if creative.
- **infill_rhythm_correlation**: Correlation of onset density between masked span and context.  
  **Higher is better** (−1 to 1). Good continuity: > 0.3; excellent: > 0.6.
- **masked_span_embedding_similarity**: Cosine similarity of bar pitch vectors in masked region vs. ground truth.  
  **Higher is better** (0–1). Good: > 0.5; excellent: > 0.7.

### What makes a "good" model?

For **unconditional generation**: JS < 0.10, self-similarity gap near 0, phrase similarity gap near 0, key consistency > 0.5, harmonic flux gap near 0, structural FAD < 20.  
For **infilling**: masked token accuracy > 0.4, rhythm correlation > 0.4, span similarity > 0.5, plus acceptable distributional/structural metrics.

### Using confidence intervals

With `--bootstrap_ci`, each metric gets `*_ci` (95% CI bounds) and `*_std` (standard deviation) fields. Example:

```json
{
  "pitch_class_js": 0.0834,
  "pitch_class_js_ci": [0.0712, 0.0961],
  "pitch_class_js_std": 0.0063
}
```

**Report as**: "pitch_class_js: 0.083 ± 0.006" or "0.083 (95% CI [0.071, 0.096])". Narrower CIs indicate more reliable estimates.

### Limitations and caveats

- **Coarse proxies**: Key consistency, self-similarity, and harmonic flux use simple pitch-class heuristics; they don't capture advanced harmony, voice leading, or semantic structure. They're directional indicators, not full musicological analysis.
- **Dataset bias**: POP909 is pop music with consistent 4-bar phrase structure; metrics tuned here may not generalize to jazz, classical, or through-composed forms.
- **Token accuracy ceiling**: Exact token match is harsh; a creative, musically valid infill may score low if it deviates from the original.
- **Structural FAD is lightweight**: Without a pretrained music encoder (MusicBERT/octuple), this FAD is a stand-in. Real embedding-based FAD would be stronger evidence.
- **Small test set**: 91 test songs; use `--bootstrap_ci` to report confidence intervals. If CIs are wide (std > 20% of mean), consider multiple seeds or larger n_samples.

## Reproducing results

1) Ensure checkpoint and split are in place.
2) Run unconditional and infilling commands above for each model variant (baseline vs. octuple/structured masking).
3) Collect JSONs under `logs/<log_dir>/metrics/`.
4) Parse and tabulate results:

   ```bash
   python scripts/parse_metrics.py \
     logs/baseline/metrics/metrics.json \
     logs/octuple/metrics/metrics.json \
     --labels baseline octuple
   ```

5) Report tables in paper with interpretation: highlight JS, self-similarity gap, key consistency, masked accuracy, and structural FAD.
6) For ablation (random vs. bar masking), compare `metrics.json` vs. `metrics_bar.json` for the same model.
