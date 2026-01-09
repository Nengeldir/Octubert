"""
Evaluation CLI for symbolic music generation models.

Generates samples (or loads existing) and computes task-specific metrics.
See METRICS.md for detailed documentation of all metrics.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hparams.set_up_hparams import get_sampler_hparams
from smdiff.utils.sampler_utils import get_sampler, get_samples, ns_to_np
from smdiff.utils.log_utils import load_model
from smdiff.tasks import resolve_task_id
from smdiff.configs.loader import load_config
from smdiff.registry import resolve_model_id
from smdiff.tokenizers import resolve_tokenizer_id
from smdiff.data import apply_dataset_to_config, load_dataset
from smdiff.metrics import evaluate_unconditional, evaluate_infilling
from note_seq import midi_to_note_sequence


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate symbolic music generation models",
        epilog="See METRICS.md for detailed metric documentation"
    )
    
    # Model Config
    parser.add_argument("--model", type=str, help="Model ID (required if generating samples)")
    parser.add_argument("--load_dir", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--load_step", type=int, default=0, help="Checkpoint step (0 = use best)")
    parser.add_argument("--ema", action="store_true", default=True, help="Use EMA weights")
    parser.add_argument("--no-ema", dest="ema", action="store_false")
    
    # Task Config
    parser.add_argument("--task", type=str, required=True, choices=["uncond", "infill"],
                        help="Evaluation task: 'uncond' or 'infill'")
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="Dataset ID for ground truth distribution")
    
    # Sample Generation
    parser.add_argument("--sample_dir", type=str, default=None,
                        help="Load existing samples from directory (skip generation)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples to generate/evaluate")
    parser.add_argument("--sample_steps", type=int, default=0,
                        help="Diffusion steps (0 = use config default)")
    parser.add_argument("--batch_size", type=int, default=16)
    
    # Infilling Config
    parser.add_argument("--mask_start_bar", type=int, default=16)
    parser.add_argument("--mask_end_bar", type=int, default=32)
    parser.add_argument(
        "--mask2_start_bar",
        type=int,
        default=None,
        help="Optional second mask region start bar. If set (with --mask2_end_bar), evaluation runs each MIDI twice: once with region1 and once with region2.",
    )
    parser.add_argument(
        "--mask2_end_bar",
        type=int,
        default=None,
        help="Optional second mask region end bar (exclusive).",
    )
    parser.add_argument("--input_midi", type=str, default=None,
                        help="MIDI file for infill task (single file)")
    parser.add_argument("--input_midi_dir", type=str, default=None,
                        help="Directory of MIDI files for infill task (multiple conditionings)")
    parser.add_argument(
        "--n_midis",
        type=int,
        default=None,
        help="If using --input_midi_dir, limit to the first N MIDI files (sorted).",
    )
    parser.add_argument("--samples_per_midi", type=int, default=1,
                        help="How many samples to generate per conditioning MIDI")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: load_dir/evaluation)")
    parser.add_argument("--save_samples", action="store_true",
                        help="Save generated samples to disk")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def _resolve_infill_mask_regions(args):
    """Return list of (start_bar, end_bar) regions to run.
    """
    region1 = (args.mask_start_bar, args.mask_end_bar)
    # We now ignore region2 arguments completely for simplicity
    return [region1]


def _mask_conditioning_tokens_inplace(tokens: np.ndarray, tokenizer_id: str, mask_id: np.ndarray, start_bar: int, end_bar: int):
    """Overwrite the masked region in-place with the sampler's mask token(s).

    This is the mechanism used by AbsorbingDiffusion: positions equal to mask_id are
    considered *unknown* and will be sampled, while all other positions are treated
    as conditioning.
    """
    if start_bar < 0 or end_bar < 0 or end_bar <= start_bar:
        raise ValueError(f"Invalid mask region: start={start_bar}, end={end_bar}")

    # Octuple encoding: identify positions by Bar token in column 0.
    if "octuple" in tokenizer_id:
        if tokens.ndim != 3 or tokens.shape[-1] != mask_id.shape[0]:
            raise ValueError(
                f"Expected octuple tokens with shape (B, T, C=8), got {tokens.shape}; mask_id shape={mask_id.shape}"
            )

        bar_tokens = tokens[:, :, 0]
        mask_pos = (bar_tokens >= start_bar) & (bar_tokens < end_bar)  # (B, T)
        tokens[mask_pos] = mask_id
        return

    # Grid / one-hot style: assume 16 steps per bar and mask along the time axis.
    start_idx = start_bar * 16
    end_idx = end_bar * 16
    start_idx = max(0, start_idx)
    end_idx = max(start_idx, end_idx)

    if tokens.ndim == 2:
        # (B, T)
        tokens[:, start_idx:end_idx] = mask_id.item() if mask_id.size == 1 else mask_id[0].item()
    elif tokens.ndim == 3:
        # (B, T, C)
        tokens[:, start_idx:end_idx, :] = mask_id.reshape(1, 1, -1)
    else:
        raise ValueError(f"Unexpected token array shape for masking: {tokens.shape}")


def _list_midi_files(root_dir: str, limit: int | None = None) -> list[str]:
    """Recursively list MIDI files under root_dir, excluding any under a 'versions' folder."""
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    exclude_dirs = {"versions"}
    midi_paths = [
        str(p)
        for p in sorted(root_path.rglob("*.mid"))
        if not any(part.lower() in exclude_dirs for part in p.parts)
    ]
    if limit is not None:
        midi_paths = midi_paths[:limit]
    return midi_paths


def load_samples_from_dir(sample_dir, n_samples=None):
    """Load pre-generated samples from MIDI files."""
    print(f"Loading samples from {sample_dir}...")

    midi_files = _list_midi_files(sample_dir, limit=n_samples)
    
    from smdiff.utils.sampler_utils import ns_to_np
    samples = []
    for midi_file in midi_files:
        with open(midi_file, 'rb') as f:
            ns = midi_to_note_sequence(f.read())
        # Convert to tokens (assumes 64 bars)
        tokens = ns_to_np(ns, bars=64, tokenizer_id='trio_octuple')  # TODO: detect from config
        samples.append(tokens)
    
    print(f"Loaded {len(samples)} samples")
    return samples


def setup_infilling(args, tokenizer_id: str, mask_id: np.ndarray, seq_len: int):
    """Prepare conditioning tokens for infilling from one or many MIDIs.

    Returns:
        conditioning_tokens: (B, T, C) or (B, T) numpy array where masked region is set to mask_id
        originals_tokens:    numpy array with the corresponding unmasked ground truth (same shape)
        region_index:        list[int] of length B, where each entry is which region was used (0 or 1)
    """
    midi_files = []
    if args.input_midi_dir:
        midi_files = _list_midi_files(args.input_midi_dir, limit=args.n_midis)
        if not midi_files:
            raise ValueError(f"No MIDI files found in --input_midi_dir={args.input_midi_dir}")
    elif args.input_midi:
        midi_files = [args.input_midi]
    else:
        raise ValueError("Infill generation requires --input_midi or --input_midi_dir")

    mask_regions = _resolve_infill_mask_regions(args)

    bars = 64
    conditioning_list = []
    originals_list = []
    region_index: list[int] = []
    source_ids: list[str] = []

    for midi_path in midi_files:
        print(f"Conditioning on MIDI: {midi_path}")
        # Get filename stem for ID
        fname = os.path.basename(midi_path)
        stem, _ = os.path.splitext(fname)

        with open(midi_path, "rb") as f:
            ns = midi_to_note_sequence(f.read())

        tokens = ns_to_np(ns, bars, tokenizer_id)
        
        # Check if sample is valid for current mask arguments
        # If masking requires Bar 16, verify Bar 16 exists.
        if "octuple" in tokenizer_id:
            # tokens is (T, C) or (1, T, C)
            tok_check = tokens
            if tok_check.ndim == 3: tok_check = tok_check[0]
            if tok_check.ndim == 2 and tok_check.shape[1] > 0:
                 # Check max bar ID
                 max_bar_id = tok_check[:, 0].max()
                 req_end = args.mask_end_bar
                 if max_bar_id < req_end:
                      print(f"Skipping {midi_path}: Max Bar {max_bar_id} < Required End {req_end}")
                      continue

        if tokens.ndim == 1:
            # (T,)
            if tokens.shape[0] < seq_len:
                tokens = np.concatenate([tokens, np.zeros(seq_len - tokens.shape[0], dtype=tokens.dtype)], axis=0)
            elif tokens.shape[0] > seq_len:
                tokens = tokens[:seq_len]
            # Add batch dim
            tokens = tokens[np.newaxis, :] # (1, T)
        elif tokens.ndim == 2:
            # (T, C)
            if tokens.shape[0] < seq_len:
                 padding = np.zeros((seq_len - tokens.shape[0], tokens.shape[1]), dtype=tokens.dtype)
                 tokens = np.concatenate([tokens, padding], axis=0)
            elif tokens.shape[0] > seq_len:
                 tokens = tokens[:seq_len, :]
            # Add batch dim
            tokens = tokens[np.newaxis, :, :] # (1, T, C)
        
        for region_i, (start_bar, end_bar) in enumerate(mask_regions):
            masked_tokens = tokens.copy()
            _mask_conditioning_tokens_inplace(masked_tokens, tokenizer_id, mask_id, start_bar, end_bar)

            # Repeat for samples_per_midi
            masked_rep = np.repeat(masked_tokens, args.samples_per_midi, axis=0)
            orig_rep = np.repeat(tokens, args.samples_per_midi, axis=0)

            conditioning_list.append(masked_rep)
            originals_list.append(orig_rep)
            region_index.extend([region_i] * masked_rep.shape[0])
            source_ids.extend([stem] * masked_rep.shape[0])

    conditioning = np.concatenate(conditioning_list, axis=0) if conditioning_list else np.empty((0,))
    originals = np.concatenate(originals_list, axis=0) if originals_list else np.empty((0,))

    # Keep args.n_samples behavior as a *cap*, but warn if it would break the expected
    # (n_midis * samples_per_midi * n_regions) structure.
    if args.n_samples:
        expected = len(region_index)
        if args.n_samples != expected and len(mask_regions) > 1 and args.n_midis is not None:
            print(
                f"[warn] --n_samples={args.n_samples} does not match expected {expected} "
                f"(n_midis={args.n_midis} * samples_per_midi={args.samples_per_midi} * regions={len(mask_regions)}). "
                f"Capping to --n_samples may truncate region2."
            )
        conditioning = conditioning[: args.n_samples]
        originals = originals[: args.n_samples]
        region_index = region_index[: args.n_samples]
        source_ids = source_ids[: args.n_samples]

    return conditioning, originals, region_index, source_ids


def generate_samples(args, H, tokenizer_id, device, task):
    """Generate samples for evaluation (supports uncond and infill)."""
    
    # Load model
    sampler = get_sampler(H).to(device)
    
    # Load checkpoint
    checkpoints_dir = os.path.join(args.load_dir, "checkpoints")
    if args.load_step == 0:
        # Load best checkpoint
        if args.ema:
            best_paths = [
                os.path.join(checkpoints_dir, "ema_best.pt"),
                os.path.join(checkpoints_dir, "best.pt"),
            ]
            print("Using EMA weights.")
        else:
            best_paths = [
                os.path.join(checkpoints_dir, "best.pt"),
            ]
        
        checkpoint_path = None
        for path in best_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"Loading best checkpoint: {os.path.basename(path)}")
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"No best checkpoint found in {checkpoints_dir}")
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        sampler.load_state_dict(state_dict)
    else:
        load_key = f"{H.sampler}_ema" if args.ema else H.sampler
        print(f"Loading checkpoint from step {args.load_step}")
        sampler = load_model(sampler, load_key, args.load_step, args.load_dir)
    
    sampler.eval()
    
    conditioning = None
    original_tokens = None
    region_index = None
    source_ids = None

    if task == "infill":
        # Use the sampler's mask_id token(s) to mark unknown region.
        mask_id_np = sampler.mask_id.detach().cpu().numpy()
        # Ensure we pad/crop to the model's sequence length (H.NOTES)
        seq_len = getattr(H, 'NOTES', 1024)
        conditioning_np, original_tokens, region_index, source_ids = setup_infilling(args, tokenizer_id, mask_id_np, seq_len)
        args.n_samples = conditioning_np.shape[0]
        conditioning = torch.from_numpy(conditioning_np).long().to(device)

    # Generate samples
    all_samples = []
    num_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            current_b = min(args.batch_size, args.n_samples - len(all_samples))
            b_x_T = conditioning[:current_b] if conditioning is not None else None
            
            batch_out = get_samples(
                sampler,
                H.sample_steps,
                x_T=b_x_T,
                b=current_b
            )
            
            all_samples.append(batch_out)
            print(f"Batch {i+1}/{num_batches} finished")
            if conditioning is not None:
                conditioning = conditioning[current_b:]
    
    final_samples = np.concatenate(all_samples, axis=0)
    print(f"Generated samples shape: {final_samples.shape}")
    
    # Convert to list of (T, C) arrays
    samples = [final_samples[i] for i in range(len(final_samples))]
    originals = [original_tokens[i] for i in range(len(final_samples))] if original_tokens is not None else None
    region_index_out = region_index[: len(final_samples)] if region_index is not None else None
    source_ids_out = source_ids[: len(final_samples)] if source_ids is not None else None

    return samples, originals, region_index_out, source_ids_out


def load_training_data(dataset_id, n_samples=1000):
    """Load training data for distribution comparison."""
    print(f"Loading training data from dataset: {dataset_id}")
    
    from smdiff.data import resolve_dataset_id
    spec = resolve_dataset_id(dataset_id)
    
    data = np.load(spec.dataset_path, allow_pickle=True)
    
    # Random sample
    indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
    train_samples = [data[i] for i in indices]
    
    # Convert to consistent format
    processed = []
    for sample in train_samples:
        if isinstance(sample, np.ndarray) and sample.ndim == 0:
            sample = sample.item()
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample, dtype=np.int64)
        if sample.ndim == 3:
            sample = sample[0]  # Remove batch dim if present
        processed.append(sample)
    
    print(f"Loaded {len(processed)} training samples")
    return processed


def evaluate_task(task, generated_samples, train_samples=None, original_samples=None, 
                  mask_start_step=None, mask_end_step=None, is_octuple=False):
    """Run task-specific evaluation."""
    metrics = {}
    
    if task == "uncond":
        if train_samples is None:
            raise ValueError("Unconditional evaluation requires training data")
        
        print("\n" + "="*60)
        print("UNCONDITIONAL GENERATION METRICS")
        print("="*60)
        
        metrics = evaluate_unconditional(generated_samples, train_samples, is_octuple=is_octuple)
        
        print("\nDistribution Similarity:")
        print(f"  Pitch Class Histogram KL: {metrics['pch_kl']:.4f}")
        print(f"  Duration KL: {metrics['duration_kl']:.4f}")
        print(f"  Velocity KL: {metrics['velocity_kl']:.4f}")
        print(f"  Note Density KL: {metrics['note_density_kl']:.4f}")
        
        print("\nMusical Coherence:")
        print(f"  Self-Similarity: {metrics['self_similarity']:.4f} ± {metrics['self_similarity_std']:.4f}")
        print(f"  Pitch Range: {metrics['pitch_range_mean']:.1f} ± {metrics['pitch_range_std']:.1f} semitones")
        print(f"  Average Polyphony: {metrics['avg_polyphony']:.2f}")
        
        print("\nDiversity & Validity:")
        print(f"  Sample Diversity: {metrics['sample_diversity']:.4f}")
        print(f"  Valid Samples: {metrics['valid_samples_pct']:.1f}%")
        
    elif task == "infill":
        if original_samples is None:
            raise ValueError("Infilling evaluation requires original samples")
        
        print("\n" + "="*60)
        print("INFILLING METRICS")
        print("="*60)
            
        metrics = evaluate_infilling(
            generated_samples, original_samples,
            mask_start_step, mask_end_step,
            is_octuple=is_octuple  # Pass flag to metrics
        )
        
        print("\nReconstruction Accuracy (Masked Region):")
        print(f"  Pitch Accuracy: {metrics['pitch_accuracy']:.2f}% ± {metrics['pitch_accuracy_std']:.2f}")
        print(f"  Duration Accuracy: {metrics['duration_accuracy']:.2f}% ± {metrics['duration_accuracy_std']:.2f}")
        print(f"  Token Accuracy: {metrics['token_accuracy']:.2f}% ± {metrics['token_accuracy_std']:.2f}")
        
        print("\nMusical Quality (Masked Region):")
        print(f"  Infilled PCH KL: {metrics['infilled_pch_kl']:.4f}")
        print(f"  Infilled Density Error: {metrics['infilled_density_error']:.4f} notes/bar")
        
        print("\nBoundary Coherence:")
        print(f"  Pitch Smoothness: {metrics['boundary_pitch_smoothness']:.2f} semitones")
        print(f"  Rhythm Smoothness: {metrics['boundary_rhythm_smoothness']:.2f}")
        
        print("\nOverall Quality:")
        print(f"  Self-Similarity: {metrics['self_similarity']:.4f}")
        print(f"  Pitch Range: {metrics['pitch_range_mean']:.1f} semitones")
        print(f"  Sample Diversity: {metrics['sample_diversity']:.4f}")
        print(f"  Valid Samples: {metrics['valid_samples_pct']:.1f}%")
    
    else:
        raise ValueError(f"Unknown task '{task}'")

    return metrics


def main():
    args = get_args()

    # Validate second mask region args early.
    _ = _resolve_infill_mask_regions(args)
    
    # Validate task
    task_spec = resolve_task_id(args.task)
    print(f"Evaluation Task: {task_spec.description}")
    
    # Setup output directory
    if args.output_dir is None:
        if args.load_dir:
            args.output_dir = os.path.join(args.load_dir, "metrics")
        else:
            args.output_dir = f"metrics_{int(time.time())}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Metrics output directory: {args.output_dir}")
    
    generated_originals = None
    generated_region_index = None

    # Load or generate samples
    if args.sample_dir:
        generated_samples = load_samples_from_dir(args.sample_dir, args.n_samples)
    else:
        # Generate samples
        if not args.model or not args.load_dir:
            raise ValueError("--model and --load_dir required for sample generation")
        
        # Load config
        cfg = load_config(args.model, None, None)
        if args.dataset_id:
            cfg = apply_dataset_to_config(cfg, args.dataset_id)
        
        tokenizer_id = cfg.get("tokenizer_id") or cfg.get("tracks", "melody")
        resolve_tokenizer_id(tokenizer_id)
        
        # Build H object
        model_spec = resolve_model_id(args.model)
        argv = [
            sys.argv[0],
            "--model", model_spec.internal_model,
            "--tracks", cfg.get("tracks", "octuple" if "octuple" in tokenizer_id else "melody"),
            "--bars", "64",
            "--batch_size", str(args.batch_size),
        ]
        
        if cfg.get("dataset_path"):
            argv += ["--dataset_path", cfg["dataset_path"]]
        
        prev_argv = sys.argv
        sys.argv = argv
        try:
            H = get_sampler_hparams('sample')
        finally:
            sys.argv = prev_argv
        
        H.tokenizer_id = tokenizer_id
        H.model_id = args.model
        if args.sample_steps > 0:
            H.sample_steps = args.sample_steps
        
        generated_samples, generated_originals, generated_region_index, generated_source_ids = generate_samples(
            args, H, tokenizer_id, args.device, args.task
        )
        
        # Save samples if requested
        if args.save_samples:
            subfolder = "uncond" if args.task == "uncond" else "infill"
            sample_out_dir = os.path.join(args.output_dir, "samples", subfolder)
            os.makedirs(sample_out_dir, exist_ok=True)
            
            from smdiff.utils.log_utils import samples_2_noteseq
            
            # Save generated samples one by one with proper naming
            for i, sample_arr in enumerate(generated_samples):
                # Convert (T, C) -> (1, T, C) batch for conversion
                # Use a specific batch dimension to ensure samples_2_noteseq works
                if sample_arr.ndim == 1:
                    batch = sample_arr[np.newaxis, :]
                else:
                    batch = sample_arr[np.newaxis, :, :]
                    
                ns_list = samples_2_noteseq(batch, tokenizer_id)
                if not ns_list:
                    print(f"Warning: Failed to convert sample {i}")
                    continue
                ns = ns_list[0]
                
                if args.task == "uncond":
                    filename = f"uncond_{i}.mid"
                else:
                    # Infill naming: infill_{source_id}_{i}.mid
                    src_id = generated_source_ids[i] if generated_source_ids else f"unknown_{i}"
                    filename = f"infill_{src_id}_{i}.mid"
                
                out_path = os.path.join(sample_out_dir, filename)
                # Use note_seq to write
                from note_seq import note_sequence_to_midi_file
                note_sequence_to_midi_file(ns, out_path)
            
            print(f"Saved {len(generated_samples)} samples to {sample_out_dir}")
    
    # Load training data for distribution comparison
    train_samples = load_training_data(args.dataset_id, n_samples=1000)

    metrics = None
    
    # Task-specific evaluation
    if args.task == "uncond":
        metrics = evaluate_task(
            "uncond",
            generated_samples,
            train_samples=train_samples,
            is_octuple=("octuple" in tokenizer_id)
        )
    
    elif args.task == "infill":
        # For infilling, prefer originals from conditioning if generation was used
        if generated_originals is not None:
            original_samples = generated_originals
        else:
            # Fallback: slice training data
            original_samples = train_samples[: len(generated_samples)]

        mask_regions = _resolve_infill_mask_regions(args)

        # If two regions are requested and we generated with region tagging, evaluate per region
        # (the masked region differs per sample, so a single mask window would be wrong).
        if len(mask_regions) > 1 and generated_region_index is not None:
            per_region_metrics = []
            for region_i, (start_bar, end_bar) in enumerate(mask_regions):
                idxs = [i for i, r in enumerate(generated_region_index) if r == region_i]
                region_gen = [generated_samples[i] for i in idxs]
                region_orig = [original_samples[i] for i in idxs]

                mask_start_step = start_bar * 16
                mask_end_step = end_bar * 16

                metrics_i = evaluate_task(
                    "infill",
                    region_gen,
                    train_samples=None,
                    original_samples=region_orig,
                    mask_start_step=mask_start_step,
                    mask_end_step=mask_end_step,
                    is_octuple=("octuple" in tokenizer_id)
                )
                per_region_metrics.append(((start_bar, end_bar), metrics_i))

                metrics_file_i = os.path.join(
                    args.output_dir, f"metrics_infill_region{region_i+1}_{start_bar}-{end_bar}.json"
                )
                with open(metrics_file_i, "w") as f:
                    json.dump(metrics_i, f, indent=2)
                print(f"\nRegion {region_i+1} metrics saved to: {metrics_file_i}")

            # Also provide a combined view (simple mean of scalar metrics across regions where possible)
            combined = {}
            keys = set().union(*[m.keys() for _, m in per_region_metrics])
            for k in keys:
                vals = []
                for _, m in per_region_metrics:
                    v = m.get(k)
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        vals.append(float(v))
                if vals:
                    combined[k] = float(np.mean(vals))

            metrics = combined
        else:
            # Single-region infilling evaluation
            mask_start_step = args.mask_start_bar * 16
            mask_end_step = args.mask_end_bar * 16

            metrics = evaluate_task(
                "infill",
                generated_samples,
                train_samples=None,
                original_samples=original_samples,
                mask_start_step=mask_start_step,
                mask_end_step=mask_end_step,
                is_octuple=("octuple" in tokenizer_id)
            )

    if metrics is None:
        raise RuntimeError("Evaluation produced no metrics")
    
    # Save metrics to JSON
    metrics_file = os.path.join(args.output_dir, f"metrics_{args.task}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    print("\n" + "="*60)
    print("Evaluation complete! See METRICS.md for metric interpretation.")
    print("="*60)


if __name__ == "__main__":
    main()
