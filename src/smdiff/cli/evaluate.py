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
    parser.add_argument("--input_midi", type=str, default=None,
                        help="MIDI file for infill task (single file)")
    parser.add_argument("--input_midi_dir", type=str, default=None,
                        help="Directory of MIDI files for infill task (multiple conditionings)")
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


def load_samples_from_dir(sample_dir, n_samples=None):
    """Load pre-generated samples from MIDI files."""
    print(f"Loading samples from {sample_dir}...")
    
    from glob import glob
    midi_files = sorted(glob(os.path.join(sample_dir, "*.mid")))
    
    if n_samples:
        midi_files = midi_files[:n_samples]
    
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


def setup_infilling(args, tokenizer_id):
    """Prepare conditioning tokens for infilling from one or many MIDIs."""
    midi_files = []
    if args.input_midi_dir:
        from glob import glob
        midi_files = sorted(glob(os.path.join(args.input_midi_dir, "*.mid")))
        if not midi_files:
            raise ValueError(f"No MIDI files found in --input_midi_dir={args.input_midi_dir}")
    elif args.input_midi:
        midi_files = [args.input_midi]
    else:
        raise ValueError("Infill generation requires --input_midi or --input_midi_dir")

    bars = 64
    tokens_list = []
    for midi_path in midi_files:
        print(f"Conditioning on MIDI: {midi_path}")
        with open(midi_path, 'rb') as f:
            ns = midi_to_note_sequence(f.read())
        tokens = ns_to_np(ns, bars, tokenizer_id)
        if tokens.ndim == 2:
            tokens = tokens[np.newaxis, :]
        tokens_rep = np.repeat(tokens, args.samples_per_midi, axis=0)
        tokens_list.append(tokens_rep)

    tokens = np.concatenate(tokens_list, axis=0)
    if args.n_samples:
        tokens = tokens[:args.n_samples]

    return tokens


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
    if task == "infill":
        conditioning_np = setup_infilling(args, tokenizer_id)
        args.n_samples = conditioning_np.shape[0]
        original_tokens = conditioning_np.copy()
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
    
    return samples, originals


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
                  mask_start_step=None, mask_end_step=None):
    """Run task-specific evaluation."""
    
    if task == "uncond":
        if train_samples is None:
            raise ValueError("Unconditional evaluation requires training data")
        
        print("\n" + "="*60)
        print("UNCONDITIONAL GENERATION METRICS")
        print("="*60)
        
        metrics = evaluate_unconditional(generated_samples, train_samples)
        
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
            mask_start_step, mask_end_step
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
    
    return metrics


def main():
    args = get_args()
    
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
        
        generated_samples, generated_originals = generate_samples(args, H, tokenizer_id, args.device, args.task)
        
        # Save samples if requested
        if args.save_samples:
            sample_out_dir = os.path.join(args.output_dir, "samples")
            os.makedirs(sample_out_dir, exist_ok=True)
            from smdiff.utils.sampler_utils import save_generated_samples
            gen_array = np.array(generated_samples)
            save_generated_samples(gen_array, tokenizer_id, sample_out_dir, prefix="eval")
            print(f"Saved samples to {sample_out_dir}")
    
    # Load training data for distribution comparison
    train_samples = load_training_data(args.dataset_id, n_samples=1000)
    
    # Task-specific evaluation
    if args.task == "uncond":
        metrics = evaluate_task(
            "uncond",
            generated_samples,
            train_samples=train_samples
        )
    
    elif args.task == "infill":
        # For infilling, prefer originals from conditioning if generation was used
        if generated_originals is not None:
            original_samples = generated_originals
        else:
            # Fallback: slice training data
            original_samples = train_samples[:len(generated_samples)]
        
        # Convert bar indices to steps
        mask_start_step = args.mask_start_bar * 16
        mask_end_step = args.mask_end_bar * 16
        
        metrics = evaluate_task(
            "infill",
            generated_samples,
            train_samples=None,
            original_samples=original_samples,
            mask_start_step=mask_start_step,
            mask_end_step=mask_end_step
        )
    
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
