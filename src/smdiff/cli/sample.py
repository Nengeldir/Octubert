import argparse
import os
import sys
import time
import numpy as np
import torch
import yaml
from note_seq import midi_to_note_sequence

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
    
# Also ensure 'src' is on sys.path so 'smdiff' package resolves when running by path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Internal imports
from hparams.set_up_hparams import get_sampler_hparams
from smdiff.utils.sampler_utils import get_sampler, get_samples, ns_to_np, save_generated_samples
from smdiff.utils.log_utils import load_model
from smdiff.tasks import resolve_task_id
from smdiff.configs.loader import load_config
from smdiff.registry import resolve_model_id
from smdiff.tokenizers import resolve_tokenizer_id
from smdiff.data import apply_dataset_to_config

def get_args():
    parser = argparse.ArgumentParser(description="Sampling / Infilling CLI")
    
    # Model Config
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument("--tokenizer_id", type=str, default=None, help="Tokenizer ID (auto-detected if not specified)")
    parser.add_argument("--dataset_id", type=str, default=None, help="Dataset ID for auto-config")
    parser.add_argument("--load_dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--load_step", type=int, default=0, help="Step to load (0 = auto-detect best/latest)")
    parser.add_argument("--config", type=str, default=None, help="Optional config override")
    
    # Task Config
    parser.add_argument("--task", type=str, default="uncond", help="Task ID: 'uncond' or 'infill'")
    
    # Generation Settings
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=0, help="Diffusion steps (0 = use config default)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bars", type=int, default=None, help="Number of bars (auto-detected if not specified)")
    
    # Infilling Condition
    parser.add_argument("--input_midi", type=str, default=None, help="MIDI file for infill task (single file)")
    parser.add_argument("--input_midi_dir", type=str, default=None, help="Directory of MIDI files for infill task (use multiple conditionings)")
    parser.add_argument("--samples_per_midi", type=int, default=1, help="How many samples to generate per conditioning MIDI")
    parser.add_argument("--mask_start_bar", type=int, default=16)
    parser.add_argument("--mask_end_bar", type=int, default=32)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ema", action="store_true", default=True, help="Use EMA weights (default: True)")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="Don't use EMA weights")

    return parser.parse_args()

def setup_infilling(args, H, tokenizer_id, device):
    """Prepares conditioning for Infilling - extracts 64 bars from one or many input MIDIs."""
    midi_files = []
    if args.input_midi_dir:
        from glob import glob
        midi_files = sorted(glob(os.path.join(args.input_midi_dir, "*.mid")))
        if not midi_files:
            raise ValueError(f"No MIDI files found in --input_midi_dir={args.input_midi_dir}")
    elif args.input_midi:
        midi_files = [args.input_midi]
    else:
        raise ValueError("Task 'infill' requires --input_midi or --input_midi_dir.")

    bars = 64
    tokens_list = []
    for midi_path in midi_files:
        print(f"Processing input MIDI for infilling: {midi_path}")
        with open(midi_path, 'rb') as f:
            ns = midi_to_note_sequence(f.read())
        tokens = ns_to_np(ns, bars, tokenizer_id)
        if tokens.ndim == 2:
            tokens = tokens[np.newaxis, :]
        # Repeat per conditioning MIDI
        tokens_rep = np.repeat(tokens, args.samples_per_midi, axis=0)
        tokens_list.append(tokens_rep)

    tokens = np.concatenate(tokens_list, axis=0)

    # Trim or cap to requested n_samples if specified
    if args.n_samples:
        tokens = tokens[:args.n_samples]

    # Create Mask (1=Keep/Known, 0=Masked/Generate)
    mask = np.ones_like(tokens)

    is_octuple = "octuple" in tokenizer_id

    # Apply Masking Logic
    if is_octuple:
        # Octuple: Scan for Bar tokens (Column 0)
        for b in range(tokens.shape[0]):
            bar_tokens = tokens[b, :, 0]
            mask_idx = (bar_tokens >= args.mask_start_bar) & (bar_tokens < args.mask_end_bar)
            mask[b, mask_idx, :] = 0
    else:
        # Grid: Fixed steps (16 per bar)
        notes = H.NOTES if hasattr(H, 'NOTES') else H.get('NOTES', 1024)
        start_idx = args.mask_start_bar * 16
        end_idx = args.mask_end_bar * 16
        start_idx = max(0, min(start_idx, notes))
        end_idx = max(0, min(end_idx, notes))
        mask[:, start_idx:end_idx] = 0

    return torch.from_numpy(tokens).long().to(device), torch.from_numpy(mask).float().to(device)

def main():
    args = get_args()
    
    # 1. Validate Task
    task_spec = resolve_task_id(args.task)
    print(f"Running Task: {task_spec.description}")

    # 2. Load Config and merge with dataset if provided
    config_path = args.config
    if not config_path:
        # Auto-detect config in load_dir
        for c in ["configs/config.yaml", "configs/hparams.yaml"]:
            p = os.path.join(args.load_dir, c)
            if os.path.exists(p):
                config_path = p
                print(f"Auto-detected config: {config_path}")
                break
    
    cfg = load_config(args.model, config_path, None)
    
    # Apply dataset config if specified
    if args.dataset_id:
        cfg = apply_dataset_to_config(cfg, args.dataset_id)
    
    # Determine tokenizer_id
    tokenizer_id = args.tokenizer_id or cfg.get("tokenizer_id") or cfg.get("tracks", "melody")
    resolve_tokenizer_id(tokenizer_id)  # Validate
    
    # Force 64 bars for generation
    bars = 64
    
    # Build argv for legacy hparams system
    model_spec = resolve_model_id(args.model)
    
    argv = [
        sys.argv[0],
        "--model", model_spec.internal_model,
        "--tracks", cfg.get("tracks", "octuple" if "octuple" in tokenizer_id else "melody"),
        "--bars", str(bars),
        "--batch_size", str(args.batch_size),
    ]
    
    # Add optional params
    if cfg.get("dataset_path"):
        argv += ["--dataset_path", cfg["dataset_path"]]
    
    # Swap sys.argv temporarily to use legacy hparams system
    prev_argv = sys.argv
    sys.argv = argv
    try:
        H = get_sampler_hparams('sample')
    finally:
        sys.argv = prev_argv
    
    # Apply overrides
    H.tokenizer_id = tokenizer_id
    H.model_id = args.model
    if args.sample_steps > 0:
        H.sample_steps = args.sample_steps
    
    # 3. Load Model
    print(f"Loading {args.model} from {args.load_dir}...")
    sampler = get_sampler(H).to(args.device)
    
    # Load weights - prioritize best checkpoint
    checkpoints_dir = os.path.join(args.load_dir, "checkpoints")
    
    if args.load_step == 0:
        # Default: load best checkpoint
        if args.ema:
            # Try ema_best.pt first, then best.pt
            best_paths = [
                os.path.join(checkpoints_dir, "ema_best.pt"),
                os.path.join(checkpoints_dir, "best.pt"),
            ]
            print("Using EMA weights.")
        else:
            best_paths = [
                os.path.join(checkpoints_dir, "best.pt"),
            ]
        
        # Find first existing checkpoint
        checkpoint_path = None
        for path in best_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"Loading best checkpoint: {os.path.basename(path)}")
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"No best checkpoint found in {checkpoints_dir}")
        
        # Load directly from path
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        sampler.load_state_dict(state_dict)
    else:
        # Load specific step
        load_key = f"{H.sampler}_ema" if args.ema else H.sampler
        print(f"Loading checkpoint from step {args.load_step} ({'EMA' if args.ema else 'non-EMA'})")
        sampler = load_model(sampler, load_key, args.load_step, args.load_dir)
    sampler.eval()

    # 4. Prepare Logic based on Task
    x_T = None
    mask = None

    if task_spec.id == 'infill':
        x_T, mask = setup_infilling(args, H, tokenizer_id, args.device)
        # Sync n_samples with actual conditioning batch (may come from many MIDIs)
        args.n_samples = x_T.shape[0]
    elif task_spec.id == 'uncond':
        pass  # x_T and mask remain None
    else:
        raise NotImplementedError(f"Logic for {task_spec.id} not implemented in main loop yet.")

    # 5. Generate
    all_samples = []
    num_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    
    print(f"Starting Generation ({H.sample_steps} steps)...")
    with torch.no_grad():
        for i in range(num_batches):
            current_b = min(args.batch_size, args.n_samples - len(all_samples))
            
            # Slice batch conditioning if present
            b_x_T = x_T[:current_b] if x_T is not None else None
            
            # For infilling: mask is applied inside the model via x_T pre-conditioning
            # The masked regions in x_T are already set to mask_id
            # So we don't need to pass mask separately to get_samples
            
            batch_out = get_samples(
                sampler, 
                H.sample_steps, 
                x_T=b_x_T,
                b=current_b
            )
            
            # batch_out is already numpy from get_samples
            all_samples.append(batch_out)
            print(f"Batch {i+1}/{num_batches} finished.")

    final_samples = np.concatenate(all_samples, axis=0)
    
    # Debug: Check shape
    print(f"Generated samples shape: {final_samples.shape}")
    print(f"Expected shape: ({args.n_samples}, {H.NOTES}, 8) = ({args.n_samples}, 1024, 8)")

    # 6. Save directly to MIDI
    out_dir = os.path.join(args.load_dir, "samples", args.task)
    os.makedirs(out_dir, exist_ok=True)
    
    save_generated_samples(final_samples, tokenizer_id, out_dir, prefix=f"{args.task}_{int(time.time())}")
    print("Done.")

if __name__ == "__main__":
    main()