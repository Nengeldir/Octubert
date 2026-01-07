import os
import torch
import numpy as np
import logging
from ..preprocessing.data import POP909TrioConverter, OneHotMelodyConverter
from ..cluster import sync_to_home
from note_seq import note_sequence_to_midi_file


def log(output):
    """Log message to both file and console."""
    logging.info(output)
    print(output)


def config_log(log_dir, filename="log.txt"):
    """
    Configure logging to write to log_dir/logs/filename.
    
    Args:
        log_dir: Base directory for logs (e.g., runs/model_id/)
        filename: Name of log file (default: log.txt)
    """
    logs_dir = os.path.join(log_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, filename),
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )


def start_training_log(hparams):
    """Log all hyperparameters at training start."""
    log("Using following hparams:")
    param_keys = list(hparams)
    param_keys.sort()
    for key in param_keys:
        log(f"> {key}: {hparams[key]}")


def _normalize_ckpt_name(name: str) -> str:
    # Map legacy absorbing names to clearer identifiers
    if name.endswith("_optim"):
        return "optim"
    if name.endswith("_ema"):
        return "ema"
    if name in ("absorbing", "sampler", "model"):
        return "model"
    return name


def save_model(model, model_save_name, step, log_dir):
    """
    Save model checkpoint to log_dir/checkpoints/.
    
    Args:
        model: PyTorch model to save
        model_save_name: Name identifier (e.g., "model", "ema", "optim")
        step: Training step number
        log_dir: Base directory (e.g., runs/model_id/)
    """
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    friendly_name = _normalize_ckpt_name(model_save_name)
    model_name = f"{friendly_name}_{step}.th"
    print(f"Saving {model_save_name} as {model_name}")
    save_path = os.path.join(ckpt_dir, model_name)
    torch.save(model.state_dict(), save_path)
    
    # Sync to permanent storage if on cluster
    sync_to_home(save_path)


def load_model(model, model_load_name, step, log_dir, strict=True):
    """
    Load model checkpoint from log_dir/checkpoints/.
    
    Args:
        model: PyTorch model to load weights into
        model_load_name: Name identifier (e.g., "model", "ema", "optim")
        step: Training step number
        log_dir: Base directory (e.g., runs/model_id/)
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        model: Model with loaded weights
    """
    ckpt_dir = os.path.join(log_dir, "checkpoints")

    friendly_name = _normalize_ckpt_name(model_load_name)
    candidates = [f"{friendly_name}_{step}.th"]
    if friendly_name != model_load_name:
        candidates.append(f"{model_load_name}_{step}.th")

    last_error = None
    search_dirs = [ckpt_dir]
    for base in search_dirs:
        for fname in candidates:
            path = os.path.join(base, fname)
            if not os.path.exists(path):
                continue
            print(f"Loading {fname} from {base}")
            try:
                state = torch.load(path)
                model.load_state_dict(state, strict=strict)
                return model
            except TypeError:
                model.load_state_dict(torch.load(path))
                return model
            except Exception as e:
                last_error = e
                continue

    if last_error:
        raise last_error
    raise FileNotFoundError(f"No checkpoint found for names {candidates} in {search_dirs}")


def save_samples(np_samples, step, log_dir):
    """
    Save generated samples to log_dir/samples/.
    
    Args:
        np_samples: NumPy array of generated samples
        step: Training step number
        log_dir: Base directory (e.g., runs/model_id/)
    """
    samples_dir = os.path.join(log_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    save_path = os.path.join(samples_dir, f'samples_{step}.npz.npy')
    np.save(save_path, np_samples, allow_pickle=True)
    
    # Sync to permanent storage if on cluster
    sync_to_home(save_path)


def save_stats(H, stats, step):
    """
    Save training statistics to log_dir/stats/.
    
    Args:
        H: Hyperparameters object with log_dir
        stats: Dictionary of training statistics
        step: Training step number
    """
    base_dir = H.log_dir if os.path.isabs(H.log_dir) else H.log_dir
    stats_dir = os.path.join(base_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    save_path = os.path.join(stats_dir, f"stats_{step}.pt")
    log(f"Saving stats to {save_path}")
    torch.save(stats, save_path)
    
    # Sync to permanent storage if on cluster
    sync_to_home(save_path)


def load_stats(H, step):
    """
    Load training statistics from log_dir/stats/.
    
    Args:
        H: Hyperparameters object with log_dir
        step: Training step number
        
    Returns:
        dict: Dictionary of training statistics
    """
    base_dir = H.log_dir if os.path.isabs(H.log_dir) else H.log_dir
    stats_dir = os.path.join(base_dir, "stats")
    load_path = os.path.join(stats_dir, f"stats_{step}.pt")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Stats file not found: {load_path}")
    
    log(f"Loading stats from {load_path}")
    return torch.load(load_path)


def log_stats(step, stats):
    """
    Log training statistics to console and file.
    
    Args:
        step: Training step number
        stats: Dictionary of statistics to log
    """
    msg_parts = [f"Step {step}"]
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            msg_parts.append(f"{key}: {value:.6f}")
        elif hasattr(value, 'item'):  # torch.Tensor
            msg_parts.append(f"{key}: {value.item():.6f}")
    log(" | ".join(msg_parts))


def save_noteseqs(ns, prefix='pre_adv'):
    for i, n in enumerate(ns):
        note_sequence_to_midi_file(n, prefix + f'_{i}.mid')


def samples_2_noteseq(np_samples, tokenizer_id=None):
    """
    Convert numpy samples to note_seq objects using tokenizer registry.
    """
    if tokenizer_id:
        from ..tokenizers.registry import TOKENIZER_REGISTRY
        spec = TOKENIZER_REGISTRY.get(tokenizer_id)
        
        if spec and spec.factory:
            converter = spec.factory()
                    
            # 1. Handle OneHot shape quirk (Melody OneHot expects 2D, Octuple expects 2D/3D)
            # If it's legacy OneHot and has an extra dim, squeeze it.
            if 'melody' in tokenizer_id and 'onehot' in tokenizer_id:
                 if np_samples.ndim == 3 and np_samples.shape[2] == 1:
                    np_samples = np_samples[:, :, 0]

            # 2. Call the unified method
            return converter.from_tensors(np_samples)
            
    return []
    


