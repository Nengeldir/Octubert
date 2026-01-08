import numpy as np
import torch
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def optim_warmup(H, step, optim):
    lr = H.lr * float(step) / H.warmup_iters
    for param_group in optim.param_groups:
        param_group['lr'] = lr



def augment_note_tensor(H, batch):
    """
    Apply pitch augmentation.
    Auto-detects OneHot vs Octuple based on shape and config.
    Returns the same type as the input (numpy or torch).
    """
    # 1. Check if global augmentation is enabled
    if not hasattr(H, 'augment') or not H.augment:
        return batch

    # 2. Convert to Torch
    was_numpy = False
    device = None
    if isinstance(batch, np.ndarray):
        batch_t = torch.from_numpy(batch).long()
        was_numpy = True
    else:
        batch_t = batch.long()
        device = batch_t.device if torch.is_tensor(batch_t) else None

    B = batch_t.shape[0]

    # 3. Identify Data Type & Pitch Column
    if batch_t.ndim == 3 and batch_t.shape[2] == 8:
        # --- OCTUPLE MODE ---
        is_octuple = True
        pitch_dim = 3

        # Get Pitch Vocab Size from Config
        # models.yaml says codebook_size is a list: [2048, 128, 129, 256, ...]
        # We need index 3 (Pitch)
        if hasattr(H, 'codebook_size') and isinstance(H.codebook_size, list):
            vocab_size = int(H.codebook_size[3])
        else:
            vocab_size = 128  # Fallback safe limit

    else:
        # --- ONEHOT MODE ---
        is_octuple = False
        # For non-octuple tensors, assume first feature/channel holds pitch/token ids
        pitch_dim = 0 if batch_t.ndim == 3 else None

        # Get Vocab Size
        if hasattr(H, 'codebook_size'):
            # Handle if codebook_size is a list (some configs) or int
            if isinstance(H.codebook_size, list) and len(H.codebook_size) > 0:
                vocab_size = int(H.codebook_size[0])
        else:
            vocab_size = 128

    # 4. Apply Augmentation
    for i in range(B):
        # Extract the values to modify
        if is_octuple:
            vals = batch_t[i, :, pitch_dim]
        elif batch_t.ndim == 3:
            vals = batch_t[i, :, 0]
        else:
            vals = batch_t[i]

        # Ignore special tokens (Padding=0, Start=1, etc usually < 2)
        mask = vals > 1
        if not mask.any():
            continue

        valid_notes = vals[mask]

        min_pitch = int(valid_notes.min().item())
        max_pitch = int(valid_notes.max().item())

        # Bounds: Keep notes > 1 and < vocab_size
        lower_limit = int(-min_pitch + 2)
        upper_limit = int(vocab_size - max_pitch - 1)

        if upper_limit <= lower_limit:
            continue

        # Random shift (e.g. -4 to +4) within safe bounds
        shift = int(np.random.randint(lower_limit, upper_limit))

        if shift != 0:
            vals[mask] += shift

    if was_numpy:
        return batch_t.numpy()
    if device is not None and batch_t.device != device:
        batch_t = batch_t.to(device)
    return batch_t
