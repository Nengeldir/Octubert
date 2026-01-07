"""Base dataset utilities for generic data loading."""
import numpy as np
import torch


def cycle(iterable):
    """Infinite iterator that cycles through an iterable."""
    while True:
        for x in iterable:
            yield x


class SimpleNpyDataset(torch.utils.data.Dataset):
    """
    Wraps a numpy array for torch DataLoader compatibility.
    Uses tokenizer_id to determine processing logic (Octuple vs OneHot).
    """
    def __init__(self, data: np.ndarray, seq_len: int, tokenizer_id: str | None = None):
        self.data = data
        self.seq_len = seq_len
        self.tokenizer_id = tokenizer_id or ""
        self.is_octuple = "octuple" in self.tokenizer_id
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 1. Get the full sequence
        x = self.data[idx]
        
        # 2. Random Crop (Common to both)
        length = x.shape[0]
        if length > self.seq_len:
            start = np.random.randint(0, length - self.seq_len + 1)
            x = x[start : start + self.seq_len]
        
        # 3. Pad (Common logic, handles shapes automatically)
        elif length < self.seq_len:
            pad_len = self.seq_len - length
            # Use ndim here just for the padding command structure, 
            # which is harmless since it's just numpy syntax.
            if x.ndim == 1:
                x = np.pad(x, (0, pad_len), 'constant')
            else:
                x = np.pad(x, [(0, pad_len), (0, 0)], 'constant')

        # 4. EXPLICIT Octuple Logic (Triggered by ID, not shape)
        if self.is_octuple:
            # Bar Normalization:
            # If we cropped into the middle of a song, reset Bar numbers to start at 0.
            if x.shape[0] > 0:
                first_bar = x[0, 0]
                if first_bar > 0:
                    x[:, 0] -= first_bar
                    x[:, 0] = np.maximum(x[:, 0], 0)

        # 5. Return
        return torch.from_numpy(x).long()
