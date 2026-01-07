"""Base dataset utilities for generic data loading."""
import numpy as np


def cycle(iterable):
    """Infinite iterator that cycles through an iterable."""
    while True:
        for x in iterable:
            yield x


class SubseqSampler:
    """Sampler that extracts random subsequences from a dataset."""
    def __init__(self, dataset, seq_len):
        self.dataset = dataset
        self.seq_len = seq_len
        
    def __getitem__(self, item):
        if self.seq_len == self.dataset.shape[1]:
            return self.dataset[item]
        seq_start = np.random.randint(0, self.dataset.shape[1] - self.seq_len)
        return self.dataset[item][:, seq_start:seq_start+self.seq_len]

    def __len__(self):
        return len(self.dataset)


class SimpleNpyDataset:
    """Wraps a numpy array for torch DataLoader compatibility."""
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x = self.data[idx]
        if x.shape[0] > self.seq_len:
            start = np.random.randint(0, x.shape[0] - self.seq_len + 1)
            x = x[start:start+self.seq_len]
        elif x.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - x.shape[0], x.shape[1]), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        return x
