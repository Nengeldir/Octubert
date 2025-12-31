import numpy as np
import os
import torch


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class SubseqSampler:
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


class OctupleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len):
        self.seq_len = seq_len
        self.data_files = []
        
        # Find all .npy files in the directory
        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.npy'):
                        self.data_files.append(os.path.join(root, file))
        else:
            raise ValueError(f"Data path {data_path} is not a directory")
            
        print(f"Found {len(self.data_files)} data files.")

    def __getitem__(self, idx):
        # Load the file
        file_path = self.data_files[idx]
        data = np.load(file_path) # Shape: (L, 8)
        
        # Check length
        if data.shape[0] < self.seq_len:
             # Pad if too short using 0 (assuming 0 is a valid padding token or we need a specific one)
             # Looking at OctupleEncoding, 0 usually means Bar 0, Pos 0 etc. 
             # Ideally we should repeat or pad. Let's pad with zeros for now.
             padding = np.zeros((self.seq_len - data.shape[0], data.shape[1]), dtype=data.dtype)
             data = np.concatenate([data, padding], axis=0)
        
        # Random crop
        if data.shape[0] > self.seq_len:
            start = np.random.randint(0, data.shape[0] - self.seq_len)
            data = data[start:start+self.seq_len]

        # Normalize bar numbers (column 0) to start from 0
        if data.shape[0] > 0:
            data[:, 0] = data[:, 0] - data[:, 0].min()

        return data # (L, 8)

    def __len__(self):
        return len(self.data_files)
