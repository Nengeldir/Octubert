"""PyTorch Dataset for Octuple-encoded MIDI data."""
import os
import numpy as np
import torch


class OctupleDataset(torch.utils.data.Dataset):
    """Dataset for loading octuple-encoded MIDI files from a directory.
    
    Each .npy file contains a sequence of octuple tokens (L, 8) where each row is:
    [bar, position, program, pitch, duration, velocity, time_sig, tempo]
    """
    
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
        data = np.load(file_path)  # Shape: (L, 8)
        
        # Check length
        if data.shape[0] < self.seq_len:
            # Pad if too short
            padding = np.zeros((self.seq_len - data.shape[0], data.shape[1]), dtype=data.dtype)
            data = np.concatenate([data, padding], axis=0)
        
        # Random crop
        if data.shape[0] > self.seq_len:
            start = np.random.randint(0, data.shape[0] - self.seq_len)
            data = data[start:start+self.seq_len]

        # Normalize bar numbers (column 0) to start from 0
        if data.shape[0] > 0:
            data[:, 0] = data[:, 0] - data[:, 0].min()

        return data  # (L, 8)

    def __len__(self):
        return len(self.data_files)
