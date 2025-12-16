import sys
import os
import torch
import numpy as np
import shutil

# Add src to path
sys.path.append(os.getcwd())

from utils.data_utils import OctupleDataset

def test_octuple_loading():
    print("Testing OctupleDataset...")
    # Create dummy data directory and file
    dummy_dir = "tests/dummy_data"
    if os.path.exists(dummy_dir):
        shutil.rmtree(dummy_dir)
    os.makedirs(dummy_dir)
    
    # Create a dummy .npy file (L=128, C=8)
    dummy_data = np.random.randint(0, 10, (200, 8)).astype(np.int64)
    np.save(os.path.join(dummy_dir, "song1.npy"), dummy_data)
    
    # Initialize dataset
    seq_len = 128
    dataset = OctupleDataset(dummy_dir, seq_len)
    
    assert len(dataset) == 1
    
    # Test getitem
    item = dataset[0]
    print(f"Item shape: {item.shape}")
    assert item.shape == (seq_len, 8)
    
    # Test padding
    short_data = np.random.randint(0, 10, (50, 8)).astype(np.int64)
    np.save(os.path.join(dummy_dir, "song2.npy"), short_data)
    dataset = OctupleDataset(dummy_dir, seq_len)
    
    # Force load of the short file (index 0 or 1 depending on walk order, try both)
    # We can iterate
    for i in range(len(dataset)):
        item = dataset[i]
        assert item.shape == (seq_len, 8)
        
    print("OctupleDataset tests passed!")
    
    # Clean up
    shutil.rmtree(dummy_dir)

if __name__ == "__main__":
    test_octuple_loading()
