import sys
import os
import torch
import argparse

# Add src to path
sys.path.append(os.getcwd())

from hparams.set_up_hparams import HparamsOctuple
from models.transformer import Transformer

def test_model_octuple():
    print("Testing Transformer with HparamsOctuple...")
    
    # Mock parser
    # Mock parser
    class MockParser:
        def __init__(self):
            self.model = 'octuple'
            self.n_vis = 1
            self.visdom_port = 8097
            self.batch_size = 2
            self.notes = 128
            self.bars = 8
            self.epochs = 1
            self.lr = 0.001
            self.load_dir = None
            self.log_base_dir = None
            self.tracks = 'string'
            self.ema = False
            self.amp = False
            self.load_step = 0
            self.validation_set_size = 0.1
        
    parser = MockParser()
    H = HparamsOctuple(parser)
    
    # Modify codebook size to be smaller for faster test if needed, but defaults are fine
    # H.codebook_size = (10, 10, 10, 10, 10, 10, 10, 10)
    
    model = Transformer(H)
    
    # Dummy input (B, T, C) -> (2, 128, 8)
    x = torch.randint(0, 10, (2, 128, 8))
    
    # Forward pass
    logits = model(x)
    
    # Check output
    # logits should be a list of 8 tensors
    assert isinstance(logits, list)
    assert len(logits) == 8
    
    print(f"Logits length: {len(logits)}")
    for i, l in enumerate(logits):
        print(f"Logits {i} shape: {l.shape}")
        # Expected shape: (B, T, V)
        assert l.shape == (2, 128, H.codebook_size[i])
        
    print("Model forward pass tests passed!")

if __name__ == "__main__":
    test_model_octuple()
