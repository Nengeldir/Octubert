import torch
import torch.nn as nn
import sys
import os

# Add local directory to path to import utils
sys.path.append(os.getcwd())

from hparams.set_up_hparams import HparamsOctuple
from models.transformer import Transformer
from models.absorbing_diffusion import AbsorbingDiffusion

class MockParser:
    def __init__(self):
        self.model = 'octuple'
        self.n_vis = 1
        self.visdom_port = 8097
        self.batch_size = 4
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

print("Initializing hyperparameters...")
parser = MockParser()
H = HparamsOctuple(parser)
print(f"Codebook Sizes: {H.codebook_size}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("Initializing Transformer...")
model = Transformer(H).to(device)
print("Transformer initialized.")

print("Initializing AbsorbingDiffusion...")
sampler = AbsorbingDiffusion(H, model, H.codebook_size).to(device)
print("AbsorbingDiffusion initialized.")

print("SUCCESS: Initialization complete without error.")
