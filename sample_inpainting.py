
import os
import torch
import numpy as np
import argparse
from hparams import get_sampler_hparams
from utils import get_sampler, load_model
from utils.data_utils import OctupleDataset
from utils.sampler_utils import ns_to_np, get_samples, np_to_ns
from note_seq import note_sequence_to_midi_file

def apply_mask(x_0, strategy):
    """
    Applies the specified masking strategy to x_0.
    Returns the mask (boolean tensor).
    Logic derived from AbsorbingDiffusion.q_sample_partial.
    """
    b, seq_len, _ = x_0.shape
    device = x_0.device
    
    # Octuple indices
    # 0: Bar, 1: Pos, 2: Prog, 3: Pitch, 4: Dur, 5: Vel, 6: TimeSig, 7: Tempo
    
    mask = torch.zeros_like(x_0, dtype=torch.bool, device=device)
    
    if strategy == '1_bar_all':
        bar_indices = x_0[:, :, 0]
        max_bars = bar_indices.max(dim=1)[0]
        target_bars = (torch.rand(b, device=device) * (max_bars.float() + 1)).long()
        target_bars_exp = target_bars.unsqueeze(1).expand(-1, seq_len)
        
        target_attributes = torch.tensor([3, 4, 5, 7], device=device)
        attr_mask = torch.zeros(8, dtype=torch.bool, device=device)
        attr_mask[target_attributes] = True
        attr_mask = attr_mask.unsqueeze(0).unsqueeze(0).expand(b, seq_len, -1)
        
        mask = (bar_indices == target_bars_exp).unsqueeze(-1) & attr_mask
        
    elif strategy == '2_bar_all':
        bar_indices = x_0[:, :, 0]
        max_bars = bar_indices.max(dim=1)[0]
        r1 = (torch.rand(b, device=device) * (max_bars.float() + 1)).long()
        r2 = (torch.rand(b, device=device) * (max_bars.float() + 1)).long()
        
        target_bars1 = r1.unsqueeze(1).expand(-1, seq_len)
        target_bars2 = r2.unsqueeze(1).expand(-1, seq_len)
        
        target_attributes = torch.tensor([3, 4, 5, 7], device=device)
        attr_mask = torch.zeros(8, dtype=torch.bool, device=device)
        attr_mask[target_attributes] = True
        attr_mask = attr_mask.unsqueeze(0).unsqueeze(0).expand(b, seq_len, -1)
        
        mask = ((bar_indices == target_bars1) | (bar_indices == target_bars2)).unsqueeze(-1) & attr_mask

    elif strategy == '1_bar_attribute':
        bar_indices = x_0[:, :, 0]
        max_bars = bar_indices.max(dim=1)[0]
        target_bars = (torch.rand(b, device=device) * (max_bars.float() + 1)).long()
        target_bars_exp = target_bars.unsqueeze(1).expand(-1, seq_len)
        
        avail_attrs = torch.tensor([3, 4, 5, 7], device=device)
        rand_indices = (torch.rand(b, device=device) * 4).long()
        selected_attrs = avail_attrs[rand_indices]
        selected_attrs_exp = selected_attrs.unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)
        feature_indices = torch.arange(8, device=device).unsqueeze(0).unsqueeze(0).expand(b, seq_len, 8)
        
        attr_mask = (feature_indices == selected_attrs_exp)
        mask = (bar_indices == target_bars_exp).unsqueeze(-1) & attr_mask

    elif strategy == '2_bar_attribute':
        bar_indices = x_0[:, :, 0]
        max_bars = bar_indices.max(dim=1)[0]
        r1 = (torch.rand(b, device=device) * (max_bars.float() + 1)).long()
        r2 = (torch.rand(b, device=device) * (max_bars.float() + 1)).long()
        
        target_bars1 = r1.unsqueeze(1).expand(-1, seq_len)
        target_bars2 = r2.unsqueeze(1).expand(-1, seq_len)
        
        avail_attrs = torch.tensor([3, 4, 5, 7], device=device)
        rand_indices = (torch.rand(b, device=device) * 4).long()
        selected_attrs = avail_attrs[rand_indices]
        selected_attrs_exp = selected_attrs.unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)
        feature_indices = torch.arange(8, device=device).unsqueeze(0).unsqueeze(0).expand(b, seq_len, 8)
        
        attr_mask = (feature_indices == selected_attrs_exp)
        mask = ((bar_indices == target_bars1) | (bar_indices == target_bars2)).unsqueeze(-1) & attr_mask
        
    elif strategy == 'rand_attribute':
        avail_attrs = torch.tensor([3, 4, 5, 7], device=device)
        rand_indices = (torch.rand(b, device=device) * 4).long()
        selected_attrs = avail_attrs[rand_indices]
        selected_attrs_exp = selected_attrs.unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)
        feature_indices = torch.arange(8, device=device).unsqueeze(0).unsqueeze(0).expand(b, seq_len, 8)
        
        mask = (feature_indices == selected_attrs_exp)

    return mask

def main():
    H = get_sampler_hparams('sample')
    
    print(f"Loading model: {H.model}")
    print(f"Masking Strategy for sampling: {H.masking_strategy}")

    sampler = get_sampler(H).cuda()
    
    # Load checkpoint
    # If load_step is not provided, try to find latest
    if H.load_step > 0:
        print(f"Loading checkpoint step: {H.load_step}")
        sampler = load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
    else:
        # Simple auto-discovery
        log_dir = H.load_dir if H.load_dir else f"logs/log_{H.model}_{H.tracks}_{H.NOTES}/saved_models"
        print(f"Looking for checkpoints in: {log_dir}")
        if os.path.exists(log_dir):
            import re
            pattern = re.compile(r"absorbing_ema_(\d+)\.th")
            max_step = 0
            for f in os.listdir(log_dir):
                match = pattern.match(f)
                if match:
                    step = int(match.group(1))
                    if step > max_step:
                        max_step = step
            
            if max_step > 0:
                print(f"Found latest checkpoint: {max_step}")
                sampler = load_model(sampler, f'{H.sampler}_ema', max_step, H.load_dir) # load_model prepends log_dir logic internally?
                # Actually load_model takes log_dir
                # load_model(model, name, step, log_dir) -> torch.load(f'{log_dir}/saved_models/{name}_{step}.th')
                # So we should pass H.load_dir (argument) or constructed path?
                # set_up_hparams -> H.log_dir is constructed.
                # If H.load_dir is None, H.log_dir is used.
                pass 
            else:
                 print("No checkpoint found! Initializing random weights (WARNING: Output will be noise).")
        else:
            print(f"Log directory not found: {log_dir}")
            print("Initializing random weights (WARNING: Output will be noise).")

    sampler.eval()

    # Load Dataset
    print(f"Loading dataset from: {H.dataset_path}")
    if os.path.isdir(H.dataset_path):
        midi_data = OctupleDataset(H.dataset_path, H.NOTES)
    else:
        # Fallback for simple .npy
        data = np.load(H.dataset_path, allow_pickle=True)
        # We need a wrapper similar to OctupleDataset to get items
        # Assuming data is list of tensors
        class SimpleDataset:
             def __init__(self, data): self.data = data
             def __len__(self): return len(self.data)
             def __getitem__(self, idx): return self.data[idx]
        midi_data = SimpleDataset(data)

    print(f"Dataset size: {len(midi_data)}")
    
    # Select Random Samples
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    indices = np.random.choice(len(midi_data), H.n_samples, replace=False)
    
    os.makedirs("inpainting_results", exist_ok=True)
    
    for i, idx in enumerate(indices):
        print(f"Processing sample {i+1}/{H.n_samples} (Index: {idx})")
        
        # Get Sample
        item = midi_data[idx]
        if isinstance(item, tuple): # OctupleDataset might return tensor
             # Check OctupleDataset implementation: returns torch.from_numpy(self.data[idx]).long()
             pass
        
        x_0 = torch.tensor(item).unsqueeze(0).cuda().long() # (1, L, 8)
        
        # Apply Mask
        mask = apply_mask(x_0, H.masking_strategy)
        
        # Create x_T (Conditional Input)
        x_T = x_0.clone()
        # Set masked tokens to mask_id
        # mask_id for each channel. 
        # sampler.mask_id is tensor (8,)
        
        for k in range(8):
             # channel k mask
             m_k = mask[:, :, k]
             x_T[:, :, k][m_k] = sampler.mask_id[k]
        
        # Save Original
        ns_orig = np_to_ns(x_0.cpu().numpy()[0]) 
        # np_to_ns expects (L, 8)
        # Note: np_to_ns implementation in sampler_utils handles octuple?
        # Let's check sampler_utils.np_to_ns
        # It handles 'melody' (1 shape) or 'trio' (3 shape).
        # Octuple has 8. I need to check how to decode Octuple.
        # models/octuple.py has OctupleEncoding.decode.
        # But sampler_utils.np_to_ns seems to use OneHotMelodyConverter?
        # I need to use OctupleDecoding.
        
        # For simplicity, I'll rely on the existing utils if they support it, 
        # OR I should check `sample.py` again.
        # `sample.py`: `ns = np_to_ns(sa)`
        # `np_to_ns` in `sampler_utils.py`:
        #     if x.shape[-1] == 1: ...
        #     elif x.shape[-1] == 3: ...
        #     else: raise Exception(f"unsupported number of tracks: {x.shape[-1]}")
        # So `sample.py` DOES NOT support Octuple out of the box?
        # `octuple_demo.ipynb` likely has the decoding logic.
        
        # RE-CHECK NEEDED: How to decode Octuple?
        pass
        
        # Perform Inpainting
        print("Running diffusion...")
        with torch.no_grad():
            sample_out = get_samples(sampler, H.sample_steps, x_T=x_T) # Returns numpy (B, L, 8)
        
        # Save Results
        from utils.octuple import OctupleEncoding
        encoder = OctupleEncoding()
        
        # Save Original Helper
        def save_midi(tokens, filename):
             midi_obj = encoder.decode(tokens)
             midi_obj.dump(filename)
             print(f"Saved {filename}")

        save_midi(x_0.cpu().numpy()[0], f"inpainting_results/sample_{i}_original.mid")
        
        # Save Masked (Might be broken if mask_id is invalid for MIDI, but worth a try or skip)
        # x_T has mask_id (usually large int) which might break decoder if not handled.
        # So we skip saving masked MIDI, or save it if we want to debug.
        # Let's save it as NPY for debugging mask structure if needed.
        np.save(f"inpainting_results/sample_{i}_masked.npy", x_T.cpu().numpy())
        
        # Save Inpainted
        save_midi(sample_out[0], f"inpainting_results/sample_{i}_inpainted.mid")


    print("Done.")

