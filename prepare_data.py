import argparse
import itertools
import os
import warnings
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from note_seq import midi_to_note_sequence
from tqdm import tqdm

from preprocessing.data import TrioConverter, OneHotMelodyConverter



def process_midi_file(args):
    """Worker function for processing a single MIDI file."""
    midi_path, mode, bars, max_t_per_ns = args
    
    if mode == 'melody':
        converter = OneHotMelodyConverter(slice_bars=bars, max_tensors_per_notesequence=max_t_per_ns, gap_bars=None, presplit_on_time_changes=False)
    else:  # trio
        converter = TrioConverter(slice_bars=bars, max_tensors_per_notesequence=max_t_per_ns, gap_bars=None, presplit_on_time_changes=False)
        
    result = []
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # Read file content
            with open(midi_path, 'rb') as f:
                content = f.read()
            ns = midi_to_note_sequence(content)
            
            # Conversion returns a named tuple with 'outputs' being the list of tensors
            tensors = converter.to_tensors(ns).outputs
            result = list(tensors)
    except Exception as e:
        # Pass silently to avoid spamming logs during large dataset processing
        pass
    return result


def load_dataset(root_dir, mode='melody', bars=64, max_tensors_per_ns=5, cache_path=None, limit=0):
    """
    Loads and processes a dataset of MIDI files.
    
    Args:
        root_dir (str): Path to the root directory containing MIDI files (searched recursively).
        mode (str): 'melody' or 'trio'.
        bars (int): Number of bars per sequence.
        max_tensors_per_ns (int): Max tensors to extract per MIDI file.
        cache_path (str): Path to save/load the .npy cache.
        limit (int): Limit the number of files processed (0 for no limit).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        return np.load(cache_path, allow_pickle=True)

    print(f"Processing dataset from {root_dir} in '{mode}' mode...")
    root_path = Path(root_dir)
    
    # Verify root_dir exists
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    # Gather all MIDI files
    all_midis = sorted(root_path.rglob("*.mid"))
    
    if limit > 0:
        all_midis = all_midis[:limit]
        
    print(f"Found {len(all_midis)} MIDI files.")
    
    worker_args = [(str(m), mode, bars, max_tensors_per_ns) for m in all_midis]
    
    # Use multiprocessing
    num_processes = min(40, os.cpu_count() or 4)
    result = []
    
    with Pool(num_processes) as p:
        for file_res in tqdm(p.imap(process_midi_file, worker_args), total=len(worker_args)):
            result.extend(file_res)

    print(f"Extracted {len(result)} sequences.")
    
    if cache_path:
        print(f"Saving to {cache_path}...")
        np.save(cache_path, result)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare dataset for Octubert")
    parser.add_argument("--root_dir", type=str, default="data/POP909", help="Root directory of the dataset")
    parser.add_argument("--mode", type=str, default="melody", choices=['melody', 'trio'], help="Extraction mode")
    parser.add_argument("--target", type=str, default="data/POP909_melody.npy", help="Output .npy file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process")
    parser.add_argument("--bars", type=int, default=64, help="Sequence length in bars")

    args = parser.parse_args()

    load_dataset(
        root_dir=args.root_dir,
        mode=args.mode,
        bars=args.bars,
        cache_path=args.target,
        limit=args.limit
    )

