"""Prepare octuple-encoded datasets from MIDI files.

This script walks a MIDI directory, encodes each file with OctupleEncoding,
and writes one .npy per MIDI into a target directory, ready for OctupleDataset.
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..data.octuple import OctupleEncoding


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _pad_or_crop(arr: np.ndarray, seq_len: Optional[int]) -> np.ndarray:
    """Pad or crop an (N, 8) array to seq_len along axis 0."""
    if seq_len is None:
        return arr
    n = arr.shape[0]
    if n == seq_len:
        return arr
    if n > seq_len:
        return arr[:seq_len]
    pad = np.zeros((seq_len - n, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _process_one(args: Tuple[str, str, Optional[int]]):
    midi_path, out_path, seq_len = args
    encoder = OctupleEncoding()
    try:
        encoding = encoder.encode(midi_path)
        if encoding.size == 0:
            return False
        encoding = _pad_or_crop(encoding, seq_len).astype(np.int32, copy=False)
        np.save(out_path, encoding)
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare octuple .npy files from MIDI directory")
    parser.add_argument("--root_dir", type=str, default="data/POP909", help="Root directory containing MIDI files")
    parser.add_argument("--target_dir", type=str, default="data/octuple", help="Output directory for octuple .npy files")
    parser.add_argument("--seq_len", type=int, default=1024, help="Pad/crop sequences to this length (notes)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = all)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: min(32, cpu_count))")
    args = parser.parse_args()

    root = Path(args.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(list(root.rglob("*.mid")) + list(root.rglob("*.midi")))
    if args.limit and args.limit > 0:
        midi_files = midi_files[: args.limit]

    if not midi_files:
        print("No MIDI files found.")
        return

    workers = args.workers or min(32, os.cpu_count() or 4)
    print(f"Processing {len(midi_files)} MIDI files with {workers} workers...")

    tasks = []
    for midi_path in midi_files:
        out_name = midi_path.stem + ".npy"
        out_path = target / out_name
        tasks.append((str(midi_path), str(out_path), args.seq_len))

    success = 0
    with Pool(processes=workers) as pool:
        for ok in tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks)):
            if ok:
                success += 1

    print(f"Done. Saved {success}/{len(tasks)} files to {target}")


if __name__ == "__main__":
    main()
