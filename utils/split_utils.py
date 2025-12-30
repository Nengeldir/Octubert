"""Split loading helpers for deterministic dataset filtering."""

import json
import os
from typing import List

import numpy as np


def _expand_ids(raw_ids) -> List[str]:
    """Expand mixed list of ids or ranges into zero-padded strings.

    Accepts entries like "001", 1, or "001-010".
    """
    ids: List[str] = []
    for item in raw_ids:
        if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
            ids.append(f"{int(item):03d}")
        elif isinstance(item, str) and "-" in item:
            lo, hi = item.split("-")
            lo_i, hi_i = int(lo), int(hi)
            step = 1 if hi_i >= lo_i else -1
            ids.extend([f"{i:03d}" for i in range(lo_i, hi_i + step, step)])
        else:
            raise ValueError(f"Unsupported split id entry: {item}")
    return ids


def load_split_ids(split_path: str, partition: str) -> List[str]:
    """Load ids for a given split partition (train/val/test)."""
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    if partition in split:
        raw = split[partition]
    elif "ranges" in split and partition in split["ranges"]:
        raw = split["ranges"][partition]
    else:
        raise KeyError(f"Partition {partition} not found in split file {split_path}")

    return _expand_ids(raw)


def load_processed_subset(processed_dir: str, ids: List[str], pad_length: int = 1024) -> np.ndarray:
    """Load and stack per-song .npy files for the selected ids.

    Pads/truncates each song to pad_length to ensure consistent shapes.
    """
    arrays = []
    for i in ids:
        path = os.path.join(processed_dir, f"{i}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing processed file: {path}")
        arr = np.load(path, allow_pickle=True)
        
        # Pad or truncate to consistent length
        if arr.shape[0] < pad_length:
            # Pad with zeros
            pad_width = ((0, pad_length - arr.shape[0]),) + tuple((0, 0) for _ in range(arr.ndim - 1))
            arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
        else:
            # Truncate
            arr = arr[:pad_length]
        
        arrays.append(arr)
    
    return np.stack(arrays)
