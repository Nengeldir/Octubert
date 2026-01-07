import os
from typing import Dict

import numpy as np

from .base import SimpleNpyDataset
from .octuple import OctupleDataset
from .registry import resolve_dataset_id


def load_dataset(cfg: Dict):
    """Load dataset based on dataset_id or dataset_path in cfg."""
    dataset_id = cfg.get("dataset_id")
    if dataset_id:
        spec = resolve_dataset_id(dataset_id)
        dataset_path = spec.dataset_path
        seq_len = spec.notes
    else:
        dataset_path = cfg.get("dataset_path")
        seq_len = cfg.get("NOTES")
        spec = None

    if dataset_path is None:
        raise ValueError("dataset_path or dataset_id must be provided")

    if not os.path.exists(dataset_path):
        if spec:
            raise FileNotFoundError(
                f"Dataset path not found for dataset_id='{dataset_id}': {dataset_path}"
            )
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if os.path.isdir(dataset_path):
        # Octuple directory case
        dataset = OctupleDataset(dataset_path, seq_len)
    else:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        data = np.load(dataset_path, allow_pickle=True)
        # Melody/trio npy combined
        dataset = SimpleNpyDataset(data, seq_len) # type: ignore

    return dataset, spec
