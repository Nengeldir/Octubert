from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    description: str
    dataset_path: str
    tracks: str
    bars: int
    notes: int
    tokenizer_id: str
    is_directory: bool = False
    available: bool = True
    notes_txt: Optional[str] = None


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "pop909_melody": DatasetSpec(
        id="pop909_melody",
        description="POP909 melody combined .npy",
        dataset_path="data/POP909_melody.npy",
        tracks="melody",
        bars=64,
        notes=1024,
        tokenizer_id="melody_onehot",
        is_directory=False,
    ),
    "pop909_octuple": DatasetSpec(
        id="pop909_octuple",
        description="POP909 Octuple encoding per-file",
        dataset_path="data/POP909/processed",
        tracks="octuple",
        bars=64,
        notes=1024,
        tokenizer_id="octuple",
        is_directory=True,
    ),
    # Placeholder for trio once built
    "pop909_trio": DatasetSpec(
        id="pop909_trio",
        description="POP909 trio combined .npy",
        dataset_path="data/POP909_trio.npy",
        tracks="trio",
        bars=64,
        notes=1024,
        tokenizer_id="trio_onehot",
        is_directory=False,
        available=True,
        notes_txt="Generate via prepare_data.py --mode trio --target data/POP909_trio.npy",
    ),
}


def resolve_dataset_id(dataset_id: str) -> DatasetSpec:
    key = dataset_id.strip().lower()
    if key not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset id '{dataset_id}'. Known: {known}")
    spec = DATASET_REGISTRY[key]
    if not spec.available:
        raise ValueError(f"Dataset '{dataset_id}' not available: {spec.notes_txt or 'N/A'}")
    return spec


def apply_dataset_to_config(cfg: Dict, dataset_id: str) -> Dict:
    spec = resolve_dataset_id(dataset_id)
    updated = dict(cfg)
    updated.update({
        "dataset_path": spec.dataset_path,
        "tracks": spec.tracks,
        "bars": spec.bars,
        "NOTES": spec.notes,
        "tokenizer_id": spec.tokenizer_id,
    })
    return updated
