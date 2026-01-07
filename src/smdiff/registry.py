from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    id: str
    internal_model: str
    description: str
    available: bool = True
    notes: Optional[str] = None


# Canonical model IDs and mapping to current internal model strings
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # SchmuBERT (VAE-based) variants
    "schmu_conv_vae": ModelSpec(
        id="schmu_conv_vae",
        internal_model="conv_transformer",
        description="Conv_Transformer (VAE-based SchmuBERT)",
    ),
    "schmu_tx_vae": ModelSpec(
        id="schmu_tx_vae",
        internal_model="transformer",
        description="Transformer (VAE-based SchmuBERT)",
    ),

    # Octuple discrete diffusion
    # Using an 'octuple' prefix ensures existing code paths activate Octuple hparams
    "octuple_ddpm": ModelSpec(
        id="octuple_ddpm",
        internal_model="octuple_ddpm",
        description="Octuple MIDI + Transformer (discrete diffusion)",
    ),
    "octuple_mask_ddpm": ModelSpec(
        id="octuple_mask_ddpm",
        internal_model="octuple_mask_ddpm",
        description="Octuple MIDI + Partial Masking + Transformer (discrete diffusion)",
    ),

    # MusicBERT + Transformer (planned)
    # Not available in current codebase – guarded by 'available=False'.
    "musicbert_ddpm": ModelSpec(
        id="musicbert_ddpm",
        internal_model="musicbert_ddpm",
        description="MusicBERT + Transformer (discrete diffusion)"
    ),
}


def resolve_model_id(model_id: str) -> ModelSpec:
    key = model_id.strip().lower()
    if key not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model id '{model_id}'. Known: {known}")
    spec = MODEL_REGISTRY[key]
    if not spec.available:
        raise ValueError(
            f"Model '{model_id}' is not available yet. Notes: {spec.notes or 'N/A'}"
        )
    return spec
