from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class MaskingSpec:
    id: str
    description: str
    available: bool = True
    notes: Optional[str] = None


MASKING_REGISTRY: Dict[str, MaskingSpec] = {
    # Note: In diffusion training we still apply time gating (t/T). The descriptions below
    # describe the *structural* masking unit; the final mask is gated by t/T.
    "random": MaskingSpec(
        id="random",
        description="Token-level masking: mask whole token (all channels) at random positions",
    ),
    "mixed": MaskingSpec(
        id="mixed",
        description="Randomly choose one masking strategy per batch (includes 'random')",
    ),
    "1_bar_all": MaskingSpec(
        id="1_bar_all",
        description="Bar-level masking: select 1 bar and mask the selected attribute set for that bar",
        notes="Implementation masks attributes {pitch,duration,velocity,tempo} (channels 3,4,5,7), gated by t/T.",
    ),
    "2_bar_all": MaskingSpec(
        id="2_bar_all",
        description="Bar-level masking: select 2 bars and mask the selected attribute set for those bars",
        notes="Implementation masks attributes {pitch,duration,velocity,tempo} (channels 3,4,5,7), gated by t/T. Bars may coincide.",
    ),
    "1_bar_attribute": MaskingSpec(
        id="1_bar_attribute",
        description="Bar-level masking: select 1 bar and mask 1 attribute across that entire bar",
        notes="Attribute is chosen from {pitch,duration,velocity,tempo} (channels 3,4,5,7), gated by t/T.",
    ),
    "2_bar_attribute": MaskingSpec(
        id="2_bar_attribute",
        description="Bar-level masking: select 2 bars and mask 1 attribute across those bars",
        notes="Attribute is chosen from {pitch,duration,velocity,tempo} (channels 3,4,5,7), gated by t/T. Bars may coincide.",
    ),
    "rand_attribute": MaskingSpec(
        id="rand_attribute",
        description="Attribute-level masking: select 1 attribute and mask it across the whole sequence",
        notes="Attribute is chosen from {pitch,duration,velocity,tempo} (channels 3,4,5,7), gated by t/T.",
    ),
}


def resolve_masking_id(masking_id: str) -> MaskingSpec:
    key = masking_id.strip().lower()
    if key not in MASKING_REGISTRY:
        known = ", ".join(sorted(MASKING_REGISTRY.keys()))
        raise ValueError(f"Unknown masking id '{masking_id}'. Known: {known}")
    spec = MASKING_REGISTRY[key]
    if not spec.available:
        raise ValueError(f"Masking '{masking_id}' not available: {spec.notes or 'N/A'}")
    return spec
