from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class MaskingSpec:
    id: str
    description: str
    available: bool = True
    notes: Optional[str] = None


MASKING_REGISTRY: Dict[str, MaskingSpec] = {
    "1_bar_all": MaskingSpec(id="1_bar_all", description="Mask all attributes in one random bar"),
    "2_bar_all": MaskingSpec(id="2_bar_all", description="Mask all attributes in two random bars"),
    "1_bar_attribute": MaskingSpec(id="1_bar_attribute", description="Mask one attribute in one random bar"),
    "2_bar_attribute": MaskingSpec(id="2_bar_attribute", description="Mask one attribute in two random bars"),
    "rand_attribute": MaskingSpec(id="rand_attribute", description="Mask one attribute across the whole sequence"),
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
