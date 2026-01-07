from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ..preprocessing import OneHotMelodyConverter, TrioConverter
from ..data.octuple import OctupleEncoding


@dataclass(frozen=True)
class TokenizerSpec:
    id: str
    description: str
    factory: Callable[[], object]
    available: bool = True
    notes: Optional[str] = None


TOKENIZER_REGISTRY: Dict[str, TokenizerSpec] = {
    "melody_onehot": TokenizerSpec(
        id="melody_onehot",
        description="One-hot melody converter (1 track)",
        factory=lambda: OneHotMelodyConverter(),
    ),
    "trio_onehot": TokenizerSpec(
        id="trio_onehot",
        description="One-hot trio converter (3 tracks)",
        factory=lambda: TrioConverter(),
    ),
    "octuple": TokenizerSpec(
        id="octuple",
        description="Octuple MIDI encoding (8-tuple tokens)",
        factory=lambda: OctupleEncoding(),
    ),
}


def resolve_tokenizer_id(tokenizer_id: str) -> TokenizerSpec:
    key = tokenizer_id.strip().lower()
    if key not in TOKENIZER_REGISTRY:
        known = ", ".join(sorted(TOKENIZER_REGISTRY.keys()))
        raise ValueError(f"Unknown tokenizer id '{tokenizer_id}'. Known: {known}")
    spec = TOKENIZER_REGISTRY[key]
    if not spec.available:
        raise ValueError(f"Tokenizer '{tokenizer_id}' not available: {spec.notes or 'N/A'}")
    return spec
