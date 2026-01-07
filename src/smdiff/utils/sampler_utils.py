import numpy as np
import torch
import torch.distributions as dists
from torch.nn import DataParallel

from ..models import Transformer, AbsorbingDiffusion, ConVormer, HierarchTransformer, UTransformer
from ..registry import resolve_model_id


# Map internal_model strings to model classes
_MODEL_CLASS_MAP = {
    'transformer': Transformer,
    'octuple_ddpm': Transformer,
    'octuple_mask_ddpm': Transformer,
    'conv_transformer': ConVormer,
    'hierarch_transformer': HierarchTransformer,
    'U_transformer': UTransformer,
}


def get_sampler(H):
    """
    Factory function to create a sampler based on model type.
    
    Uses the model registry to resolve model IDs to internal implementations.
    
    Args:
        H: Hyperparameters object with model configuration
        
    Returns:
        AbsorbingDiffusion sampler with appropriate denoising model
    """
    # Resolve model_id to get internal_model string
    try:
        model_spec = resolve_model_id(H.model)
        internal_model = model_spec.internal_model
    except (ValueError, AttributeError):
        # Fallback for legacy code paths that use internal names directly
        internal_model = H.model
    
    # Get the model class from the map
    model_class = _MODEL_CLASS_MAP.get(internal_model)
    if model_class is None:
        raise ValueError(
            f"Unknown internal model '{internal_model}'. "
            f"Known models: {', '.join(_MODEL_CLASS_MAP.keys())}"
        )
    
    denoise_fn = model_class(H).cuda()
    denoise_fn = DataParallel(denoise_fn).cuda()
    sampler = AbsorbingDiffusion(H, denoise_fn, H.codebook_size)

    return sampler


@torch.no_grad()
def get_samples(sampler, sample_steps, x_T=None, temp=1.0, b=None, progress_handler=None):
    sampler.eval()

    if x_T is not None and not torch.is_tensor(x_T):
        x_T = torch.tensor(x_T).to(next(sampler.parameters()).device)

    result = sampler.sample(sample_steps=sample_steps, x_T=x_T, temp=temp, B=b, progress_handler=progress_handler)
    return result.cpu().numpy()


def np_to_ns(x, tokenizer_id=None):
    """
    Convert numpy array to note_seq objects.
    
    Args:
        x: NumPy array of shape (batch, seq_len, tracks)
        tokenizer_id: Optional tokenizer ID (e.g., 'melody_onehot', 'trio_onehot')
                     If None, infers from shape
        
    Returns:
        list of note_seq.NoteSequence objects
    """
    from ..tokenizers.registry import resolve_tokenizer_id
    
    # Infer tokenizer from shape if not provided
    if tokenizer_id is None:
        if x.shape[-1] == 1:
            tokenizer_id = 'melody_onehot'
            x = x.squeeze(-1)  # Remove tracks dimension for melody
        elif x.shape[-1] == 3:
            tokenizer_id = 'trio_onehot'
        elif x.shape[-1] == 8:
            tokenizer_id = 'octuple'
        else:
            raise ValueError(f"Cannot infer tokenizer for shape {x.shape}. Please specify tokenizer_id.")
    
    tokenizer_spec = resolve_tokenizer_id(tokenizer_id)
    converter = tokenizer_spec.factory()
    return converter.from_tensors(x)


def ns_to_np(ns, bars, tokenizer_id='melody_onehot'):
    """
    Convert note_seq objects to numpy arrays.
    
    Args:
        ns: list of note_seq.NoteSequence objects
        bars: Number of bars to slice
        tokenizer_id: Tokenizer ID (e.g., 'melody_onehot', 'trio_onehot')
        
    Returns:
        NumPy array of tokenized sequences
    """
    from ..tokenizers.registry import TOKENIZER_REGISTRY
    from ..preprocessing import OneHotMelodyConverter, TrioConverter
    
    # Get converter with slice_bars parameter
    # Note: OctupleEncoding doesn't use slice_bars, only the music converters do
    if tokenizer_id == 'melody_onehot':
        converter = OneHotMelodyConverter(slice_bars=bars)
    elif tokenizer_id == 'trio_onehot':
        converter = TrioConverter(slice_bars=bars)
    elif tokenizer_id == 'octuple':
        # Octuple doesn't slice by bars - it encodes full sequences
        from ..data.octuple import OctupleEncoding
        converter = OctupleEncoding()
    else:
        raise ValueError(f"Unknown tokenizer_id: {tokenizer_id}")
    
    return converter.to_tensors(ns)
