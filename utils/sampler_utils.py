import numpy as np
import torch
import torch.distributions as dists
from torch.nn import DataParallel

# from .log_utils import save_latents, log
from models import Transformer, AbsorbingDiffusion, ConVormer, HierarchTransformer, UTransformer
from preprocessing import OneHotMelodyConverter, TrioConverter
from utils.octuple import OctupleEncoding
from note_seq import midi_to_note_sequence


def get_sampler(H):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if H.model == 'transformer' or H.model.startswith('octuple'):
        denoise_fn = Transformer(H)
    elif H.model == 'hierarch_transformer':
        denoise_fn = HierarchTransformer(H)
    elif H.model == 'U_transformer':
        denoise_fn = UTransformer(H)
    else:
        denoise_fn = ConVormer(H)

    denoise_fn = denoise_fn.to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        denoise_fn = DataParallel(denoise_fn)

    sampler = AbsorbingDiffusion(H, denoise_fn, H.codebook_size).to(device)
    return sampler


@torch.no_grad()
def get_samples(sampler, sample_steps, x_T=None, temp=1.0, b=None, progress_handler=None):
    sampler.eval()

    if x_T is not None and not torch.is_tensor(x_T):
        device = next(sampler.parameters()).device
        x_T = torch.as_tensor(x_T, device=device, dtype=torch.long)

    result = sampler.sample(sample_steps=sample_steps, x_T=x_T, temp=temp, B=b, progress_handler=progress_handler)
    return result.cpu().numpy()


def np_to_ns(x):
    if x.shape[-1] == 1:
        converter = OneHotMelodyConverter()
        return converter.from_tensors(x.squeeze(-1))
    elif x.shape[-1] == 3:
        converter = TrioConverter()
        return converter.from_tensors(x)
    elif x.shape[-1] == 8:
        enc = OctupleEncoding()
        seqs = []
        import tempfile, os
        for sample in x:
            midi_obj = enc.decode(sample)
            tmp = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
            tmp_path = tmp.name
            tmp.close()  # Windows needs the handle closed before reuse
            try:
                midi_obj.dump(tmp_path)
                with open(tmp_path, 'rb') as fh:
                    seqs.append(midi_to_note_sequence(fh.read()))
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return seqs
    else:
        raise Exception(f"unsupported number of tracks: {x.shape[-1]}")


def ns_to_np(ns, bars, mode='melody'):
    if mode == 'melody':
        converter = OneHotMelodyConverter(slice_bars=bars)
    else:
        converter = TrioConverter(slice_bars=bars)
    return converter.to_tensors(ns)
