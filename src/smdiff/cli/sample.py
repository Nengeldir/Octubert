import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from note_seq import note_sequence_to_midi_file

from hparams.set_up_hparams import get_sampler_hparams
from smdiff.utils.log_utils import load_model
from smdiff.data.octuple import OctupleEncoding
from smdiff.utils.sampler_utils import get_samples, np_to_ns, get_sampler


from smdiff.registry import resolve_model_id
from smdiff.tasks import resolve_task_id
from smdiff.masking import resolve_masking_id
from smdiff.tokenizers import resolve_tokenizer_id, TOKENIZER_REGISTRY
from smdiff.configs.loader import load_config
from smdiff.data import apply_dataset_to_config

# Reuse masking logic from the legacy infilling script
from sample_inpainting import apply_mask


def build_underlying_argv(cfg: Dict, ns: argparse.Namespace) -> List[str]:
    """Translate unified CLI args to the legacy hparams parser argv."""
    spec = resolve_model_id(ns.model)

    def pick(key, default=None):
        val = getattr(ns, key, None)
        return val if val is not None else cfg.get(key, default)

    args = [
        "--model", spec.internal_model,
        "--dataset_path", pick("dataset_path"),
        "--batch_size", str(pick("batch_size", 16)),
        "--n_samples", str(pick("n_samples", 10)),
        "--bars", str(pick("bars", 64)),
        "--tracks", pick("tracks", "melody"),
    ]

    # Optional flags/values
    if pick("masking_strategy"):
        args += ["--masking_strategy", pick("masking_strategy")]
    if pick("load_dir"):
        args += ["--load_dir", pick("load_dir")]
    if pick("load_step"):
        args += ["--load_step", str(pick("load_step"))]
    if pick("log_base_dir"):
        args += ["--log_base_dir", pick("log_base_dir")]
    if pick("port"):
        args += ["--port", str(pick("port"))]
    if pick("amp", False):
        args += ["--amp"]
    if pick("ema", True):
        args += ["--ema"]

    return args


def load_hparams_for_sample(cfg: Dict, ns: argparse.Namespace):
    translated_argv = [sys.argv[0]] + build_underlying_argv(cfg, ns)
    prev_argv = sys.argv
    sys.argv = translated_argv
    try:
        H = get_sampler_hparams('sample')
    finally:
        sys.argv = prev_argv
    return H


def decode_and_save_sequence(tokens: np.ndarray, out_path: str, tokenizer_id: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    spec = TOKENIZER_REGISTRY[tokenizer_id]
    # Lightweight decode based on tokenizer id; reuse OctupleEncoding for octuple
    if tokenizer_id == "octuple":
        encoder = OctupleEncoding()
        midi_obj = encoder.decode(tokens)
        midi_obj.dump(out_path) # type: ignore
    else:
        # One-hot family decoders use np_to_ns for now
        note_sequence_to_midi_file(np_to_ns(tokens)[0], out_path)


def run_unconditional(H, outdir: str, tokenizer_id: str):
    H.sample_schedule = "rand"
    sampler = get_sampler(H).cuda()
    sampler = load_model(sampler, f"{H.sampler}_ema", H.load_step, H.load_dir)
    sampler.eval()

    n_done = 0
    while n_done < H.n_samples:
        sa = get_samples(sampler, H.sample_steps)
        for sample_idx, sample_tokens in enumerate(sa):
            out_path = os.path.join(outdir, f"sample_{n_done + sample_idx}.mid")
            decode_and_save_sequence(sample_tokens, out_path, tokenizer_id)
            n_done += 1
            if n_done >= H.n_samples:
                break
        print(f"{n_done}/{H.n_samples} saved to {outdir}")


def build_dataset(H):
    if os.path.isdir(H.dataset_path):
        return OctupleDataset(H.dataset_path, H.NOTES)
    data = np.load(H.dataset_path, allow_pickle=True)

    class SimpleDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    return SimpleDataset(data)


def run_infill(H, outdir: str, tokenizer_id: str):
    if H.masking_strategy is None:
        raise ValueError("--masking_strategy is required for infill task")
    resolve_masking_id(H.masking_strategy)

    sampler = get_sampler(H).cuda()
    sampler = load_model(sampler, f"{H.sampler}_ema", H.load_step, H.load_dir)
    sampler.eval()

    dataset = build_dataset(H)
    n = min(H.n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)

    os.makedirs(outdir, exist_ok=True)
    encoder = OctupleEncoding()

    for i, idx in enumerate(indices):
        item = dataset[idx]
        x_0 = torch.tensor(item).unsqueeze(0).cuda().long()
        mask = apply_mask(x_0, H.masking_strategy)

        x_T = x_0.clone()
        for k in range(x_T.shape[-1]):
            m_k = mask[:, :, k]
            x_T[:, :, k][m_k] = sampler.mask_id[k]

        with torch.no_grad():
            sample_out = get_samples(sampler, H.sample_steps, x_T=x_T)

        orig_path = os.path.join(outdir, f"sample_{i}_original.mid")
        infill_path = os.path.join(outdir, f"sample_{i}_infill.mid")
        masked_path = os.path.join(outdir, f"sample_{i}_masked.npy")

        if tokenizer_id == "octuple":
            midi_orig = encoder.decode(x_0.cpu().numpy()[0])
            midi_orig.dump(orig_path) # type: ignore
        else:
            decode_and_save_sequence(x_0.cpu().numpy()[0], orig_path, tokenizer_id)

        np.save(masked_path, x_T.cpu().numpy())

        if tokenizer_id == "octuple":
            midi_infill = encoder.decode(sample_out[0])
            midi_infill.dump(infill_path) # type: ignore
        else:
            decode_and_save_sequence(sample_out[0], infill_path, tokenizer_id)
        print(f"[{i+1}/{n}] saved infill result -> {infill_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified sampling CLI (tasks only affect sampling/eval)")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional experiment config YAML to merge")
    parser.add_argument("--set", action="append", default=[],
                        help="Override config keys, e.g. --set n_samples=8")
    parser.add_argument("--model", required=True, type=str,
                        help="Model id: schmu_conv_vae | schmu_tx_vae | octuple_ddpm | octuple_mask_ddpm | musicbert_ddpm")
    parser.add_argument("--task", required=True, type=str, choices=["uncond", "infill"],
                        help="Sampling task")
    parser.add_argument("--dataset_id", type=str, default=None,
                        help="Dataset id from DATASET_REGISTRY (e.g., pop909_melody, pop909_octuple)")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--tracks", type=str, default=None)
    parser.add_argument("--masking_strategy", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="samples")

    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--load_step", type=int, default=None)
    parser.add_argument("--log_base_dir", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--ema", action="store_true", default=None)

    ns = parser.parse_args()

    resolve_task_id(ns.task)  # validate early
    cfg = load_config(ns.model, ns.config, ns.set)
    if ns.dataset_id:
        cfg = apply_dataset_to_config(cfg, ns.dataset_id)
    dataset_path = cfg.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set --dataset_id or --dataset_path to an existing location."
        )
    tokenizer_id = cfg.get("tokenizer_id") or cfg.get("tracks", "melody")
    resolve_tokenizer_id(tokenizer_id)

    H = load_hparams_for_sample(cfg, ns)
    H.tokenizer_id = tokenizer_id
    if not H.load_dir:
        H.load_dir = os.path.join("runs", ns.model)

    outdir = os.path.join(ns.outdir, ns.task, ns.model)

    if ns.task == "uncond":
        run_unconditional(H, outdir, tokenizer_id)
    else:
        run_infill(H, outdir, tokenizer_id)


if __name__ == "__main__":
    main()
