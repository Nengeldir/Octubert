import numpy as np
import torch
from note_seq import note_sequence_to_midi_file

from hparams import get_sampler_hparams
from utils import get_sampler, load_model
from utils.sampler_utils import get_samples, np_to_ns


def sample_nogui(sampler, H):
    n_samples = 0
    sampler.sampling_batch_size = H.batch_size
    while n_samples < H.n_samples:
        sa = get_samples(sampler, H.sample_steps)
        ns = np_to_ns(sa)

        for n in ns:
            note_sequence_to_midi_file(n, f'data/out/{n_samples}.mid')
            n_samples += 1
        print(f'{n_samples}/{H.n_samples}')


if __name__ == '__main__':
    H = get_sampler_hparams('sample')
    H.sample_schedule = "rand"
    sampler = get_sampler(H).cuda()
    sampler = load_model(
                sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)

    sample_nogui(sampler, H)
