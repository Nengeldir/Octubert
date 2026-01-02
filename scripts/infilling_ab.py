import numpy as np
from pathlib import Path
from utils.sampler_utils import np_to_ns
import note_seq

model = "log_octuple_mixed_melody_1024"  # change per model
cache = Path(f"logs/{model}/cache")
out = Path("logs/infilling/listening") / model
out.mkdir(parents=True, exist_ok=True)

refs = np.load(cache / "refs.npy")      # shape (B, 1024, 8) or melody
samps = np.load(cache / "samples.npy")  # model outputs
gap_start, gap_end = 256, 512  # bars 16–32 at 16 steps/bar

def save_ns(arr, path):
    ns = np_to_ns(arr)
    note_seq.sequence_proto_to_midi_file(ns, path)

for i in range(min(8, len(refs))):  # export first 8
    ref = refs[i].copy()
    gen = samps[i].copy()
    masked = ref.copy()
    masked[gap_start:gap_end, 3:6] = 0  # zero pitch/dur/vel in gap
    save_ns(ref, out / f"ref_{i}.mid")
    save_ns(masked, out / f"masked_{i}.mid")
    save_ns(gen, out / f"gen_{i}.mid")
    print(f"wrote trio {i}")

print("Compare masked_i.mid → gen_i.mid vs ref_i.mid")