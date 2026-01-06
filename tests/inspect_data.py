
import argparse
import os
import sys
import random
import numpy as np
from note_seq import note_sequence_to_midi_file

# Ensure we can import from project root
sys.path.append(os.getcwd())

from preprocessing.data import OneHotMelodyConverter, TrioConverter


def load_data(file_path):
    """Loads the .npy data file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    print(f"Loading {file_path}...")
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Loaded data with {len(data)} items.")
        return data
    except Exception as e:
        print(f"Failed to load file: {e}")
        return None


def get_converter(mode):
    """Returns the appropriate data converter based on mode."""
    if mode == 'melody':
        return OneHotMelodyConverter(slice_bars=None, gap_bars=None)
    elif mode == 'trio':
        return TrioConverter(slice_bars=None, gap_bars=None)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def save_sample(item, converter, output_path, idx):
    """Converts a single data item to MIDI and saves it."""
    if not isinstance(item, np.ndarray):
        print(f"Sample {idx} is not an ndarray, skipping (type: {type(item)}).")
        return

    # Data items are usually flattened arrays of indices or one-hot structures.
    # The converter expects a list of samples.
    # We flatten the item to ensure it's in the expected format (e.g. 1D array of indices).
    try:
        ns_list = converter.from_tensors([item.flatten()])
        if not ns_list:
            print(f"Sample {idx}: Converter returned no NoteSequences.")
            return

        ns = ns_list[0]
        note_sequence_to_midi_file(ns, output_path)
        print(f"Saved {output_path} (Notes: {len(ns.notes)})")
    except Exception as e:
        print(f"Error converting sample {idx}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Inspect prepared training data (NPY to MIDI)')
    parser.add_argument('--file', type=str, required=True, help='Path to input .npy file (e.g. data/POP909_melody.npy)')
    parser.add_argument('--mode', type=str, default='melody', choices=['melody', 'trio'], help='Data mode (melody or trio)')
    parser.add_argument('--samples', type=int, default=5, help='Number of random samples to inspect')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--out_dir', type=str, default='inspected_data', help='Directory to save output MIDI files')

    args = parser.parse_args()

    random.seed(args.seed)

    data = load_data(args.file)
    if data is None or len(data) == 0:
        print("No data to inspect.")
        return

    converter = get_converter(args.mode)

    # Select random samples
    num_samples = min(args.samples, len(data))
    sample_indices = random.sample(range(len(data)), num_samples)
    print(f"Selecting {len(sample_indices)} random samples...")

    os.makedirs(args.out_dir, exist_ok=True)

    for idx in sample_indices:
        item = data[idx]
        output_filename = f"inspected_{args.mode}_{idx}.mid"
        output_path = os.path.join(args.out_dir, output_filename)
        save_sample(item, converter, output_path, idx)


if __name__ == "__main__":
    main()
