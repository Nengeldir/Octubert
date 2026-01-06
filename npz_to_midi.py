import argparse
import numpy as np
import os
from utils import samples_2_noteseq, save_noteseqs

def main():
    parser = argparse.ArgumentParser(description='Convert NPZ samples to MIDI')
    parser.add_argument('--file', type=str, required=True, help='Path to input .npz file')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    print(f"Loading {args.file}...")
    try:
        data = np.load(args.file, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            samples = data['arr_0']
        else:
            samples = data

        print(f"Loaded samples with shape: {samples.shape}")
        
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # Create output directory
    input_dir = os.path.dirname(args.file)
    basename = os.path.basename(args.file).replace('.npz', '')
    output_dir = os.path.join(input_dir, f"{basename}_midi")
    os.makedirs(output_dir, exist_ok=True)

    print("Converting to MIDI...")
    try:
        note_seqs = samples_2_noteseq(samples)
        
        for i, ns in enumerate(note_seqs):
            out_path = os.path.join(output_dir, f"sample_{i}.mid")
            save_noteseqs([ns], prefix=os.path.join(output_dir, f"sample"))
            
        save_noteseqs(note_seqs, prefix=os.path.join(output_dir, basename))
        print(f"Saved {len(note_seqs)} MIDI files to {output_dir}")
        
    except Exception as e:
        print(f"Error converting/saving: {e}")

if __name__ == "__main__":
    main()
