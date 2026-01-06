
import os
import numpy as np
import tqdm

def inspect_data():
    data_dir = "data/processed"
    # Codebook size from HparamsOctuple
    limits = (256, 128, 129, 256, 128, 32, 256, 49)
    
    print(f"Checking files in {data_dir} against limits {limits}")
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    max_vals = [0] * 8
    min_vals = [9999] * 8
    
    for f in tqdm.tqdm(files):
        path = os.path.join(data_dir, f)
        try:
            arr = np.load(path)
            # arr shape should be (T, 8)
            if len(arr.shape) != 2 or arr.shape[1] != 8:
                print(f"Skipping {f}, shape {arr.shape}")
                continue
                
            for i in range(8):
                mx = arr[:, i].max()
                mn = arr[:, i].min()
                if mx > max_vals[i]:
                    max_vals[i] = mx
                if mn < min_vals[i]:
                    min_vals[i] = mn
                
                if mx >= limits[i]:
                    print(f"ALARM: File {f} has value {mx} in track {i} (limit {limits[i]})")
                if mn < 0:
                     print(f"ALARM: File {f} has negative value {mn} in track {i}")
                     
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print("Max values found:", max_vals)
    print("Min values found:", min_vals)
    print("Limits:", limits)

if __name__ == "__main__":
    inspect_data()
