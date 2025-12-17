
import subprocess
import sys
import os
import re
import platform
import shutil

# Strategies to train
strategies = [
    '1_bar_all',
    '2_bar_all',
    '1_bar_attribute',
    '2_bar_attribute',
    'rand_attribute'
]

# Base model name
base_model = "octuple"

# Common arguments
# OctupleDataset expects a directory containing .npy files
dataset = "data/processed" 
bars = "64"
batch_size = "4"
tracks = "melody" # Used for log directory naming
# Note: If epochs is set (!= None), it overrides train_steps
epochs = "150"
train_steps = "100000"
steps_per_log = "10"
steps_per_eval = "2000"
steps_per_sample = "2000"
steps_per_checkpoint = "500"

SCRATCH_DIR_BASE = "/work/scratch/lconconi"

def is_cluster():
    """
    Detects if running on the cluster. 
    Heuristic: Checks if the OS is Ubuntu.
    """
    try:
        # Check /etc/os-release which is standard on modern Linux
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                content = f.read()
                if "Ubuntu" in content or "ID=ubuntu" in content:
                    return True
        
        # Fallback check
        if "Ubuntu" in platform.version():
            return True
            
    except Exception:
        pass
        
    return False

def get_latest_checkpoint(model_name):
    # Calculate NOTES as per default_hparams.py logic
    # bars is a string in the global vars, convert to int
    n_bars = int(bars)
    notes = n_bars * 16
    
    # Construct log directory name
    log_dir_name = f"log_{model_name}_{tracks}_{notes}"
    # We always look for checkpoints in the permanent storage "logs/" 
    # because scratch is purged.
    log_dir_path = os.path.join("logs", log_dir_name, "saved_models")
    
    if not os.path.exists(log_dir_path):
        return 0, log_dir_name
        
    max_step = 0
    # Pattern to match checkpoint files, e.g., absorbing_500.th
    # H.sampler is "absorbing" for HparamsOctuple (inherited from HparamsAbsorbing)
    pattern = re.compile(r"absorbing_(\d+)\.th")
    
    for filename in os.listdir(log_dir_path):
        match = pattern.match(filename)
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                
    return max_step, log_dir_name

def train_model(strategy):
    # Construct specific model name for this strategy
    # This ensures unique log directories: log_octuple_1_bar_all_melody_1024
    model_name = f"{base_model}_{strategy}"
    
    print(f"==============================================")
    print(f"Starting training for strategy: {strategy}")
    print(f"Model Name: {model_name}")
    print(f"==============================================")
    
    load_step, log_dir_name = get_latest_checkpoint(model_name)
    
    on_cluster = is_cluster()
    log_base_dir = None
    
    if on_cluster:
        log_base_dir = SCRATCH_DIR_BASE
        print(f"[CLUSTER DETECTED] Running on Ubuntu cluster.")
        print(f"[CLUSTER DETECTED] Output will be redirected to scratch: {log_base_dir}")
        
        # Ensure scratch directory exists
        if not os.path.exists(log_base_dir):
            try:
                os.makedirs(log_base_dir, exist_ok=True)
                print(f"[CLUSTER DETECTED] Created scratch directory: {log_base_dir}")
            except Exception as e:
                print(f"[CLUSTER ERROR] Failed to create scratch directory: {e}")
    
    cmd = [
        sys.executable, "train.py",
        "--dataset", dataset,
        "--bars", bars,
        "--batch_size", batch_size,
        "--tracks", tracks,
        "--model", model_name,
        "--masking_strategy", strategy, # Pass the strategy
        "--epochs", epochs,
        "--train_steps", train_steps,
        "--steps_per_log", steps_per_log,
        "--steps_per_eval", steps_per_eval,
        "--steps_per_sample", steps_per_sample,
        "--steps_per_checkpoint", steps_per_checkpoint,
        "--amp",
    ]
    
    if log_base_dir:
        cmd.extend(["--log_base_dir", log_base_dir])

    if load_step > 0:

        print(f"Resuming from checkpoint: step {load_step} in {log_dir_name}")
        cmd.extend(["--load_step", str(load_step)])
        cmd.extend(["--load_dir", log_dir_name])
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished training for strategy: {strategy}")
        
        if on_cluster:
            print(f"[CLUSTER DETECTED] Syncing to permanent storage is handled automatically by log_utils.")

    except subprocess.CalledProcessError as e:
        print(f"Failed to train strategy: {strategy}")
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    print("----------------------------------------------")

if __name__ == "__main__":
    for strategy in strategies:
        train_model(strategy)

    print("All strategy training tasks completed.")

