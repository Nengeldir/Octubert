import subprocess
import sys

# Models to train
models = ["conv_transformer", "transformer", "hierarch_transformer", "U_transformer"]

# Common arguments
dataset = "data/POP909_melody.npy"
bars = "64"
batch_size = "64"
tracks = "melody"

def train_model(model_name):
    print(f"==============================================")
    print(f"Starting training for model: {model_name}")
    print(f"==============================================")
    
    cmd = [
        sys.executable, "train.py",
        "--dataset", dataset,
        "--bars", bars,
        "--batch_size", batch_size,
        "--tracks", tracks,
        "--model", model_name
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished training for model: {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to train model: {model_name}")
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    print("----------------------------------------------")

if __name__ == "__main__":
    for model in models:
        train_model(model)

    print("All training tasks completed.")
