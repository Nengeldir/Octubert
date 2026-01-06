import os
import torch
import numpy as np
import logging
from preprocessing.data import TrioConverter, OneHotMelodyConverter
from note_seq import note_sequence_to_midi_file


def log(output):
    logging.info(output)
    print(output)


def config_log(log_dir, filename="log.txt"):
    if not os.path.isabs(log_dir):
        log_dir = "logs/" + log_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, filename),
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )


def start_training_log(hparams):
    log("Using following hparams:")
    param_keys = list(hparams)
    param_keys.sort()
    for key in param_keys:
        log(f"> {key}: {hparams[key]}")



SCRATCH_DIR_BASE = "/work/scratch/lconconi"    # swap lconconi TODO
# Determine project root: utils/log_utils.py -> project_root/utils/log_utils.py -> project_root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME_LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def sync_to_home(scratch_path, log_dir):
    """
    If the file is in scratch space, sync it to the corresponding location in the home directory.
    Deletes older versions of the same file type in the destination to strictly keep only the latest.
    """
    if not scratch_path.startswith(SCRATCH_DIR_BASE):
        return

    # path is like /work/scratch/lconconi/log_name/.../chunk.ext
    # relative path from scratch base
    rel_path = os.path.relpath(scratch_path, SCRATCH_DIR_BASE)
    
    # Destination is HOME_LOGS_DIR/rel_path
    dest_path = os.path.join(HOME_LOGS_DIR, rel_path)
    dest_dir = os.path.dirname(dest_path)
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy the file
    import shutil
    import re
    
    try:
        shutil.copy2(scratch_path, dest_path)
        print(f"Synced {scratch_path} to {dest_path}")
        
        # Cleanup older files in destination
        # Heuristic: match base name and extension
        filename = os.path.basename(scratch_path)
        # Expected patterns: "absorbing_X.th", "samples_X.npz", "stats_X"
        
        # Determine prefix and extension
        if filename.endswith(".th"):
            # e.g. absorbing_500.th
            prefix = filename.rsplit('_', 1)[0] # absorbing
            ext = ".th"
            pattern = re.compile(rf"{prefix}_(\d+)\.th")
        elif filename.endswith(".npz"):
            # e.g. samples_500.npz
            prefix = "samples"
            ext = ".npz"
            pattern = re.compile(rf"samples_(\d+)\.npz")
        elif "stats" in filename:
             # e.g. stats_500 (no extension in save_stats?)
             # save_stats uses torch.save but doesn't assume extension?
             # logic in save_stats: save_path = f"{save_dir}/stats_{step}"
             prefix = "stats"
             ext = ""
             pattern = re.compile(rf"stats_(\d+)")
        else:
             return

        current_step_match = pattern.search(filename)
        if not current_step_match:
            return
        current_step = int(current_step_match.group(1))

        # Check existing files in dest_dir
        for f in os.listdir(dest_dir):
            if f == filename:
                continue
            
            match = pattern.match(f)
            if match:
                step = int(match.group(1))
                # If step is DIFFERENT (assumed older, or even newer if we are overwriting but shouldn't happen), delete it
                # We strictly want only the LATEST.
                if step < current_step:
                    full_p = os.path.join(dest_dir, f)
                    try:
                        os.remove(full_p)
                        print(f"Removed older file: {full_p}")
                    except OSError as e:
                        print(f"Error removing {full_p}: {e}")

    except Exception as e:
        print(f"Failed to sync {scratch_path} to home: {e}")


def save_model(model, model_save_name, step, log_dir):
    if not os.path.isabs(log_dir):
        log_dir = "logs/" + log_dir
    log_dir = log_dir + "/saved_models"
    os.makedirs(log_dir, exist_ok=True)
    model_name = f"{model_save_name}_{step}.th"
    print(f"Saving {model_save_name} to {model_save_name}_{str(step)}.th")
    save_path = os.path.join(log_dir, model_name)
    torch.save(model.state_dict(), save_path)
    
    sync_to_home(save_path, log_dir)


def load_model(model, model_load_name, step, log_dir, strict=True):
    print(f"Loading {model_load_name}_{str(step)}.th")
    if not os.path.isabs(log_dir):
        log_dir = "logs/" + log_dir
    log_dir = log_dir + "/saved_models"

    print(f"Loading {model_load_name}_{str(step)}.th from {log_dir}")
    try:
        model.load_state_dict(
            torch.load(os.path.join(log_dir, f"{model_load_name}_{step}.th")),
            strict=strict,
        )
    except TypeError:  # for some reason optimisers don't like the strict keyword
        model.load_state_dict(
            torch.load(os.path.join(log_dir, f"{model_load_name}_{step}.th")),
        )

    return model


def save_samples(np_samples, step, log_dir):
    if not os.path.isabs(log_dir):
        log_dir = "logs/" + log_dir
    log_dir = log_dir + "/samples"
    os.makedirs(log_dir, exist_ok=True)
    save_path = log_dir + f'/samples_{step}.npz.npy'
    np.save(save_path, np_samples, allow_pickle=True)
    
    sync_to_home(save_path, log_dir)


def save_stats(H, stats, step):
    if os.path.isabs(H.log_dir):
        log_dir = H.log_dir
    else:
        log_dir = f"logs/{H.log_dir}"
    save_dir = f"{log_dir}/saved_stats"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/stats_{step}"
    log(f"Saving stats to {save_path}")
    torch.save(stats, save_path)
    
    sync_to_home(save_path, log_dir)


def save_noteseqs(ns, prefix='pre_adv'):
    for i, n in enumerate(ns):
        note_sequence_to_midi_file(n, prefix + f'_{i}.mid')


def samples_2_noteseq(np_samples):
    if np_samples.shape[2] == 3:
        converter = TrioConverter(16)#todo: Hparams, async
        return converter.from_tensors(np_samples)
    elif np_samples.shape[2] == 8:
        from utils.octuple import OctupleEncoding
        import tempfile
        import note_seq
        converter = OctupleEncoding()
        note_seqs = []
        for s in np_samples:
            midi_obj = converter.decode(s)
            with tempfile.NamedTemporaryFile(suffix='.mid') as tmp:
                midi_obj.dump(tmp.name)
                ns = note_seq.midi_to_note_sequence(
                    open(tmp.name, 'rb').read())
            note_seqs.append(ns)
        return note_seqs
    else:
        converter = OneHotMelodyConverter()
        np_samples = np_samples[:, :, 0]
        return converter.from_tensors(np_samples)


def vis_samples(vis, samples, step):
    pass  # Visualization removed


def set_up_visdom(H):
    class MockVisdom:
        def __getattribute__(self, name):
            return lambda *args, **kwargs: None
    return MockVisdom()
