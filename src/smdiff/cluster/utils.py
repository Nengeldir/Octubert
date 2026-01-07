"""
Cluster utilities for ETH HPC environment.

Handles cluster detection, scratch space management, and syncing between
scratch (100GB, purged) and home directories (20GB, permanent).
"""
import os
import platform
import getpass
import shutil
import re


def is_cluster():
    """
    Detect if running on the ETH cluster.
    
    Heuristic: Checks if the OS is Ubuntu (cluster runs Ubuntu).
    
    Returns:
        bool: True if running on cluster, False otherwise
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


def get_current_username():
    """
    Get the current username.
    
    Returns:
        str: Username of the current user
    """
    try:
        return getpass.getuser()
    except Exception:
        # Fallback to environment variables
        return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'


def get_scratch_dir(username=None):
    """
    Get the scratch directory path for the current user.
    
    ETH cluster scratch space: /work/scratch/{username}
    - 100GB storage
    - Purged regularly
    - Use for temporary outputs during training
    
    Args:
        username: Optional username override. If None, uses current user.
        
    Returns:
        str: Path to user's scratch directory
    """
    if username is None:
        username = get_current_username()
    
    return f"/work/scratch/{username}"


def get_home_logs_dir():
    """
    Get the permanent run directory in the project (project_root/runs).
    All synced checkpoints, samples, and logs live under runs/{model_id}/...
    """
    # Navigate from cluster/utils.py -> cluster -> smdiff -> src -> project_root
    module_dir = os.path.dirname(os.path.abspath(__file__))  # cluster/
    smdiff_dir = os.path.dirname(module_dir)  # smdiff/
    src_dir = os.path.dirname(smdiff_dir)  # src/
    project_root = os.path.dirname(src_dir)  # project_root/

    return os.path.join(project_root, "runs")


def sync_to_home(scratch_path, home_base_dir=None):
    """
    Sync a file from scratch space to permanent home storage.
    
    Deletes older versions of the same file type in the destination to
    keep only the latest checkpoint/sample/metric.
    
    Args:
        scratch_path: Full path to file in scratch space
        home_base_dir: Base directory in home storage (default: project_root/logs)
    
    Example:
        scratch_path = "/work/scratch/user/log_name/checkpoints/model_500.th"
        -> syncs to -> "project_root/runs/log_name/checkpoints/model_500.th"
    """
    if home_base_dir is None:
        home_base_dir = get_home_logs_dir()
    
    # Get the scratch base directory for current user
    scratch_base = get_scratch_dir()
    
    # Only sync if file is actually in scratch space
    if not scratch_path.startswith(scratch_base):
        return
    
    # Calculate relative path from scratch base
    # e.g., /work/scratch/user/log_name/checkpoints/model_500.th
    #    -> log_name/checkpoints/model_500.th
    rel_path = os.path.relpath(scratch_path, scratch_base)
    
    # Destination is home_base_dir/rel_path
    dest_path = os.path.join(home_base_dir, rel_path)
    dest_dir = os.path.dirname(dest_path)
    
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        # Copy the file
        shutil.copy2(scratch_path, dest_path)
        print(f"Synced {scratch_path} -> {dest_path}")
        
        # Cleanup older files of the same type
        _cleanup_old_versions(dest_path, dest_dir)
        
    except Exception as e:
        print(f"Failed to sync {scratch_path} to home: {e}")


def _cleanup_old_versions(current_file, dest_dir):
    """
    Delete older versions of checkpoint/sample/metric files.
    
    Keeps only the latest version based on step number in filename.
    
    Args:
        current_file: Path to the file just synced
        dest_dir: Directory containing the file
    """
    filename = os.path.basename(current_file)
    
    # Determine file pattern based on naming convention
    # Expected patterns: "model_X.th", "ema_X.th", "optim_X.th", 
    #                   "samples_X.npz", "stats_X.pt"
    
    pattern = None
    current_step = None
    
    if filename.endswith(".th"):
        # Checkpoint files: model_500.th, ema_500.th, optim_500.th
        match = re.match(r"(.+?)_(\d+)\.th$", filename)
        if match:
            prefix = match.group(1)
            current_step = int(match.group(2))
            pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.th$")
    
    elif filename.endswith(".npz") or filename.endswith(".npz.npy"):
        # Sample files: samples_500.npz or samples_500.npz.npy
        match = re.match(r"samples_(\d+)\.npz(?:\.npy)?$", filename)
        if match:
            current_step = int(match.group(1))
            pattern = re.compile(r"samples_(\d+)\.npz(?:\.npy)?$")
    
    elif filename.endswith(".pt"):
        # Metric files: stats_500.pt
        match = re.match(r"stats_(\d+)\.pt$", filename)
        if match:
            current_step = int(match.group(1))
            pattern = re.compile(r"stats_(\d+)\.pt$")
    
    if pattern is None or current_step is None:
        return  # Unknown pattern, skip cleanup
    
    # Find and delete older versions
    try:
        for f in os.listdir(dest_dir):
            if f == filename:
                continue  # Skip the current file
            
            match = pattern.match(f)
            if match:
                step = int(match.group(1))
                if step < current_step:
                    old_file = os.path.join(dest_dir, f)
                    try:
                        os.remove(old_file)
                        print(f"Removed older file: {old_file}")
                    except OSError as e:
                        print(f"Error removing {old_file}: {e}")
    except Exception as e:
        print(f"Error during cleanup in {dest_dir}: {e}")
