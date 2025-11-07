import torch
import numpy as np
import random
import os

def set_deterministic_seed(seed: int):
    """Sets random seeds for reproducibility across various libraries."""
    if seed is None: return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        # Ensure deterministic operations on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For Numba/CUDA, environment variables might be needed
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

    # For Numba, setting environment variable before import is ideal
    # If Numba is already imported, this might not take full effect.
    os.environ['NUMBA_RANDOM_SEED'] = str(seed)

    # For NetworkX, its random state is usually tied to numpy's if not explicitly set.

    # For Z3, it doesn't have a global random seed, but its behavior is deterministic
    # for a given set of constraints and solver configuration.

    print(f"[INFO] Global random seed set to {seed} for reproducibility.")
