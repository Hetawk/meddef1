import os
import torch
import torch.multiprocessing as mp
import logging

def fix_multiprocessing_issues():
    """
    Fix issues related to multiprocessing in PyTorch:
    1. Set multiprocessing start method to 'spawn' instead of 'fork'
    2. Disable threading in OpenMP (used by PyTorch C++ backend)
    
    This addresses "Leaking Caffe2 thread-pool after fork" warnings
    """
    try:
        # Set multiprocessing method to 'spawn' to avoid thread leakage
        if not torch.multiprocessing.get_start_method(allow_none=True):
            torch.multiprocessing.set_start_method('spawn', force=True)
            logging.info("PyTorch multiprocessing method set to 'spawn'")
        
        # Disable OpenMP threading to avoid warnings
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # Additional PyTorch settings for efficiency
        torch.backends.cudnn.benchmark = True
    except Exception as e:
        logging.warning(f"Could not configure multiprocessing settings: {e}")
        
    return True
