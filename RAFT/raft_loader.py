"""
RAFT model loader for DragGAN.
Handles model initialization and weight loading.
"""

import os
import sys
import torch
from argparse import Namespace

# Add core directory to path
core_dir = os.path.join(os.path.dirname(__file__), 'core')
sys.path.insert(0, core_dir)

try:
    from RAFT.core.raft import RAFT
    RAFT_CORE_AVAILABLE = True
except ImportError as e:
    RAFT_CORE_AVAILABLE = False
    print(f"[RAFT Loader] Failed to import RAFT core: {e}")


def load_raft_model():
    """
    Load and return RAFT model class for instantiation.
    
    Returns:
        tuple: (model_class, loaded_successfully)
            - model_class: RAFT class or None
            - loaded_successfully: bool indicating if model was successfully loaded
    """
    
    if not RAFT_CORE_AVAILABLE:
        return None, False
    
    try:
        print("[RAFT Loader] RAFT model class loaded successfully")
        return RAFT, True
        
    except Exception as e:
        print(f"[RAFT Loader] Error loading RAFT: {e}")
        import traceback
        traceback.print_exc()
        return None, False


if __name__ == "__main__":
    # Test loading
    model_cls, success = load_raft_model()
    if success:
        print("✓ RAFT model class loaded successfully")
        print(f"  Model class: {model_cls}")
    else:
        print("✗ Failed to load RAFT model")
