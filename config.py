"""Defaults for the portable PTB-XL Mamba experiment (no import side effects)."""
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
# Evaluate the repository-relative default only for local runs. Modal mounts
# this module at /experiment/config.py, where parents[1] does not exist.
ZARR_DIR = Path(os.environ['PTBXL_ZARR_DIR']) if 'PTBXL_ZARR_DIR' in os.environ else None
DATA_DIR = (ZARR_DIR if ZARR_DIR is not None else Path(os.environ['PTBXL_DATA_DIR']) if 'PTBXL_DATA_DIR' in os.environ
            else SCRIPT_DIR.parents[1] / 'datasets' / 'ptb-xl')
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
MODEL = dict(d_model=32, d_state=16, expand=2, d_conv=4, n_blocks=2,
             dropout=0.2, mix_dim=64)
TRAINING = dict(seed=42, batch_size=16, lr=1e-3, weight_decay=1e-4,
                epochs=50, patience=10, clip_grad_norm=1.0)
SPLITS = dict(train=list(range(1, 9)), val=[9], test=[10])
LABEL_DEFINITION = 'MI if any diagnostic SCP code maps to diagnostic_class MI, regardless of likelihood or co-occurring codes; otherwise non-MI.'
