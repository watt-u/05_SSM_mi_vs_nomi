"""CUDA training on the prepared PTB-XL Modal Volume v2.

Run from this directory: python -m modal run --detach modal_app.py
First build and upload datasets/ptbxl_zarr/pipeline.py.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
import uuid
import tempfile
import os

import modal

APP_NAME = 'ptbxl-ssm-mi-vs-nomi'
DATA_VOLUME = 'ptbxl-100hz-zarr-v2'
RESULTS_VOLUME = 'experiment-results'
EXPERIMENT_NAME = '05_SSM_mi_vs_nomi'
DATA_ROOT = '/data/1.0.3-100hz-v1'
STAGE_ROOT = '/tmp/ptbxl_zarr/1.0.3-100hz-v1'
CODE_ROOT = '/experiment'
CACHE_MODULE = (Path(__file__).resolve().parents[2] / 'datasets/ptbxl_zarr/ptbxl_cache.py'
                if modal.is_local() else Path(CODE_ROOT) / 'ptbxl_cache.py')

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME, version=2)
results_volume = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True, version=2)
train_image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.8.0', 'numpy==2.2.6', 'pandas==2.3.3', 'wfdb==4.3.1',
                 'scikit-learn==1.7.2', 'matplotlib==3.10.8', 'seaborn==0.13.2',
                 'zarr==3.3.0', 'numcodecs==0.16.5')
    .env({'PTBXL_ZARR_DIR': STAGE_ROOT, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'})
    .add_local_file(CACHE_MODULE, CODE_ROOT + '/ptbxl_cache.py')
    .add_local_dir(Path(__file__).parent, CODE_ROOT,
                   ignore=lambda path: path.suffix != '.py')
)


def run_logged(command, log_path):
    with log_path.open('a', buffering=1) as log:
        process = subprocess.Popen(command, cwd=CODE_ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


@app.function(image=train_image,
              volumes={'/data': data_volume.with_mount_options(read_only=True),
                       '/results': results_volume},
              gpu='A10', cpu=4, memory=16384, timeout=24 * 3600)
def train_gpu(run_id: str, epochs: int = 50, batch_size: int = 16, smoke_only: bool = False):
    import torch

    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', run_id):
        raise ValueError('Invalid run ID')
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Modal GPU function requires CUDA')
    root = Path('/results') / EXPERIMENT_NAME / run_id
    if not smoke_only and root.exists():
        raise FileExistsError(f'Run already exists: {root}')
    sys.path.insert(0, CODE_ROOT)
    from ptbxl_cache import stage_cache, sha256
    staging = stage_cache(DATA_ROOT, STAGE_ROOT)
    os.environ['PTBXL_STAGING_SECONDS'] = str(staging['staging_seconds'])
    marker = dict(version=Path(DATA_ROOT).name, manifest_sha256=sha256(Path(STAGE_ROOT) / 'manifest.json'))
    print(f'Cache staging: {json.dumps(staging)}', flush=True)
    # All validation output is ephemeral, including logs and checkpoints.
    # train_model already reloads the best checkpoint before evaluation.
    with tempfile.TemporaryDirectory(prefix='ptbxl-smoke-') as temporary:
        smoke_root = Path(temporary)
        run_logged([sys.executable, '-m', 'unittest', '-v', 'test_experiment', 'test_cache'], smoke_root / 'validation.log')
        run_logged([sys.executable, '-u', 'run.py', '--device', 'cuda', '--smoke-test',
                    '--batch-size', str(min(batch_size, 8)), '--output-dir', str(smoke_root)],
                   smoke_root / 'validation.log')
    print('Validation passed; temporary smoke artifacts removed.', flush=True)
    if smoke_only:
        return dict(experiment=EXPERIMENT_NAME, run_id=run_id, status='completed', smoke_only=True, **staging)

    root.mkdir(parents=True, exist_ok=False)
    status = dict(experiment=EXPERIMENT_NAME, run_id=run_id, status='training', gpu=torch.cuda.get_device_name(),
                  epochs=epochs, batch_size=batch_size, smoke_only=False, validation='passed', data=marker,
                  staging=staging)
    (root / 'launch.json').write_text(json.dumps(status, indent=2))
    try:
        run_logged([sys.executable, '-u', 'run.py', '--device', 'cuda', '--epochs', str(epochs),
                    '--batch-size', str(batch_size), '--output-dir', str(root / 'full')], root / 'training.log')
        status['status'] = 'completed'
        return dict(status, output_path=str(root))
    except BaseException as exc:
        status.update(status='failed', error=str(exc))
        raise
    finally:
        (root / 'launch.json').write_text(json.dumps(status, indent=2))
        results_volume.commit()


@app.local_entrypoint()
def main(stage: str = 'train', run_id: str = '', epochs: int = 50, batch_size: int = 16):
    if stage not in {'smoke', 'train'}:
        raise ValueError('stage must be smoke or train; prepare data using datasets/ptbxl_zarr/pipeline.py')
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    print(f'Run ID: {run_id}; data: {DATA_VOLUME}; results: {RESULTS_VOLUME}/{EXPERIMENT_NAME}/{run_id}', flush=True)
    result = train_gpu.remote(run_id, epochs, batch_size, smoke_only=stage == 'smoke')
    print(json.dumps(result, indent=2))
