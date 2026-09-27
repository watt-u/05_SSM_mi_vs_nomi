"""Validate on MPS, then launch a persistent full run with macOS sleep prevention.

Run using the shared interpreter from a normal Terminal or approved unsandboxed
execution. Repeated launches are refused while an earlier process is alive.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import numpy as np
import pandas as pd
import config


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError('This launcher requires MPS access outside the sandbox')
    output = config.OUTPUT_DIR
    output.mkdir(exist_ok=True)
    launch_path = output / 'launch.json'
    if launch_path.exists():
        previous = json.loads(launch_path.read_text())
        try:
            os.kill(previous['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f'Previous run may still be active: PID {previous["pid"]}')
    if (output / 'best_model.pt').exists():
        raise RuntimeError('Full checkpoint already exists; choose a new run directory manually')
    subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_experiment'],
                   cwd=config.SCRIPT_DIR, check=True)
    smoke_root = output / 'launch_validation'
    smoke_output = smoke_root / 'smoke'
    if not (smoke_output / 'best_model.pt').exists():
        subprocess.run([sys.executable, '-u', 'run.py', '--smoke-test',
                        '--device', 'mps', '--output-dir', str(smoke_root)],
                       cwd=config.SCRIPT_DIR, check=True)
    before = pd.read_csv(smoke_output / 'predictions.csv')
    subprocess.run([sys.executable, '-u', 'run.py', '--smoke-test', '--skip-training',
                    '--device', 'mps', '--output-dir', str(smoke_root)],
                   cwd=config.SCRIPT_DIR, check=True)
    after = pd.read_csv(smoke_output / 'predictions.csv')
    np.testing.assert_array_equal(before.ecg_id, after.ecg_id)
    np.testing.assert_array_equal(before.true_mi, after.true_mi)
    np.testing.assert_allclose(before.pred_proba_mi, after.pred_proba_mi, atol=1e-6, rtol=1e-6)
    log_path = output / 'training.log'
    with log_path.open('a') as log:
        process = subprocess.Popen(['/usr/bin/caffeinate', '-dims', sys.executable, '-u',
                                    'run.py', '--device', 'mps'], cwd=config.SCRIPT_DIR,
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    details = dict(pid=process.pid, started_unix=time.time(), log=str(log_path), device='mps')
    launch_path.write_text(json.dumps(details, indent=2))
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError(f'Training exited during startup; inspect {log_path}')
    print(json.dumps(details, indent=2), flush=True)


if __name__ == '__main__':
    main()
