# 05_SSM_mi_vs_nomi

Binary MI versus non-MI classification on PTB-XL with a portable bidirectional
Mamba-1 model. One output logit becomes P(MI) via sigmoid. MI means any diagnostic
SCP code mapped to the MI superclass, regardless of likelihood or co-occurring
labels, matching experiment 02.

## Run

Run from this directory with the project interpreter:

```bash
cd /Users/bmer/Documents/1_Projects/ECG_STEMI/experiments/05_SSM_mi_vs_nomi
/Users/bmer/Documents/1_Projects/.general_venv/bin/python -u run.py --smoke-test
/Users/bmer/Documents/1_Projects/.general_venv/bin/python -u run.py
```

Options: `--epochs 50`, `--batch-size 16`, `--lr 1e-3`,
`--device auto|cuda|mps|cpu`, `--output-dir PATH`, `--skip-training`, `--smoke-test`.
The smoke test uses 8 training, 4 validation and 4 test records, retains both
classes, runs one epoch without generating plots, and always appends `smoke/` to the output
path. Smoke metrics are debugging results, not estimates of generalization.

```bash
# Evaluate using saved architecture, preprocessing, record IDs and threshold.
/Users/bmer/Documents/1_Projects/.general_venv/bin/python -u run.py --skip-training
# Verify numerical and data contracts.
/Users/bmer/Documents/1_Projects/.general_venv/bin/python -m unittest -v test_experiment
```

A new training run refuses to overwrite an existing best checkpoint. Use a new
output directory for another run. Evaluation-only mode preserves the original
training configuration and timing and writes `evaluation_timing.json`.

For a persistent Mac GPU run, `launch_local.py` first runs the numerical tests,
creates/reuses a separate smoke checkpoint and verifies evaluation-only results,
then starts `run.py --device mps` under `caffeinate -dims`. It writes the process
ID to `outputs/launch.json` and logs to `outputs/training.log`. Run this only when
ready for a potentially long run; it keeps the Mac awake until the process exits.

```bash
/Users/bmer/Documents/1_Projects/.general_venv/bin/python launch_local.py
```

## Run on Modal

Build the local 100 Hz Zarr cache and upload it with
[`datasets/ptbxl_zarr/pipeline.py`](../../datasets/ptbxl_zarr/pipeline.py).
See the [cache preparation instructions](../../datasets/ptbxl_zarr/README.md).
The CPU upload verifier must finish before starting training.

The experiment's `modal_app.py` only handles validation, training, and evaluation:

```bash
cd /Users/bmer/Documents/1_Projects/ECG_STEMI/experiments/05_SSM_mi_vs_nomi
# CUDA tests and one-epoch smoke training, then full training.
/Users/bmer/Documents/1_Projects/.general_venv/bin/python -m modal run --detach modal_app.py
# Only run CUDA validation and smoke training.
/Users/bmer/Documents/1_Projects/.general_venv/bin/python -m modal run --detach modal_app.py --stage smoke
```

The default GPU is an A10, with 4 CPUs, 16 GiB of host memory and a 24-hour function
timeout. Full training keeps the default batch size 16 and maximum 50 epochs;
override with `--batch-size` and `--epochs`. No automatic retry or interrupted-run
resume is configured. The detached job continues if the local client disconnects.

Data is mounted read-only from `ptbxl-100hz-zarr-v2` at
`/data/1.0.3-100hz-v1`, then staged to `/tmp/ptbxl_zarr/1.0.3-100hz-v1`
with four concurrent copy workers and SHA256 verification. `PTBXL_ZARR_DIR`
selects that local cache. Full runs bulk-load signals into RAM; smoke runs load
only selected records. Staging and data-loading durations are recorded separately.
The launcher requires a verified completion marker and never creates the dataset
volume. Local runs retain WFDB loading by default; set `PTBXL_ZARR_DIR` to an
absolute cache directory to use Zarr locally. A selected invalid cache fails
explicitly. ECG IDs preserve metadata and checkpoint ordering.

Run `python -m modal run benchmark_modal.py` with the project interpreter to
compare both paths in fresh CPU containers; the report is saved in
`datasets/ptbxl_zarr/benchmark.json`.

Results go to the shared **`experiment-results`** volume, automatically created
with `version=2`, under `05_SSM_mi_vs_nomi/RUN_ID/`. Other experiments can use
the same volume with their own experiment folder. Each run ID is unique
(printed at launch, or supplied with `--run-id`). `full/` holds the full experiment outputs. `training.log` and `launch.json`
record execution and status. Modal smoke checkpoints, metrics, and validation
logs use a temporary container directory that is removed after validation; they
are never saved to the results volume. Validation logs remain visible in Modal
app logs. `--stage smoke` writes no run artifacts to the volume. The smoke run
reloads its best checkpoint before evaluation; the redundant second evaluation
subprocess has been removed. Reusing a run ID
within this experiment is refused to protect existing outputs. Download a completed run with:

```bash
/Users/bmer/Documents/1_Projects/.general_venv/bin/python -m modal volume get experiment-results /05_SSM_mi_vs_nomi/RUN_ID ./outputs/modal/RUN_ID
```

## Device and dependencies

Dependencies: torch, numpy, pandas, wfdb, scikit-learn, matplotlib, seaborn;
cached loading also requires zarr==3.3.0 and numcodecs==0.16.5.
No `mamba_ssm`, Triton, CUDA extensions, or additional dependency installs are
required. The shared environment already provides these dependencies.

`auto` selects CUDA when available, then MPS, otherwise CPU. Explicit
`--device cuda` fails clearly if CUDA is unavailable. This machine has an
M2 Pro GPU: MPS is available outside the Codex sandbox, but appears unavailable
inside it. Explicit `--device mps` fails clearly if unavailable. Use an approved
unsandboxed run or a normal Terminal session for GPU training. CPU runs are valid
but substantially slower. Computation is float32; the numerical reference test
also checks CPU float64. First-order training gradients are supported; the
memory-efficient scan does not support higher-order differentiation.

## Data and training

- Read-only 100 Hz, ten-second, 12-lead recordings under `datasets/ptb-xl`.
- Official folds 1–8 / 9 / 10: 17,418 / 2,183 / 2,198 ECGs. Patient sets must be disjoint.
- Strict sample-rate, shape, finite-value and lead checks. Leads are reordered to
  I, II, III, aVR, aVL, aVF, V1–V6 when names are valid but differently ordered.
  Invalid recordings raise errors identifying the ECG, never become zero signals.
- Per-lead mean/std and positive weighting come only from training records.
  Positive weight is `clip(N_negative/N_positive, 1, 20)`, matching experiment 02.
- AdamW, learning rate 0.001, weight decay 0.0001, gradient clipping 1.0,
  batch size 16, seed 42, 50 epochs maximum. Mamba A/D parameters have no decay.
- Validation-AUROC scheduler (factor 0.5, patience 5) and early stopping (patience
  10). The best validation checkpoint is restored before final evaluation.
- Validation F1 chooses the threshold over 0.05 through 0.90, step 0.05, with
  first-best tie handling matching experiment 02. Test data never selects it.

## Artifacts

`outputs/` contains configuration including record IDs, `best_model.pt`,
`history.json`, `timing.json`, `progress.json`, `metrics.json`, `predictions.csv`,
training curves, ROC/PR curves, and a confusion matrix. The checkpoint stores
architecture, preprocessing, split IDs, lead order, label definition, and the
selected threshold. Checkpoints are written with atomic replacement.

Metrics include AUROC, AUPRC, F1, sensitivity, specificity, PPV/NPV, the confusion
matrix, and a second set of metrics at threshold 0.5. Predictions include ECG ID,
true MI label and probability. Timing reports synchronized batch durations,
process peak RSS, MPS allocated tensor memory, and CUDA current/peak tensor
allocation when applicable. RSS and MPS allocation
are distinct measurements and should not be added together on unified memory.

`progress.json` updates at the first and every hundredth training batch, then
reports completion after held-out evaluation. `history.json` updates each epoch.
Inspect the process log for failures if progress stops updating.

See `architecture.md` for tensor flow and scan equations, and `VALIDATION.md`
for measured verification results. No claim of improvement over CNN is implied.
