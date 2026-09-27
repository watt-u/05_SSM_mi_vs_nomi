# Validation — 2026-09-13

Implementation and smoke validation are complete. Full training and full held-out
evaluation have **not** run: the request to launch long MPS training outside the
sandbox was declined. No full-training process was started.

## Checks performed

- Seven CPU unit tests passed. The eighth, an explicit MPS/CPU scan parity test,
  was skipped in the sandbox because MPS is unavailable there. The unsandboxed
  final test-suite invocation was included in the declined launch and did not run.
- Scan outputs and all six input gradients matched an independent literal
  float64 recurrence over 137 time steps, crossing two chunk boundaries.
- Model shape, finite gradients, reverse alignment, lead isolation, and exact
  model checkpoint reload tests passed.
- MI labels matched experiment 02's definition. Official split sizes were
  17,418 / 2,183 / 2,198, with no patient overlap.
- Training-only normalization and invalid-signal/lead-reordering checks passed.
- All 21,799 local recordings passed strict shape, sampling-rate, lead-name and
  finite-value validation in 79.0 seconds. Data files were only read.
- Evaluation-only mode reproduced saved CPU smoke predictions within 1e-6 after
  the scan optimization. Actual saved preprocessing and threshold were reused.
- All Python modules passed compilation. Generated training curves were visually
  inspected; labels and loss/AUROC histories render correctly.

## Real-data smoke runs

All runs used the same deterministic sample: 32 training, 16 validation, and 16
test records, balanced within each split, for two epochs. These tiny-sample
results validate the pipeline; they are not full-test results.

| Run | Mean training batch | Training time | Test AUROC | Test AUPRC |
|---|---:|---:|---:|---:|
| Initial CPU scan | 9.205 s | 41.781 s | 0.71875 | 0.79934 |
| Optimized MPS scan | 2.786 s | 13.248 s | 0.71875 | 0.79934 |

The optimized MPS run used 71,553 parameters, selected epoch 2 with validation
AUROC 0.78125, and selected threshold 0.45. Test F1 was 0.66667, sensitivity
0.875 and specificity 0.25. CPU/MPS training can produce different predictions;
identical cross-device training trajectories are not assumed.

Optimized MPS peak process RSS was 653.97 MiB; post-training MPS allocated memory
was 9.34 MiB. The latter is a snapshot after transient training tensors were
released, **not peak GPU memory**. Initial CPU process peak RSS was 9,196.59 MiB;
that measurement preceded the memory-efficient analytic backward optimization.

Artifacts:

- `outputs/smoke/`: CPU checkpoint, metrics, plots and evaluation-only results.
- `outputs/mps_validation/smoke/`: initial GPU smoke test.
- `outputs/mps_optimized/smoke/`: final scan GPU smoke test and timing.

At approximately 2.8 seconds per batch, 1,089 batches per training epoch imply
about 51 minutes of training per epoch, plus validation. The maximum 50-epoch run
is estimated at roughly 40–45 hours, subject to sustained device performance;
early stopping can shorten it. This is an extrapolation, not a measured full-run
duration. No CNN comparison has been established.

## Remaining execution

When ready, run `launch_local.py` with the shared interpreter outside the
sandbox. It performs the GPU numerical checks and an evaluation-only smoke
check before launching the full official-split training under `caffeinate`.
The runner automatically restores the best checkpoint, selects the validation
threshold, and writes held-out metrics and `outputs/results.md` on completion.


## Zarr cache validation — 2026-09-13

- Converted all 21,799 local 100 Hz ECGs and verified relevant source SHA256s.
- Every saved float32 waveform exactly matched a second WFDB read.
- Cache: 22 waveform shards, 28 payload files, 880,720,045 payload bytes; manifests add two files.
- Uploaded to `ptbxl-100hz-zarr-v2/1.0.3-100hz-v1`; CPU verifier checked every uploaded artifact before publishing completion.
- Local suite: 18 tests, 16 passed, CUDA/MPS checks skipped. Cache/source normalization, reordered checkpoint IDs, and prediction parity passed.
- Modal CUDA suite: 18 tests, 16 passed; MPS and local source/cache parity skipped. CUDA numerical parity passed.
- Modal smoke: one epoch, best-checkpoint reload, and evaluation passed with cached data only. Full cache staging plus verification took 8.721 seconds; loading the 8/4/4 smoke subset took 1.766 seconds. These are not full-dataset loading timings.
- The remote CPU comparison benchmark was not run because its launch was declined. No raw-versus-cache speedup is claimed; `benchmark_modal.py` is available for a later authorized run.
- No full training was launched. Detailed receipts are in `datasets/ptbxl_zarr/validation.json` and `upload.json`.
