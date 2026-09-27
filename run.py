"""Run from this directory with the project's shared Python interpreter."""
import argparse
import json
import platform
import os
import time
from pathlib import Path
import torch
import config
from data import prepare_data
from train import get_device, seed_everything, train_model, load_model, memory_metrics
from evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(description='Portable bidirectional Mamba: PTB-XL MI vs non-MI')
    parser.add_argument('--epochs', type=int, default=config.TRAINING['epochs'])
    parser.add_argument('--batch-size', type=int, default=config.TRAINING['batch_size'])
    parser.add_argument('--lr', type=float, default=config.TRAINING['lr'])
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    parser.add_argument('--output-dir', type=Path, default=config.OUTPUT_DIR)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < float('inf'):
        parser.error('epochs, batch-size, and lr must be positive and finite')
    output_dir = args.output_dir.resolve() / 'smoke' if args.smoke_test else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_training and (output_dir / 'best_model.pt').exists():
        parser.error('Checkpoint already exists; use --skip-training or a new --output-dir')
    device = get_device(args.device)
    settings = dict(config.TRAINING, epochs=1 if args.smoke_test else args.epochs,
                    batch_size=args.batch_size, lr=args.lr)
    seed_everything(settings['seed'])
    started = time.perf_counter()
    checkpoint = None
    if args.skip_training:
        model, checkpoint = load_model(output_dir / 'best_model.pt', device)
        if checkpoint['leads'] != config.LEADS or checkpoint['sampling_rate'] != 100 or checkpoint['signal_length'] != 1000 or checkpoint['label_definition'] != config.LABEL_DEFINITION:
            raise ValueError('Checkpoint data contract is incompatible with this loader')
    loaders, metadata = prepare_data(args.batch_size, args.smoke_test, checkpoint)
    if not args.skip_training:
        snapshot = dict(metadata, model_config=config.MODEL, training_config=settings,
                        device=str(device), torch_version=str(torch.__version__), python_version=platform.python_version())
        (output_dir / 'config.json').write_text(json.dumps(snapshot, indent=2))
        model, checkpoint = train_model(loaders, metadata, config.MODEL, settings, output_dir, device)
    metrics = run_evaluation(model, loaders, checkpoint, output_dir, device)
    timing_path = output_dir / ('evaluation_timing.json' if args.skip_training else 'timing.json')
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing.update(total_run_seconds=time.perf_counter() - started,
                  data_loading_seconds=metadata['data_loading_seconds'],
                  staging_seconds=float(os.environ.get('PTBXL_STAGING_SECONDS', '0')),
                  **memory_metrics(device))
    timing_path.write_text(json.dumps(timing, indent=2))
    (output_dir / 'progress.json').write_text(json.dumps(dict(status='completed', selected_epoch=checkpoint['epoch'], total_run_seconds=time.perf_counter() - started), indent=2))
    report = ["# PTB-XL MI vs non-MI results", "",
              "Smoke test only; not a generalization estimate." if args.smoke_test else "Official held-out fold 10 evaluation; threshold selected on fold 9.", "",
              f"Device: {device}; selected epoch: {checkpoint['epoch']}.", "",
              "| Metric | Value |", "|---|---:|"]
    report += [f"| {key} | {metrics[key]:.6f} |" for key in
               ('auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'ppv', 'npv', 'threshold')]
    report += ["", f"This invocation took {timing['total_run_seconds']:.1f} seconds.",
               "See metrics.json for threshold-0.5 metrics and timing.json for training timing.",
               "No comparison with the CNN baseline has been established."]
    (output_dir / 'results.md').write_text("\n".join(report) + "\n")
    print(f'Artifacts: {output_dir}', flush=True)


if __name__ == '__main__':
    main()
