"""Training and inference utilities, with best-checkpoint restoration."""
import json
import random
import time
import resource
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from model import ECGMamba


def get_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this Python process')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable in this Python process')
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Small tensor scans suffer severe overhead with many CPU threads.
    torch.set_num_threads(4)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    if device.type == 'mps':
        torch.mps.synchronize()


def memory_metrics(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'peak_process_rss_mb': rss / (1024 ** 2 if sys.platform == 'darwin' else 1024)}
    if device.type == 'mps':
        result['mps_allocated_mb'] = torch.mps.current_allocated_memory() / 1024 ** 2
    if device.type == 'cuda':
        result['cuda_allocated_mb'] = torch.cuda.memory_allocated(device) / 1024 ** 2
        result['cuda_peak_allocated_mb'] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return result


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    model.eval()
    ys, ps, ids, loss_sum = [], [], [], 0.0
    for x, y, ecg_ids in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite inference logits')
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * len(x)
        ys.append(y.cpu().numpy().ravel())
        ps.append(logits.sigmoid().cpu().numpy().ravel())
        ids.extend(ecg_ids.tolist())
    return np.concatenate(ys), np.concatenate(ps), ids, loss_sum / len(loader.dataset)


def save_checkpoint(checkpoint, path):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ECGMamba(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def train_model(loaders, metadata, model_config, settings, output_dir, device):
    model = ECGMamba(**model_config).to(device)
    print(f'Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([metadata['pos_weight']], device=device))
    # Mamba continuous-time dynamics and skip terms are exempt from weight decay.
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith(('A_log', '.D')) else decay).append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay}, {'params': no_decay, 'weight_decay': 0}],
                                  lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    history = []
    best, stale = -float('inf'), 0
    started = time.perf_counter()
    batch_times = []
    path = output_dir / 'best_model.pt'
    for epoch in range(1, settings['epochs'] + 1):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.0
        for batch_index, (x, y, _) in enumerate(loaders['train']):
            synchronize(device)
            batch_start = time.perf_counter()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(x)
            synchronize(device)
            batch_times.append(time.perf_counter() - batch_start)
            if batch_index == 0 or (batch_index + 1) % 100 == 0:
                progress = dict(status='training', epoch=epoch, batch=batch_index + 1, batches_per_epoch=len(loaders['train']), last_batch_seconds=batch_times[-1], elapsed_seconds=time.perf_counter() - started, **memory_metrics(device))
                (output_dir / 'progress.json').write_text(json.dumps(progress, indent=2))
                print(f'Epoch {epoch} batch {batch_index + 1}/{len(loaders["train"])} loss={loss.item():.4f} seconds={batch_times[-1]:.2f}', flush=True)
        y, p, _, val_loss = predict(model, loaders['val'], device, criterion)
        val_auc = float(roc_auc_score(y, p))
        scheduler.step(val_auc)
        record = dict(epoch=epoch, train_loss=total / len(loaders['train'].dataset),
                      val_loss=val_loss, val_auroc=val_auc, seconds=time.perf_counter() - epoch_start,
                      lr=optimizer.param_groups[0]['lr'])
        history.append(record)
        (output_dir / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        if val_auc > best:
            best, stale = val_auc, 0
            checkpoint = dict(metadata, model_config=model_config, training_config=settings,
                              model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                              epoch=epoch, val_auroc=val_auc, threshold=None)
            save_checkpoint(checkpoint, path)
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    timing = dict(training_seconds=time.perf_counter() - started,
                  mean_training_batch_seconds=float(np.mean(batch_times)),
                  median_training_batch_seconds=float(np.median(batch_times)),
                  epochs_completed=len(history), device=str(device), **memory_metrics(device))
    (output_dir / 'timing.json').write_text(json.dumps(timing, indent=2))
    model, checkpoint = load_model(path, device)
    return model, checkpoint
