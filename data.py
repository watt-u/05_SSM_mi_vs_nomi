"""Strict PTB-XL loading, patient-disjoint splits, and train-only preprocessing."""
import ast
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset, DataLoader
import config

try:
    import ptbxl_cache
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'datasets' / 'ptbxl_zarr'))
    import ptbxl_cache


def load_metadata():
    if config.ZARR_DIR is not None:
        ptbxl_cache.validate_cache(config.ZARR_DIR, checksums=False)
    df = pd.read_csv(config.DATA_DIR / 'ptbxl_database.csv', index_col='ecg_id', low_memory=False)
    scp = pd.read_csv(config.DATA_DIR / 'scp_statements.csv', index_col=0)
    mi_codes = set(scp.index[(scp.diagnostic == 1) & (scp.diagnostic_class == 'MI')])
    df['mi'] = df.scp_codes.map(lambda s: int(bool(set(ast.literal_eval(s)) & mi_codes)))
    if not df.index.is_unique or df.patient_id.isna().any():
        raise ValueError('Metadata must have unique ECG IDs and nonmissing patient IDs')
    return df


def validate_splits(df, splits):
    groups = {name: df[df.strat_fold.isin(folds)] for name, folds in splits.items()}
    names = list(groups)
    for i, name in enumerate(names):
        if groups[name].empty:
            raise ValueError(f'Empty {name} split')
        for other in names[i + 1:]:
            if set(groups[name].patient_id) & set(groups[other].patient_id):
                raise ValueError(f'Patient overlap between {name} and {other}')
    if sum(len(g) for g in groups.values()) != len(df):
        raise ValueError('Splits must cover each record exactly once')
    return groups


def read_signal(row):
    try:
        signal, meta = wfdb.rdsamp(str(config.DATA_DIR / row.filename_lr))
        if meta['fs'] != 100 or signal.shape != (1000, 12):
            raise ValueError(f'Expected 100 Hz (1000,12), got {meta["fs"]} Hz {signal.shape}')
        names = [s.upper() for s in meta['sig_name']]
        if len(set(names)) != 12 or set(names) != set(config.LEADS):
            raise ValueError(f'Unexpected lead names: {names}')
        signal = signal[:, [names.index(lead) for lead in config.LEADS]].T.astype(np.float32)
        if not np.isfinite(signal).all():
            raise ValueError('Nonfinite signal')
        return signal
    except Exception as exc:
        raise ValueError(f'ecg_id={row.name}: {exc}') from exc


def training_stats(signals):
    mean = signals.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = signals.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.where(std == 0, 1, std).astype(np.float32)
    return dict(mean=mean.tolist(), std=std.tolist())


class ECGDataset(Dataset):
    def __init__(self, signals, df, normalization):
        self.signals = signals
        self.labels = df.mi.to_numpy(dtype=np.float32).reshape(-1, 1)
        self.ecg_ids = df.index.to_numpy(dtype=np.int64)
        self.mean = np.asarray(normalization['mean'], dtype=np.float32)[:, None]
        self.std = np.asarray(normalization['std'], dtype=np.float32)[:, None]
        if self.mean.shape != (12, 1) or self.std.shape != (12, 1) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid saved normalization')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.signals[index] - self.mean) / self.std, self.labels[index], self.ecg_ids[index]


def prepare_data(batch_size=16, smoke=False, checkpoint=None):
    started = time.perf_counter()
    df = load_metadata()
    groups = validate_splits(df, config.SPLITS if checkpoint is None else checkpoint['splits'])
    if checkpoint is not None:
        if bool(checkpoint['smoke_test']) != smoke:
            raise ValueError('Checkpoint smoke/full mode mismatch')
        for name in groups:
            ids = checkpoint['record_ids'][name]
            if not set(ids).issubset(set(groups[name].index)):
                raise ValueError(f'Checkpoint IDs do not match {name} folds')
            groups[name] = groups[name].loc[ids]
    elif smoke:
        # Small deterministic samples retain both classes in every split.
        groups = {name: pd.concat([g[g.mi == c].sample(n=4 if name == 'train' else 2,
                  random_state=42) for c in (0, 1)]).sort_index() for name, g in groups.items()}
    signals = {}
    if config.ZARR_DIR is not None:
        array, positions = ptbxl_cache.open_cache(config.ZARR_DIR)
        # Full runs decode once in storage order; smoke/evaluation subsets read only requested rows.
        full = array[:] if sum(len(g) for g in groups.values()) == len(df) else None
        for name, group in groups.items():
            rows = np.asarray([positions[int(ecg_id)] for ecg_id in group.index])
            signals[name] = full[rows] if full is not None else array.oindex[rows, :, :]
            if not np.isfinite(signals[name]).all():
                raise ValueError(f'Nonfinite cached signals in {name}')
        del full
    else:
        for name, group in groups.items():
            print(f'Loading {name}: {len(group)} records', flush=True)
            signals[name] = np.stack([read_signal(row) for _, row in group.iterrows()])
    loading_seconds = time.perf_counter() - started
    print(f'Data loading: {loading_seconds:.3f}s ({"zarr" if config.ZARR_DIR is not None else "wfdb"})', flush=True)
    norm = training_stats(signals['train']) if checkpoint is None else checkpoint['normalization']
    labels = groups['train'].mi.to_numpy()
    if np.unique(labels).size != 2:
        raise ValueError('Training requires both classes')
    pos_weight = float(np.clip((len(labels) - labels.sum()) / labels.sum(), 1, 20))
    generator = torch.Generator().manual_seed(config.TRAINING['seed'])
    loaders = {name: DataLoader(ECGDataset(signals[name], g, norm), batch_size=batch_size,
               shuffle=name == 'train', num_workers=0, generator=generator)
               for name, g in groups.items()}
    metadata = dict(normalization=norm, pos_weight=pos_weight,
                    record_ids={name: g.index.tolist() for name, g in groups.items()},
                    splits=config.SPLITS if checkpoint is None else checkpoint['splits'],
                    label_definition=config.LABEL_DEFINITION, leads=config.LEADS,
                    sampling_rate=100, signal_length=1000, smoke_test=smoke,
                    data_loading_seconds=loading_seconds,
                    data_backend='zarr' if config.ZARR_DIR is not None else 'wfdb')
    return loaders, metadata
