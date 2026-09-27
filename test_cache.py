"""Cache integrity, atomic staging, identity mapping, and real-data parity tests."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import zarr

import config
import data
from data import ptbxl_cache as cache


class CacheContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        self.root.mkdir()
        self.contract = dict(cache.CONTRACT, shape=[3, 12, 1000])
        self.patcher = patch.dict(cache.CONTRACT, self.contract)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        group = zarr.open_group(str(self.root / 'signals.zarr'), mode='w')
        group.attrs.update(self.contract)
        signals = group.create_array('signals', shape=(3, 12, 1000), dtype='float32',
                                     chunks=(64, 12, 1000), shards=(1024, 12, 1000))
        signals[:] = np.arange(3, dtype=np.float32)[:, None, None]
        group.create_array('ecg_ids', data=np.array([30, 10, 20], dtype=np.int64))
        pd.DataFrame({'ecg_id': [10, 20, 30]}).to_csv(self.root / 'ptbxl_database.csv', index=False)
        (self.root / 'scp_statements.csv').write_text('code,diagnostic_class\nMI,MI\n')
        self.publish()

    def publish(self):
        artifacts = {str(p.relative_to(self.root)): dict(bytes=p.stat().st_size, sha256=cache.sha256(p))
                     for p in self.root.rglob('*') if p.is_file() and p.name not in {'manifest.json', '.complete.json'}}
        cache.write_json(self.root / 'manifest.json', dict(self.contract, artifacts=artifacts))
        cache.write_json(self.root / '.complete.json', dict(manifest_sha256=cache.sha256(self.root / 'manifest.json')))

    def test_mapping_does_not_assume_metadata_or_id_order(self):
        signals, positions = cache.open_cache(self.root)
        np.testing.assert_array_equal(signals.oindex[[positions[x] for x in [20, 30, 10]], 0, 0], [2, 0, 1])

    def test_missing_and_corrupt_artifacts(self):
        path = self.root / 'scp_statements.csv'
        original = path.read_bytes()
        path.write_bytes(b'x' * len(original))
        with self.assertRaisesRegex(ValueError, 'Corrupted'):
            cache.validate_cache(self.root)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            cache.validate_cache(self.root)

    def test_missing_shard(self):
        shard = next(p for p in (self.root / 'signals.zarr/signals/c').rglob('*') if p.is_file())
        shard.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            cache.open_cache(self.root)

    def test_incomplete_and_incompatible_cache(self):
        (self.root / '.complete.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            cache.validate_cache(self.root)
        self.publish()
        manifest = json.loads((self.root / 'manifest.json').read_text())
        manifest['sampling_rate'] = 500
        cache.write_json(self.root / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            cache.validate_cache(self.root)

    def test_stale_marker_and_duplicate_ids(self):
        cache.write_json(self.root / '.complete.json', dict(manifest_sha256='stale'))
        with self.assertRaisesRegex(ValueError, 'marker'):
            cache.validate_cache(self.root)
        group = zarr.open_group(str(self.root / 'signals.zarr'), mode='a')
        group['ecg_ids'][:] = [10, 10, 20]
        self.publish()
        with self.assertRaisesRegex(ValueError, 'mapping'):
            cache.open_cache(self.root)

    def test_staging_interruption_and_recovery(self):
        dest = Path(self.tmp.name) / 'staged'
        original = shutil.copyfile
        def interrupted(src, dst):
            if Path(src).name == 'scp_statements.csv':
                raise OSError('interrupted copy')
            return original(src, dst)
        with patch.object(cache.shutil, 'copyfile', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                cache.stage_cache(self.root, dest)
        self.assertFalse(dest.exists())
        self.assertFalse(cache.stage_cache(self.root, dest)['reused'])
        self.assertTrue(cache.stage_cache(self.root, dest)['reused'])
        (dest / 'scp_statements.csv').write_text('corrupt')
        self.assertFalse(cache.stage_cache(self.root, dest)['reused'])
        cache.validate_cache(dest)


class RealCacheParityTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('PTBXL_TEST_RAW_DIR') and config.ZARR_DIR is not None,
                         'Set PTBXL_TEST_RAW_DIR and PTBXL_ZARR_DIR for source/cache parity')
    def test_normalization_checkpoint_order_and_predictions(self):
        from model import ECGMamba
        from train import seed_everything
        seed_everything(42)
        cached, metadata = data.prepare_data(smoke=True)
        # Deliberately reorder IDs as an evaluation-only checkpoint may request.
        checkpoint = dict(metadata, record_ids={k: list(reversed(v)) for k, v in metadata['record_ids'].items()})
        cached, restored = data.prepare_data(smoke=True, checkpoint=checkpoint)
        with patch.object(config, 'ZARR_DIR', None), patch.object(config, 'DATA_DIR', Path(os.environ['PTBXL_TEST_RAW_DIR'])):
            raw, reference = data.prepare_data(smoke=True)
            ordered_raw, _ = data.prepare_data(smoke=True, checkpoint=checkpoint)
        self.assertEqual(metadata['normalization'], reference['normalization'])
        self.assertEqual(metadata['record_ids'], reference['record_ids'])
        model = ECGMamba(**config.MODEL).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            torch.save(model.state_dict(), path)
            model.load_state_dict(torch.load(path, weights_only=True))
        for name in cached:
            a, b = cached[name].dataset, ordered_raw[name].dataset
            np.testing.assert_array_equal(a.ecg_ids, b.ecg_ids)
            np.testing.assert_array_equal(a.signals, b.signals)
            np.testing.assert_array_equal(a.labels, b.labels)
            np.testing.assert_array_equal(a[0][0], b[0][0])
            with torch.no_grad():
                torch.testing.assert_close(model(torch.from_numpy(a[0][0][None])),
                                           model(torch.from_numpy(b[0][0][None])), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
