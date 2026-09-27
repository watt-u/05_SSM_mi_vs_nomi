"""Numerical and data-contract checks: python -m unittest -v test_experiment."""
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn
import config
from data import load_metadata, validate_splits, training_stats, ECGDataset, read_signal
from model import selective_scan, ECGMamba, BidirectionalMamba
from train import get_device, load_model, seed_everything


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(42)

    def test_config_dataset_path_in_modal_and_local_layouts(self):
        source = Path(config.__file__).read_text()
        for module_path, override, expected in [
            ('/experiment/config.py', '/data/ptb-xl/1.0.3', '/data/ptb-xl/1.0.3'),
            ('/repo/experiments/05_SSM_mi_vs_nomi/config.py', None, '/repo/datasets/ptb-xl'),
        ]:
            with self.subTest(module_path=module_path):
                namespace = {'__file__': module_path}
                with patch.dict(os.environ, {}, clear=True):
                    if override is not None:
                        os.environ['PTBXL_DATA_DIR'] = override
                    exec(compile(source, module_path, 'exec'), namespace)
                self.assertEqual(namespace['DATA_DIR'], Path(expected))

    def test_scan_values_and_gradients(self):
        # Independent literal recurrence, including a cross-chunk boundary.
        torch.manual_seed(7)
        values = [torch.randn(2, 137, 3, dtype=torch.double),
                  torch.rand(2, 137, 3, dtype=torch.double),
                  -torch.rand(3, 4, dtype=torch.double),
                  torch.randn(2, 137, 4, dtype=torch.double),
                  torch.randn(2, 137, 4, dtype=torch.double),
                  torch.randn(3, dtype=torch.double)]
        fast = [v.clone().requires_grad_() for v in values]
        ref = [v.clone().requires_grad_() for v in values]
        actual = selective_scan(*fast)
        u, delta, a, b, c, d = ref
        state = torch.zeros(2, 3, 4, dtype=torch.double)
        outputs = []
        for t in range(137):
            state = torch.exp(delta[:, t, :, None] * a) * state + delta[:, t, :, None] * b[:, t, None, :] * u[:, t, :, None]
            outputs.append((state * c[:, t, None, :]).sum(-1) + u[:, t] * d)
        expected = torch.stack(outputs, dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        weights = torch.randn_like(actual)
        ga = torch.autograd.grad((actual * weights).sum(), fast)
        gb = torch.autograd.grad((expected * weights).sum(), ref)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, atol=1e-9, rtol=1e-9)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_scan_matches_cpu(self):
        torch.manual_seed(3)
        values = [torch.randn(2, 137, 3), torch.rand(2, 137, 3),
                  -torch.rand(3, 4), torch.randn(2, 137, 4),
                  torch.randn(2, 137, 4), torch.randn(3)]
        cpu = [v.clone().requires_grad_() for v in values]
        mps = [v.to('mps').requires_grad_() for v in values]
        a, b = selective_scan(*cpu), selective_scan(*mps)
        torch.testing.assert_close(a, b.cpu(), atol=2e-5, rtol=2e-5)
        ga = torch.autograd.grad(a.square().sum(), cpu)
        gb = torch.autograd.grad(b.square().sum(), mps)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y.cpu(), atol=3e-3, rtol=3e-4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_scan_matches_cpu(self):
        torch.manual_seed(3)
        values = [torch.randn(2, 137, 3), torch.rand(2, 137, 3),
                  -torch.rand(3, 4), torch.randn(2, 137, 4),
                  torch.randn(2, 137, 4), torch.randn(3)]
        cpu = [v.clone().requires_grad_() for v in values]
        cuda = [v.to('cuda').requires_grad_() for v in values]
        a, b = selective_scan(*cpu), selective_scan(*cuda)
        torch.testing.assert_close(a, b.cpu(), atol=2e-5, rtol=2e-5)
        ga = torch.autograd.grad(a.square().sum(), cpu)
        gb = torch.autograd.grad(b.square().sum(), cuda)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y.cpu(), atol=3e-3, rtol=3e-4)

    def test_device_selection(self):
        with patch('train.torch.cuda.is_available', return_value=True):
            self.assertEqual(get_device().type, 'cuda')
        with patch('train.torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA requested'):
                get_device('cuda')

    def test_reverse_alignment(self):
        block = BidirectionalMamba(d_model=4, d_state=2, dropout=0).eval()
        class Prefix(nn.Module):
            def forward(self, x):
                return x.cumsum(1)
        block.forward_branch = Prefix()
        block.backward_branch = Prefix()
        x = torch.randn(2, 9, 4)
        h = block.norm(x)
        # Explicit suffix sum gives the backward branch at each original position.
        suffix = torch.stack([h[:, t:].sum(1) for t in range(9)], 1)
        expected = x + block.merge(torch.cat([h.cumsum(1), suffix], -1))
        torch.testing.assert_close(block(x), expected)

    def test_shapes_gradients_and_lead_isolation(self):
        model = ECGMamba(**config.MODEL).eval()
        x = torch.randn(1, 12, 1000)
        with torch.no_grad():
            encoded = model.encode_leads(x)
            changed = x.clone()
            changed[:, 3] += 2
            other = model.encode_leads(changed)
        self.assertEqual(encoded.shape, (1, 12, 250, 32))
        torch.testing.assert_close(encoded[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]], other[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]])
        self.assertFalse(torch.allclose(encoded[:, 3], other[:, 3]))
        y = model(x)
        self.assertEqual(y.shape, (1, 1))
        y.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_predictions(self):
        model = ECGMamba(**config.MODEL).eval()
        x = torch.randn(1, 12, 40)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            torch.save(dict(model_config=config.MODEL, model_state_dict=model.state_dict()), path)
            restored, _ = load_model(path, torch.device('cpu'))
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)

    def test_metadata_and_patient_separation(self):
        df = load_metadata()
        groups = validate_splits(df, config.SPLITS)
        self.assertEqual([len(groups[n]) for n in ('train', 'val', 'test')], [17418, 2183, 2198])
        altered = df.copy()
        altered.loc[groups['val'].index[0], 'patient_id'] = groups['train'].iloc[0].patient_id
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_splits(altered, config.SPLITS)
        # Compare the existing baseline's definition without importing its config.
        import ast
        scp = pd.read_csv(config.DATA_DIR / 'scp_statements.csv', index_col=0)
        mapping = scp[scp.diagnostic == 1].diagnostic_class.to_dict()
        expected = df.scp_codes.map(lambda s: any(mapping.get(c) == 'MI' for c in ast.literal_eval(s)))
        np.testing.assert_array_equal(df.mi, expected.astype(int))

    def test_training_only_normalization(self):
        train = np.random.default_rng(1).normal(size=(3, 12, 20)).astype(np.float32)
        norm = training_stats(train)
        frame = pd.DataFrame({'mi': [0, 1, 0]})
        val = ECGDataset(train + 100, frame, norm)
        np.testing.assert_allclose(val[0][0], (train[0] + 100 - np.asarray(norm['mean'])[:, None]) / np.asarray(norm['std'])[:, None], rtol=1e-6)
        np.testing.assert_allclose(training_stats(train)['mean'], norm['mean'])

    def test_invalid_signal_and_lead_reordering(self):
        row = pd.Series({'filename_lr': 'unused'}, name=123)
        signal = np.tile(np.arange(12), (1000, 1)).astype(float)
        meta = dict(fs=100, sig_name=list(reversed(config.LEADS)))
        with patch('data.wfdb.rdsamp', return_value=(signal, meta)):
            np.testing.assert_array_equal(read_signal(row)[:, 0], np.arange(12)[::-1])
        for bad_signal, bad_meta in [(signal, dict(meta, fs=500)), (signal[:10], meta), (signal * np.nan, meta)]:
            with patch('data.wfdb.rdsamp', return_value=(bad_signal, bad_meta)):
                with self.assertRaisesRegex(ValueError, 'ecg_id=123'):
                    read_signal(row)


if __name__ == '__main__':
    unittest.main()
