"""Artifact regressions only: no training or target/model inference."""
from pathlib import Path
import copy, hashlib, json, sys, tempfile, unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrity import (IntegrityError, artifact_paths, calibration_scope, completed_or_available,
                       file_sha256, read_json, run_context, run_identity, seal_completed_run,
                       text_sha256, verify_checkpoint_metadata, verify_completed_run, verify_input_hash)


class ArtifactIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'; self.root.mkdir()
        self.out = Path(self.temp.name) / 'results'
        for directory in ['data', 'vendor/minirocket', 'evaluation']:
            (self.root / directory).mkdir(parents=True)
        self.config = {'classes': list('abcdefg'), 'training_seeds': [20261026], 'folds': [1]}
        (self.root / 'protocol.json').write_text(json.dumps(self.config) + '\n')
        for n in ['run_experiment.py', 'integrity.py', 'mira_model.py', 'missingness.py', 'vendor/minirocket/minirocket_multivariate.py']:
            (self.root / n).write_bytes(b'# fixture source\nvalue = 1\n')
        manifest = b'{"selected": []}\r\n'
        (self.root / 'data/manifest.json').write_bytes(manifest)
        (self.root / 'data/band_streams.npy').write_bytes(b'fixture binary\r\n')
        receipt = {'sha256': {n: file_sha256(self.root / 'data' / n) for n in ['manifest.json', 'band_streams.npy']}}
        (self.root / 'input_receipt.json').write_text(json.dumps(receipt) + '\n')
        (self.root / 'evaluation/legacy-release-manifest.json').write_text(json.dumps({
            'archive_sha256': '2f50e1886a84837bf8e51fb80ec418fa06f8c46da9ba108e675f36f5bee1dfbd', 'pairs': {}}))
        self.identity = run_identity('mira_csi', 20261026, 1)
        self.tag = 'mira_csi_E1_s20261026'; self.context = run_context(self.root)
        self.state = {'variant': 'mira_csi', 'seed': 20261026, 'target_environment': 1,
                      'protocol_sha256': text_sha256(self.root / 'protocol.json'),
                      'classes': self.config['classes'], 'mean': np.zeros((1, 18, 1)),
                      'scale': np.ones((1, 18, 1))}

    def artifacts(self):
        for n in artifact_paths(self.out, self.tag, 'mira_csi'):
            p = self.out / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'complete artifact bytes')
        (self.out / 'runs').mkdir(exist_ok=True)

    def sealed(self):
        self.artifacts()
        seal_completed_run(self.out, self.tag, {'temperature': 1.5}, self.identity, self.context)
        return self.out / 'runs' / f'{self.tag}.json'

    def test_lf_crlf_manifest_compatibility_is_narrow(self):
        path = self.root / 'data/manifest.json'; expected = file_sha256(path)
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
        verify_input_hash(path, expected, text=True)
        with self.assertRaises(IntegrityError): verify_input_hash(path, expected, text=False)
        path.write_bytes(b'{"selected": [1]}\n')
        with self.assertRaises(IntegrityError): verify_input_hash(path, expected, text=True)

    def test_binary_input_is_always_byte_exact(self):
        path = self.root / 'data/band_streams.npy'; expected = file_sha256(path)
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
        with self.assertRaises(IntegrityError): verify_input_hash(path, expected)

    def test_checkout_newlines_preserve_context_and_completed_reuse(self):
        self.sealed()
        for p in self.root.rglob('*'):
            if p.is_file() and p.suffix in {'.py', '.json'}:
                p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
        self.assertEqual(run_context(self.root), self.context)
        self.assertTrue(completed_or_available(self.out, self.tag, self.identity, run_context(self.root)))

    def test_changed_source_or_protocol_refuses_reuse(self):
        self.sealed()
        for name in ['mira_model.py', 'protocol.json']:
            p = self.root / name; original = p.read_bytes(); p.write_bytes(original + b'\n')
            with self.assertRaises(IntegrityError): verify_completed_run(self.out, self.tag, self.identity, run_context(self.root))
            p.write_bytes(original)

    def test_missing_corrupted_and_truncated_artifacts_refuse_reuse(self):
        self.sealed()
        for name in artifact_paths(self.out, self.tag, 'mira_csi'):
            p = self.out / name; original = p.read_bytes()
            for replacement in [b'', b'truncated', b'different checkpoint with same tag']:
                p.write_bytes(replacement)
                with self.assertRaises(IntegrityError): verify_completed_run(self.out, self.tag, self.identity, self.context)
            p.unlink()
            with self.assertRaises(IntegrityError): verify_completed_run(self.out, self.tag, self.identity, self.context)
            p.write_bytes(original)

    def test_unsealed_or_truncated_json_and_orphans_are_not_reused(self):
        path = self.sealed(); good = path.read_bytes()
        for raw in [b'{"temperature":1}', b'{"_integrity":', b'[]']:
            path.write_bytes(raw)
            with self.assertRaises(IntegrityError): completed_or_available(self.out, self.tag, self.identity, self.context)
        path.unlink()
        with self.assertRaises(IntegrityError): completed_or_available(self.out, self.tag, self.identity, self.context)
        self.assertFalse((path.with_name(path.name + '.tmp')).exists())

    def test_current_calibration_is_bound_to_actual_checkpoint(self):
        path = self.sealed(); checkpoint = self.out / 'checkpoints' / f'{self.tag}.pt'
        self.assertEqual(calibration_scope(read_json(path), path, checkpoint, self.state, self.root), 'current_checkpoint_bound')
        # Inference needs neither the large training tensor nor target inference.
        (self.root / 'data/band_streams.npy').unlink()
        self.assertEqual(calibration_scope(read_json(path), path, checkpoint, self.state, self.root), 'current_checkpoint_bound')
        checkpoint.write_bytes(b'replacement weights under identical filename')
        with self.assertRaises(IntegrityError): calibration_scope(read_json(path), path, checkpoint, self.state, self.root)

    def test_temperature_identity_class_and_protocol_mismatches_rejected(self):
        path = self.sealed(); checkpoint = self.out / 'checkpoints' / f'{self.tag}.pt'
        for value in [0, -1, float('nan'), float('inf'), 100, True]:
            meta = read_json(path); meta['temperature'] = value
            with self.assertRaises(IntegrityError): calibration_scope(meta, path, checkpoint, self.state, self.root)
        for field, value in [('classes', list('gfedcba')), ('protocol_sha256', '0' * 64),
                             ('seed', 123), ('variant', 'unrelated'), ('scale', np.zeros((1, 18, 1)))]:
            state = dict(self.state, **{field: value})
            with self.assertRaises(IntegrityError): verify_checkpoint_metadata(state, self.root)
        meta = read_json(path); meta['training_seed'] = 123
        with self.assertRaises(IntegrityError): calibration_scope(meta, path, checkpoint, self.state, self.root)

    def test_valid_but_changed_temperature_is_not_silently_accepted(self):
        path = self.sealed(); metadata = read_json(path); metadata['temperature'] = 2.0
        checkpoint = self.out / 'checkpoints' / f'{self.tag}.pt'
        with self.assertRaisesRegex(IntegrityError, 'changed'):
            calibration_scope(metadata, path, checkpoint, self.state, self.root)
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(IntegrityError, 'changed'):
            verify_completed_run(self.out, self.tag, self.identity, self.context)

    def test_exact_legacy_pair_and_explicit_unknown_opt_in(self):
        path = self.sealed(); checkpoint = self.out / 'checkpoints' / f'{self.tag}.pt'
        path.write_text(json.dumps({'temperature': 1.5}))
        with self.assertRaises(IntegrityError): calibration_scope(read_json(path), path, checkpoint, self.state, self.root)
        self.assertEqual(calibration_scope(read_json(path), path, checkpoint, self.state, self.root, True), 'legacy_unbound_explicit_opt_in')
        registry = self.root / 'evaluation/legacy-release-manifest.json'; obj = read_json(registry)
        obj['pairs'][self.tag] = {'checkpoint_sha256': file_sha256(checkpoint), 'run_json_sha256': file_sha256(path)}
        registry.write_text(json.dumps(obj))
        self.assertTrue(calibration_scope(read_json(path), path, checkpoint, self.state, self.root).startswith('exact_legacy_v0.1.0'))
        path.write_text(json.dumps({'temperature': 2.0}))
        with self.assertRaises(IntegrityError): calibration_scope(read_json(path), path, checkpoint, self.state, self.root)

    def test_predictor_rejects_bad_state_before_constructing_model(self):
        import predict
        bad = dict(self.state, protocol_sha256='0' * 64)
        with patch.object(predict, 'ROOT', self.root), patch.object(predict.torch, 'load', return_value=bad), \
             patch.object(predict, 'ActivityCNN', side_effect=AssertionError('Model must not be instantiated')), \
             patch.object(sys, 'argv', ['predict.py', '--checkpoint', 'fixture.pt', '--input', 'not-read.npy']):
            with self.assertRaises(IntegrityError): predict.main()

    def test_supplied_masks_require_finite_binary_values(self):
        from predict import load_observation_mask
        path = Path(self.temp.name) / 'observed.npy'; x = np.ones((18, 500), np.float32)
        for mask in [np.r_[np.nan, np.ones(499)], np.full(500, 2), np.full(500, .5),
                     np.ones((1, 500)), np.zeros(500), np.full(500, 'True'), np.ones(500, complex)]:
            np.save(path, mask)
            with self.assertRaises(IntegrityError): load_observation_mask(path, x)
        expected = np.arange(500) % 2 == 0
        for mask in [expected, expected.astype(np.float32), expected.astype(np.int64)]:
            np.save(path, mask); np.testing.assert_array_equal(load_observation_mask(path, x), expected)
        x[:, 3] = np.nan
        implicit = load_observation_mask(None, x)
        self.assertFalse(implicit[3]); self.assertEqual(implicit.sum(), 499)

    def test_predictor_rejects_nonfinite_mock_logits(self):
        import predict
        path = Path(self.temp.name) / 'synthetic.npy'; np.save(path, np.ones((18, 500), np.float32))
        model = MagicMock(return_value=predict.torch.full((1, 7), float('nan')))
        state = dict(self.state, state_dict={})
        with patch.object(predict, 'ROOT', self.root), patch.object(predict.torch, 'load', return_value=state), \
             patch.object(predict, 'ActivityCNN', return_value=model), \
             patch.object(sys, 'argv', ['predict.py', '--checkpoint', 'fixture.pt', '--input', str(path)]):
            with self.assertRaisesRegex(IntegrityError, 'logits'): predict.main()


if __name__ == '__main__': unittest.main()
