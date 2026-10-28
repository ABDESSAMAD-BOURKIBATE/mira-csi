"""Pipeline controls on synthetic inputs; no held/familiar data inference."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from models import CandidateMIRA
from operators import augmentation_masks
from pipeline import CleanFeatureCache, prepare_batch
from run_training import distribution, expected_tags, scores, verify_freeze

torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def test_hidden_fills_cannot_reach_normalization_features_or_logits(self):
        rng = np.random.default_rng(20261028)
        raw = rng.normal(size=(3, 18, 500)).astype(np.float32)
        masks = rng.random((3, 500)) > .5
        mean = rng.normal(size=(1, 18, 1))
        scale = np.linspace(.5, 2., 18)[None, :, None]
        for variant in 'ABCDE':
            torch.manual_seed(7)
            model = CandidateMIRA(variant).eval()
            reference = prepare_batch(raw, masks, np.arange(3), variant, mean, scale)
            for fill in [np.nan, np.inf, -1e30]:
                altered = np.where(masks[:, None, :], raw, fill)
                result = prepare_batch(altered, masks, np.arange(3), variant, mean, scale)
                for before, after in zip(reference, result):
                    if before is not None:
                        np.testing.assert_array_equal(before, after)
                with torch.inference_mode():
                    logits1 = model(*(None if x is None else torch.as_tensor(x) for x in reference))
                    logits2 = model(*(None if x is None else torch.as_tensor(x) for x in result))
                torch.testing.assert_close(logits1, logits2, rtol=0, atol=0)

    def test_source_normalization_then_exact_zero_fill(self):
        raw = np.full((1, 18, 500), 10., dtype=np.float32)
        masks = np.ones((1, 500), bool)
        masks[:, 100:350] = False
        prepared = prepare_batch(raw, masks, [0], 'D', np.full((1, 18, 1), 6.), np.full((1, 18, 1), 2.))
        np.testing.assert_array_equal(prepared[0][:, :, masks[0]], 2.)
        np.testing.assert_array_equal(prepared[0][:, :, ~masks[0]], 0.)

    def test_clean_cache_never_stores_augmented_feature_views(self):
        rng = np.random.default_rng(11)
        raw = rng.normal(size=(1, 18, 500)).astype(np.float32)
        cache = CleanFeatureCache()
        mean, scale = np.zeros((1, 18, 1)), np.ones((1, 18, 1))
        for variant in 'ABCE':
            prepare_batch(raw, np.ones((1, 500), bool), [123], variant, mean, scale, cache)
        self.assertEqual(len(cache.items), 3)  # E reuses fixed A features.
        keys = set(cache.items)
        masks = np.ones((1, 500), bool)
        masks[:, 100:350] = False
        prepare_batch(raw, masks, [123], 'A', mean, scale, cache)
        self.assertEqual(set(cache.items), keys)

    def test_augmentation_order_is_independent_of_torch_model_randomness(self):
        entries = [{'name': f'fixture-{i}'} for i in range(128)]
        one, two = np.random.default_rng(20261026), np.random.default_rng(20261026)
        order1, order2 = one.permutation(128), two.permutation(128)
        np.testing.assert_array_equal(order1, order2)
        expected = augmentation_masks(order1, entries, one)
        torch.manual_seed(17)
        CandidateMIRA('E')
        torch.rand(1234)
        actual = augmentation_masks(order2, entries, two)
        np.testing.assert_array_equal(expected, actual)

    def test_source_freeze_gate_rejects_empty_or_partial_study(self):
        config = {'folds': [1, 2, 3], 'training_seeds': list(range(20261026, 20261031)),
                  'variants': list('ABCDE')}
        self.assertEqual(len(expected_tags(config)), 75)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                verify_freeze(Path(directory), config, 'not-a-protocol')

    def test_probability_metrics_independently_known_reference(self):
        truth = np.arange(7)
        p, logp = distribution(np.zeros((7, 7)))
        metric = scores(truth, p, logp)
        self.assertAlmostEqual(metric['nll'], np.log(7.))
        self.assertAlmostEqual(metric['brier'], 6 / 7)
        self.assertAlmostEqual(metric['accuracy'], 1 / 7)
        self.assertAlmostEqual(metric['macro_f1'], (2 / 8) / 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
