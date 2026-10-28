"""Scientific calculation and synthetic-model checks; no real target inference."""
import unittest
import numpy as np
import torch
from scipy.linalg import eigh
from features import FeatureConfig, extract_features, pair_indices, pair_moments
from models import CandidateMIRA, parameter_count

torch.set_num_threads(2)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261028)
        self.x = self.rng.normal(size=(18, 500))
        self.mask = self.rng.random(500) > .5

    def test_hand_calculated_moments_and_independent_scipy_spectrum(self):
        x = np.array([[0., 1, 3, 1, 2, 0], [2., -1, 0, 3, 1, 4]])
        mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        left, right = pair_indices(mask, 0, 6, 1)
        np.testing.assert_array_equal(left, [0, 3, 4])
        np.testing.assert_array_equal(right, [1, 4, 5])
        mean, numerator, denominator = pair_moments(x, left, right)
        np.testing.assert_allclose(mean, [1., 5 / 3], atol=1e-14)
        expected_numerator = np.array([[1., -11 / 6], [-11 / 6, 11 / 3]])
        expected_denominator = np.array([[2 / 3, -2 / 3], [-2 / 3, 23 / 9]])
        np.testing.assert_allclose(numerator, expected_numerator, atol=1e-14)
        np.testing.assert_allclose(denominator, expected_denominator, atol=1e-14)
        config = FeatureConfig(channels=2, length=6, lags=(1,), cells=((0, 6),), min_pairs=1)
        result = extract_features(x, mask, 'A', config)
        # Independent LAPACK generalized eigensolver, not extractor whitening.
        spectrum = eigh(expected_numerator, expected_denominator, eigvals_only=True)
        expected = np.log(.999 * spectrum + .001)
        np.testing.assert_allclose(result.values[:2], expected, atol=1e-7)
        np.testing.assert_allclose(result.values[2:], [1, .6], atol=1e-7)

    def test_diagonal_control_has_hand_calculated_ratios(self):
        x = np.array([[0., 1, 3, 1, 2, 0], [2., -1, 0, 3, 1, 4]])
        mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        config = FeatureConfig(channels=2, length=6, lags=(1,), cells=((0, 6),), min_pairs=1)
        result = extract_features(x, mask, 'C', config)
        expected_ratios = np.array([1 / (2 / 3), (11 / 3) / (23 / 9)])
        np.testing.assert_allclose(result.values[:2], np.log(.999 * expected_ratios + .001), atol=1e-7)
        self.assertTrue(all(d['rank'] is None for d in result.diagnostics))

    def test_hidden_fill_including_nonfinite_is_never_read(self):
        for variant in 'ABC':
            reference = extract_features(self.x, self.mask, variant).values
            for fill in [np.nan, np.inf, -np.inf, -1e30]:
                changed = self.x.copy()
                changed[:, ~self.mask] = fill
                actual = extract_features(changed, self.mask, variant).values
                np.testing.assert_array_equal(actual, reference)

    def test_observed_nonfinite_is_rejected(self):
        values = self.x.copy()
        values[0, np.flatnonzero(self.mask)[0]] = np.nan
        with self.assertRaises(ValueError):
            extract_features(values, self.mask)

    def test_no_observations_and_insufficient_pairs(self):
        for variant in 'ABC':
            result = extract_features(self.x, np.zeros(500, bool), variant)
            np.testing.assert_array_equal(result.values, np.zeros(360))
            self.assertFalse(any(d['valid'] for d in result.diagnostics))
            sparse = np.zeros(500, bool)
            sparse[[0, 1, 3]] = True
            result = extract_features(self.x, sparse, variant)
            np.testing.assert_array_equal(result.values[:342], np.zeros(342))
            self.assertTrue(any(d['pairs'] > 0 for d in result.diagnostics))

    def test_constant_trace_is_invalid_not_a_fabricated_confidence(self):
        for variant in 'ABC':
            result = extract_features(np.full((18, 500), 7.), self.mask, variant)
            np.testing.assert_array_equal(result.values[:342], np.zeros(342))
            self.assertFalse(any(d['valid'] for d in result.diagnostics))

    def test_rank_one_trace_retains_one_quotient_direction(self):
        signal = np.sin(np.linspace(0, 7, 500))
        x = np.arange(1, 19)[:, None] * signal
        result = extract_features(x, self.mask, 'A')
        self.assertTrue(all(d['rank'] == 1 for d in result.diagnostics))
        spectra = result.values[:324].reshape(18, 18)
        np.testing.assert_array_equal(spectra[:, 1:], np.zeros((18, 17)))

    def test_affine_invariance_well_conditioned_full_rank_A_and_B(self):
        q1, _ = np.linalg.qr(self.rng.normal(size=(18, 18)))
        q2, _ = np.linalg.qr(self.rng.normal(size=(18, 18)))
        mixing = q1 @ np.diag(np.linspace(.5, 2., 18)) @ q2
        self.assertLessEqual(np.linalg.cond(mixing), 10.)
        shifted = mixing @ self.x + self.rng.normal(size=(18, 1))
        for variant in 'AB':
            before = extract_features(self.x, self.mask, variant)
            after = extract_features(shifted, self.mask, variant)
            self.assertTrue(all(d['rank'] == 18 for d in before.diagnostics))
            self.assertTrue(all(d['rank'] == 18 for d in after.diagnostics))
            np.testing.assert_allclose(before.values, after.values, rtol=0, atol=1e-6)

    def test_diagonal_control_affine_invariance_has_narrower_scope(self):
        transformed = np.linspace(.5, 2., 18)[:, None] * self.x + self.rng.normal(size=(18, 1))
        np.testing.assert_allclose(extract_features(self.x, self.mask, 'C').values,
                                   extract_features(transformed, self.mask, 'C').values,
                                   rtol=0, atol=1e-6)

    def test_bad_condition_changes_numerical_rank_without_claimed_invariance(self):
        transformed = np.geomspace(1., 1e-6, 18)[:, None] * self.x
        before = extract_features(self.x, self.mask)
        after = extract_features(transformed, self.mask)
        self.assertTrue(any(a['rank'] < b['rank'] for a, b in zip(after.diagnostics, before.diagnostics)))
        self.assertTrue(np.isfinite(after.values).all())

    def test_matched_bound_and_input_immutability(self):
        original = self.x.copy()
        result = extract_features(self.x, self.mask)
        for d in result.diagnostics:
            if d['valid']:
                self.assertGreaterEqual(d['minimum_eigenvalue'], 0.)
                self.assertLessEqual(d['maximum_eigenvalue'], 2.)
        np.testing.assert_array_equal(self.x, original)

    def test_large_gap_is_not_time_compressed(self):
        mask = np.zeros(500, bool)
        mask[[0, 100, 101, 499]] = True
        left, right = pair_indices(mask, 0, 500, 1)
        np.testing.assert_array_equal(left, [100])
        np.testing.assert_array_equal(right, [101])


class ModelTests(unittest.TestCase):
    def test_dimensions_parameter_counts_and_finite_backward_on_synthetic_data(self):
        torch.manual_seed(20261028)
        values = torch.randn(4, 18, 500)
        mask = torch.ones(4, 1, 500)
        mask[:, :, 100:350] = 0
        distance = torch.zeros_like(mask)
        features = torch.randn(4, 360)
        counts = {}
        for variant in 'ABCDE':
            model = CandidateMIRA(variant)
            counts[variant] = parameter_count(model)
            output = model(values, mask, distance, features)
            self.assertEqual(tuple(output.shape), (4, 7))
            torch.nn.functional.cross_entropy(output, torch.arange(4)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
        self.assertEqual(counts['A'], counts['B'])
        self.assertEqual(counts['A'], counts['C'])
        self.assertEqual(counts['A'], 66279)
        self.assertEqual(counts['D'], 40871)
        self.assertEqual(counts['E'], 25415)
        self.assertEqual(counts['A'] - counts['D'], 25408)

    def test_temporal_hidden_zero_fill_even_if_hidden_is_nan(self):
        torch.manual_seed(3)
        model = CandidateMIRA('D').eval()
        values = torch.randn(2, 18, 500)
        mask = torch.ones(2, 1, 500)
        mask[:, :, 100:350] = 0
        distance = torch.zeros_like(mask)
        with torch.inference_mode():
            reference = model(values, mask, distance)
            changed = values.clone()
            changed[:, :, 100:350] = float('nan')
            actual = model(changed, mask, distance)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_actual_mask_features_are_required(self):
        model = CandidateMIRA('A')
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 18, 500), torch.ones(1, 1, 500), torch.zeros(1, 1, 500))


if __name__ == '__main__':
    unittest.main(verbosity=2)
