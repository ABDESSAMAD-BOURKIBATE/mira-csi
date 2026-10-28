"""Interface/integrity checks; no model forward or optimizer calls."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import predict_spectral_e as api


class InputChecks(unittest.TestCase):
    def setUp(self):
        self.values=np.load(HERE/'demo/first-held-E1-power18.npy',allow_pickle=False)

    def test_shape_and_real_numeric_contract(self):
        for bad in (np.zeros((500,18)),np.zeros((18,499)),np.zeros((1,18,500)),
                    np.full((18,500),'one'),np.ones((18,500),dtype=complex)):
            with self.subTest(shape=bad.shape,dtype=str(bad.dtype)),self.assertRaises(ValueError):
                api.checked_input(bad)

    def test_mask_validated_before_boolean_cast(self):
        for bad_value in (np.nan,np.inf,-np.inf,.2,2,-1):
            mask=np.ones(500)
            mask[12]=bad_value
            with self.subTest(value=str(bad_value)),self.assertRaises(ValueError):
                api.checked_input(self.values,mask)
        for bad in (np.ones((1,500)),np.ones(499),np.full(500,'1'),np.ones(500,dtype=complex)):
            with self.subTest(shape=bad.shape),self.assertRaises(ValueError):
                api.checked_input(self.values,bad)

    def test_finite_binary_and_boolean_masks_agree(self):
        mask=np.load(HERE/'demo/first-held-E1-B50-mask.npy',allow_pickle=False)
        a,ma=api.checked_input(self.values,mask)
        b,mb=api.checked_input(self.values,mask.astype(float))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(ma,mb)
        self.assertEqual(ma.dtype,np.bool_)

    def test_nonfinite_observed_rejected_and_empty_mask_rejected(self):
        for value in (np.nan,np.inf,-np.inf):
            broken=self.values.copy()
            broken[3,40]=value
            with self.assertRaises(ValueError):
                api.checked_input(broken)
        with self.assertRaises(ValueError):
            api.checked_input(self.values,np.zeros(500,dtype=bool))


class FeatureAndIntegrityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor=api.SpectralEPredictor()
        cls.values=np.load(HERE/'demo/first-held-E1-power18.npy',allow_pickle=False)
        cls.mask=np.load(HERE/'demo/first-held-E1-B50-mask.npy',allow_pickle=False)

    def test_hidden_fills_do_not_change_actual_mask_features(self):
        reference,unused=self.predictor.features(self.values,self.mask)
        for fill in (np.nan,np.inf,-np.inf,1e20):
            hidden=self.values.astype(np.float64)
            hidden[:,~self.mask]=fill
            features,mask=self.predictor.features(hidden,self.mask)
            np.testing.assert_array_equal(features,reference)
            np.testing.assert_array_equal(mask,self.mask)

    def test_invalid_mask_rejected_before_feature_extraction(self):
        with patch.object(self.predictor,'extract_features') as extract:
            mask=self.mask.astype(float)
            mask[0]=np.nan
            with self.assertRaises(ValueError):
                self.predictor.features(self.values,mask)
            extract.assert_not_called()

    def test_bound_classes_temperature_and_parameter_count(self):
        self.assertEqual(self.predictor.classes,['sit','fall_from_sit','lie','stand','walk_tx_rx','sit_down','pick_up'])
        self.assertEqual(self.predictor.temperature,0.8688099152312774)
        self.assertEqual(sum(p.numel() for p in self.predictor.model.parameters()),25415)

    def test_tampered_files_rejected_before_checkpoint_deserialization(self):
        for relative in ('artifacts/checkpoints/E_E1_s20261026.pt',
                         'artifacts/validation/E_E1_s20261026.npz','frozen/protocol.json'):
            with self.subTest(file=relative),tempfile.TemporaryDirectory(prefix='mira-public-interface-') as directory:
                clone=Path(directory)/'package'
                # The explicitly scoped temporary directory owns this copy only.
                assert clone.resolve().is_relative_to(Path(directory).resolve())
                shutil.copytree(HERE,clone,ignore=shutil.ignore_patterns('__pycache__'))
                target=clone/relative
                target.write_bytes(target.read_bytes()+b'altered-for-rejection-test')
                with patch.object(api,'_load_checkpoint') as deserialize:
                    with self.assertRaises(api.IntegrityError):
                        api.SpectralEPredictor(package_root=clone)
                    deserialize.assert_not_called()

    def test_verified_root_is_the_actual_loaded_module_root(self):
        with tempfile.TemporaryDirectory(prefix='mira-verified-root-') as directory:
            clone=Path(directory)/'package'
            assert clone.resolve().is_relative_to(Path(directory).resolve())
            shutil.copytree(HERE,clone,ignore=shutil.ignore_patterns('__pycache__'))
            predictor=api.SpectralEPredictor(package_root=clone)
            self.assertEqual(Path(predictor.extract_features.__code__.co_filename).resolve(),
                             (clone/'frozen/features.py').resolve())
            module=sys.modules[type(predictor.model).__module__]
            self.assertEqual(Path(module.__file__).resolve(),(clone/'frozen/models.py').resolve())

    def test_extreme_finite_input_rejected_before_trained_model_forward(self):
        extreme=np.empty((18,500),dtype=np.float64)
        extreme[:,::2]=1e308
        extreme[:,1::2]=-1e308
        self.assertTrue(np.isfinite(extreme).all())
        with patch.object(self.predictor,'model') as network:
            with self.assertRaisesRegex(ValueError,'failed numerically'):
                self.predictor.predict(extreme)
            network.assert_not_called()

    def test_nonfinite_features_rejected(self):
        for value in (np.nan,np.inf,-np.inf):
            with self.subTest(value=str(value)),patch.object(self.predictor,'extract_features',
                    return_value=SimpleNamespace(values=np.full(360,value))):
                with self.assertRaisesRegex(ValueError,'non-finite'):
                    self.predictor.features(np.zeros((18,500)))

    def test_nonfinite_mock_logits_rejected_without_real_forward(self):
        for value in (np.nan,np.inf,-np.inf):
            with self.subTest(value=str(value)),patch.object(self.predictor,'features',
                    return_value=(np.zeros(360,dtype=np.float32),np.ones(500,dtype=bool))),\
                    patch.object(self.predictor,'model',return_value=torch.full((1,7),value)):
                with self.assertRaisesRegex(ValueError,'non-finite'):
                    self.predictor.predict(np.zeros((18,500)))

    def test_nonfinite_mock_probabilities_rejected_without_real_forward(self):
        with patch.object(self.predictor,'features',
                    return_value=(np.zeros(360,dtype=np.float32),np.ones(500,dtype=bool))),\
                patch.object(self.predictor,'model',return_value=torch.zeros((1,7))),\
                patch.object(api.np,'exp',side_effect=[np.ones(7),np.full(7,np.nan)]):
            with self.assertRaisesRegex(ValueError,'non-finite'):
                self.predictor.predict(np.zeros((18,500)))


if __name__=='__main__':
    unittest.main(verbosity=2)
