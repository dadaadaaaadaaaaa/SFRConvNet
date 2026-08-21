import inspect
import unittest

from Experiment.prepare_data import CachePt
from Experiment.run_train import load_config
from Experiment.train import parse_args


class NoAugmentationDefaultsTests(unittest.TestCase):
    def test_offline_mirror_is_disabled_by_default(self):
        parameter = inspect.signature(CachePt.process).parameters["mirror"]
        self.assertIs(parameter.default, False)

    def test_online_spatial_augmentation_is_disabled_by_default(self):
        args = parse_args([])
        self.assertEqual(args.sfr_aug_prob, 0.0)

    def test_noaug_training_alias_forces_zero_probability(self):
        config = load_config("Exp2_NoAug_SFRConvNet")
        self.assertEqual(config["sfr_aug_prob"], 0.0)


if __name__ == "__main__":
    unittest.main()
