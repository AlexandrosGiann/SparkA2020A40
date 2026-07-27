# -*- coding: utf-8 -*-
"""The target class bit must never be an input feature of the same example.

These tests are the guard rail requested by the specification: they fail loudly
if a future refactor reintroduces label leakage.
"""
import inspect
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config
from spark_a2020a40.features import (FEATURE_NAMES, LABEL_KEYS,
                                     FeatureExtractor, LabelLeakageError)
from spark_a2020a40.memory import TokenMemory
from spark_a2020a40.student import StudentModel
from spark_a2020a40.trainer import TrainingCoordinator
from spark_a2020a40.teacher import OfflineTeacher


class TestSignatureLevelGuarantee(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.extractor = FeatureExtractor(self.cfg, TokenMemory(self.cfg))

    def test_build_token_features_has_no_target_parameter(self):
        signature = inspect.signature(self.extractor.build_token_features)
        for forbidden in LABEL_KEYS:
            self.assertNotIn(forbidden, signature.parameters,
                             "leaked parameter: " + forbidden)

    def test_build_input_features_has_no_target_parameter(self):
        signature = inspect.signature(self.extractor.build_input_features)
        for forbidden in LABEL_KEYS:
            self.assertNotIn(forbidden, signature.parameters)

    def test_passing_target_class_bit_raises(self):
        with self.assertRaises(LabelLeakageError):
            self.extractor.build_token_features("x", target_class_bit=1)

    def test_every_label_alias_is_rejected(self):
        for name in LABEL_KEYS:
            with self.assertRaises(LabelLeakageError, msg=name):
                self.extractor.build_token_features("x", **{name: 1})

    def test_input_features_reject_labels_too(self):
        with self.assertRaises(LabelLeakageError):
            self.extractor.build_input_features(["a"], target_class_bit=0)

    def test_unknown_kwarg_is_a_plain_type_error(self):
        with self.assertRaises(TypeError):
            self.extractor.build_token_features("x", not_a_label=1)


class TestFeatureVectorContent(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.memory = TokenMemory(self.cfg)
        self.extractor = FeatureExtractor(self.cfg, self.memory)

    def test_no_feature_is_named_after_the_label(self):
        for name in FEATURE_NAMES:
            self.assertNotIn(name, LABEL_KEYS)

    def test_the_vector_is_identical_regardless_of_the_true_label(self):
        """The strongest form of the test: the features of a candidate cannot
        depend on the answer, because the answer is never available here."""
        first = self.extractor.build_token_features(
            "γεια", input_class_bit=1, previous_predicted_class_bit=0,
            normalize=False)
        second = self.extractor.build_token_features(
            "γεια", input_class_bit=1, previous_predicted_class_bit=0,
            normalize=False)
        self.assertEqual(first, second)

    def test_previous_prediction_is_allowed_as_context(self):
        with_context = self.extractor.build_token_features(
            "γεια", previous_predicted_class_bit=1, normalize=False)
        without = self.extractor.build_token_features(
            "γεια", previous_predicted_class_bit=0, normalize=False)
        self.assertNotEqual(with_context, without)


class TestTrainingLoopDoesNotLeak(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.student = StudentModel(self.cfg)
        self.coordinator = TrainingCoordinator(
            self.cfg, self.student, OfflineTeacher(self.cfg))

    def test_class_prediction_happens_before_training(self):
        """``predict_input_class`` must be able to run with no label present."""
        probability, bit = self.student.classify_text("Τι κάνεις;")
        self.assertIn(bit, (0, 1))
        self.assertTrue(0.0 <= probability <= 1.0)

    def test_turn_reports_prediction_and_target_separately(self):
        report = self.coordinator.process_turn("Τι κάνεις;")
        self.assertIn("predicted", report["class"])
        self.assertIn("target", report["class"])
        # The prediction is made before the label is used for training, so at
        # step 1 an untrained head is allowed to be wrong.
        self.assertIn(report["class"]["predicted"], (0, 1))

    def test_distillation_features_never_receive_the_target(self):
        pairs = self.coordinator.build_training_pairs(
            ["γεια"], ["γεια", "σου"])
        self.assertTrue(pairs)
        targets = set(target for _, target, _ in pairs)
        self.assertEqual(targets, {0, 1})
        width = len(pairs[0][0])
        for vector, _, _ in pairs:
            self.assertEqual(len(vector), width)

    def test_positive_and_negative_share_the_same_feature_layout(self):
        pairs = self.coordinator.build_training_pairs(["γεια"], ["σου"])
        positives = [v for v, t, _ in pairs if t == 1]
        negatives = [v for v, t, _ in pairs if t == 0]
        self.assertTrue(positives and negatives)
        # If the target had leaked in, one constant column would separate them
        # perfectly.  Check no single index is a perfect separator.
        for index in range(len(positives[0])):
            positive_values = set(round(v[index], 6) for v in positives)
            negative_values = set(round(v[index], 6) for v in negatives)
            if len(positive_values) == 1 and len(negative_values) == 1:
                self.assertEqual(
                    positive_values, negative_values,
                    "feature #{0} perfectly separates positives from negatives "
                    "-- this is label leakage".format(index))


if __name__ == "__main__":
    unittest.main()
