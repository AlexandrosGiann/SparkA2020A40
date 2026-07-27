# -*- coding: utf-8 -*-
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config
from spark_a2020a40.features import (FEATURE_NAMES, INPUT_FEATURE_NAMES,
                                     FeatureExtractor, N_FEATURES,
                                     RunningNormalizer, ord_features)
from spark_a2020a40.memory import TokenMemory
from spark_a2020a40.tokenizer import Tokenizer


class FeatureTestBase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.memory = TokenMemory(self.cfg)
        self.tokenizer = Tokenizer()
        self.extractor = FeatureExtractor(self.cfg, self.memory)
        self.memory.observe_sequence(
            self.tokenizer.tokenize_typed("γεια σου κόσμε γεια σου φίλε"),
            class_bit=1)


class TestRequiredFeatures(FeatureTestBase):
    def test_all_specified_features_exist(self):
        for name in ("commonality_in_data", "common_pos_1", "common_pos_2",
                     "common_pos_3", "input_class_bit", "error_probability",
                     "ord_sum", "ord_mean", "ord_weighted_sum",
                     "ord_weighted_sum_mod_257", "ord_weighted_sum_mod_263"):
            self.assertIn(name, FEATURE_NAMES)

    def test_length_covers_token_input_and_output(self):
        for name in ("length_token", "length_input", "length_output"):
            self.assertIn(name, FEATURE_NAMES)

    def test_repeats_cover_input_output_and_window(self):
        for name in ("repeats_in_input", "repeats_in_output", "repeats_in_window"):
            self.assertIn(name, FEATURE_NAMES)

    def test_vector_length_matches_names(self):
        vector = self.extractor.build_token_features("γεια")
        self.assertEqual(len(vector), N_FEATURES)
        self.assertEqual(len(FEATURE_NAMES), N_FEATURES)


class TestFeatureSemantics(FeatureTestBase):
    def raw(self, candidate, **kwargs):
        vector = self.extractor.build_token_features(
            candidate, normalize=False, **kwargs)
        return dict(zip(FEATURE_NAMES, vector))

    def test_commonality_is_normalised_not_a_raw_count(self):
        values = self.raw("γεια")
        self.assertGreater(values["commonality_in_data"], 0.0)
        self.assertLessEqual(values["commonality_in_data"], 1.0)

    def test_unknown_token_has_zero_commonality(self):
        self.assertEqual(self.raw("ανύπαρκτο")["commonality_in_data"], 0.0)

    def test_pos1_reflects_the_previous_token(self):
        values = self.raw("σου", output_tokens=["γεια"])
        self.assertGreater(values["common_pos_1"], 0.0)
        other = self.raw("φίλε", output_tokens=["γεια"])
        self.assertGreater(values["common_pos_1"], other["common_pos_1"])

    def test_pos2_and_pos3_use_earlier_positions(self):
        values = self.raw("κόσμε", output_tokens=["γεια", "σου"])
        self.assertGreater(values["common_pos_2"], 0.0)
        deeper = self.raw("γεια", output_tokens=["γεια", "σου", "κόσμε"])
        self.assertGreater(deeper["common_pos_3"], 0.0)

    def test_lengths(self):
        values = self.raw("κόσμε", input_tokens=["a", "b"], output_tokens=["c"])
        self.assertEqual(values["length_token"], 5.0)
        self.assertEqual(values["length_input"], 2.0)
        self.assertEqual(values["length_output"], 1.0)

    def test_repeats_are_counted_separately(self):
        values = self.raw("γεια", input_tokens=["γεια", "γεια"],
                          output_tokens=["γεια"])
        self.assertEqual(values["repeats_in_input"], 2.0)
        self.assertEqual(values["repeats_in_output"], 1.0)
        self.assertEqual(values["repeats_in_window"], 1.0)

    def test_input_class_bit_is_passed_through(self):
        self.assertEqual(self.raw("γεια", input_class_bit=1)["input_class_bit"], 1.0)
        self.assertEqual(self.raw("γεια", input_class_bit=0)["input_class_bit"], 0.0)

    def test_previous_predicted_class_bit_is_a_separate_feature(self):
        values = self.raw("γεια", input_class_bit=0,
                          previous_predicted_class_bit=1)
        self.assertEqual(values["input_class_bit"], 0.0)
        self.assertEqual(values["previous_predicted_class_bit"], 1.0)

    def test_error_probability_is_an_ema_not_a_future_value(self):
        before = self.raw("γεια")["error_probability"]
        self.memory.update_error("γεια", 1.0, self.cfg.error_ema_alpha)
        after = self.raw("γεια")["error_probability"]
        self.assertGreater(after, before)
        self.assertLessEqual(after, 1.0)


class TestOrdFeatures(unittest.TestCase):
    def test_ord_sum_matches_the_specification(self):
        text = "abc"
        expected = sum(ord(c) for c in text)
        self.assertEqual(ord_features(text)[0], float(expected))

    def test_ord_sum_uses_casefold_and_nfc(self):
        self.assertEqual(ord_features("ΑΒΓ")[0], ord_features("αβγ")[0])

    def test_ord_mean(self):
        values = ord_features("abc")
        self.assertAlmostEqual(values[1], values[0] / 3.0)

    def test_weighted_variants_break_anagram_collisions(self):
        left = ord_features("abc")
        right = ord_features("cba")
        self.assertEqual(left[0], right[0])          # ord_sum collides ...
        self.assertNotEqual(left[2], right[2])       # ... weighted does not
        self.assertNotEqual(left[3], right[3])

    def test_moduli(self):
        values = ord_features("κόσμε")
        self.assertEqual(values[3], values[2] % 257)
        self.assertEqual(values[4], values[2] % 263)

    def test_empty_text(self):
        self.assertEqual(ord_features(""), (0.0, 0.0, 0.0, 0.0, 0.0))


class TestNormalisation(FeatureTestBase):
    def test_no_feature_dominates_by_scale(self):
        """ord_sum is ~10^3 and common_pos_1 is <=1; after normalisation both
        must live in the same range."""
        for word in ("γεια", "σου", "κόσμε", "φίλε", "μακροσκελέστατη"):
            self.extractor.build_token_features(word, output_tokens=["γεια"])
        vector = self.extractor.build_token_features("κόσμε", output_tokens=["γεια"])
        self.assertTrue(all(-1.001 <= v <= 1.001 for v in vector),
                        "normalised features escaped [-1, 1]: {0}".format(vector))

    def test_normaliser_is_bounded_memory(self):
        normalizer = RunningNormalizer(4, warmup=2)
        for index in range(10000):
            normalizer.observe([index, -index, 0.0, 1.0])
        self.assertEqual(len(normalizer.mean), 4)
        self.assertEqual(len(normalizer.m2), 4)

    def test_normaliser_roundtrip(self):
        normalizer = RunningNormalizer(3, warmup=2)
        for value in range(50):
            normalizer.observe([value, value * 2.0, -value])
        restored = RunningNormalizer.from_dict(normalizer.to_dict(), warmup=2)
        self.assertEqual(restored.n, normalizer.n)
        self.assertEqual([round(v, 4) for v in restored.transform([10, 20, -10])],
                         [round(v, 4) for v in normalizer.transform([10, 20, -10])])

    def test_frozen_normaliser_stops_learning(self):
        normalizer = RunningNormalizer(2, warmup=2)
        for _ in range(20):
            normalizer.observe([1.0, 2.0])
        normalizer.frozen = True
        count = normalizer.n
        normalizer.observe([100.0, 200.0])
        self.assertEqual(normalizer.n, count)


class TestInputFeatures(FeatureTestBase):
    def raw_input(self, text):
        tokens = self.tokenizer.tokenize_typed(text)
        vector = self.extractor.build_input_features(tokens, text, normalize=False)
        return dict(zip(INPUT_FEATURE_NAMES, vector))

    def test_greek_and_latin_ratios(self):
        greek = self.raw_input("γεια σου κόσμε")
        latin = self.raw_input("hello there world")
        self.assertGreater(greek["input_greek_ratio"], 0.9)
        self.assertGreater(latin["input_latin_ratio"], 0.9)

    def test_question_detection_handles_greek_question_mark(self):
        self.assertEqual(self.raw_input("Τι κάνεις;")["input_question"], 1.0)
        self.assertEqual(self.raw_input("What is this?")["input_question"], 1.0)
        self.assertEqual(self.raw_input("Καλησπέρα.")["input_question"], 0.0)

    def test_digit_ratio(self):
        self.assertGreater(self.raw_input("1234 5678")["input_digit_ratio"], 0.9)

    def test_unique_ratio(self):
        self.assertAlmostEqual(self.raw_input("a a a a")["input_unique_ratio"], 0.25)


if __name__ == "__main__":
    unittest.main()
