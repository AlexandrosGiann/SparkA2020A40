# -*- coding: utf-8 -*-
"""Generation, rewards and the end-to-end turn."""
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config, PROFILE_DESKTOP
from spark_a2020a40.rewards import RewardEngine
from spark_a2020a40.student import StudentModel
from spark_a2020a40.teacher import OfflineTeacher
from spark_a2020a40.tokenizer import EOS, UNK
from spark_a2020a40.trainer import TrainingCoordinator

CORPUS = [
    "γεια σου κόσμε πώς είσαι σήμερα",
    "καλησπέρα φίλε τι κάνεις σήμερα",
    "hello world how are you today",
    "the student model runs on the phone",
]


def trained_student(cfg=None, passes=3):
    cfg = cfg or Config()
    student = StudentModel(cfg)
    for _ in range(passes):
        for sentence in CORPUS:
            student.memory.observe_sequence(
                student.tokenizer.tokenize_typed(sentence), class_bit=0)
    return student


class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        self.student = trained_student()

    def test_candidates_are_bounded(self):
        candidates = self.student.candidate_tokens(["γεια"], [])
        self.assertLessEqual(len(candidates), self.student.cfg.max_candidates)

    def test_candidates_prefer_real_successors(self):
        candidates = self.student.candidate_tokens(["γεια"], [])
        self.assertIn("σου", candidates)

    def test_eos_is_offered_once_there_is_output(self):
        """An answer may end -- but it may not be empty, so <eos> is withheld
        at step 0."""
        self.assertNotIn(EOS, self.student.candidate_tokens(["γεια"], []))
        self.assertIn(EOS, self.student.candidate_tokens(["γεια"], ["γεια", "σου"]))

    def test_unknown_placeholder_is_never_offered(self):
        self.assertNotIn(UNK, self.student.candidate_tokens(["γεια"], []))

    def test_empty_memory_still_offers_something(self):
        student = StudentModel(Config())
        self.assertTrue(student.candidate_tokens(["άγνωστο"], []))


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.student = trained_student(self.cfg)

    def test_generate_returns_the_expected_shape(self):
        result = self.student.generate("γεια σου")
        for key in ("text", "tokens", "class_bit", "experts", "confidences"):
            self.assertIn(key, result)
        self.assertIsInstance(result["text"], str)

    def test_length_is_bounded(self):
        result = self.student.generate("γεια σου", max_tokens=7)
        self.assertLessEqual(len(result["tokens"]), 7)

    def test_greedy_generation_is_deterministic(self):
        first = trained_student(Config()).generate("γεια σου", greedy=True)
        second = trained_student(Config()).generate("γεια σου", greedy=True)
        self.assertEqual(first["tokens"], second["tokens"])

    def test_sampling_is_reproducible_from_the_seed(self):
        first = trained_student(Config()).generate("γεια σου")
        second = trained_student(Config()).generate("γεια σου")
        self.assertEqual(first["tokens"], second["tokens"])

    def test_eos_is_not_emitted_as_text(self):
        result = self.student.generate("γεια σου", max_tokens=20)
        self.assertNotIn(EOS, result["tokens"])

    def test_answer_returns_a_non_empty_string(self):
        self.assertTrue(self.student.answer("γεια σου").strip())

    def test_generation_works_for_greek_and_english(self):
        self.assertIsInstance(self.student.answer("hello world"), str)
        self.assertIsInstance(self.student.answer("καλησπέρα φίλε"), str)

    def test_repetition_is_discouraged(self):
        result = self.student.generate("γεια σου", max_tokens=16, greedy=True)
        tokens = result["tokens"]
        if len(tokens) >= 4:
            longest = 1
            current = 1
            for index in range(1, len(tokens)):
                current = current + 1 if tokens[index] == tokens[index - 1] else 1
                longest = max(longest, current)
            self.assertLess(longest, 4, tokens)

    def test_only_routed_experts_are_reported(self):
        result = self.student.generate("γεια σου")
        self.assertLessEqual(len(result["experts"]), self.cfg.router_top_k)

    def test_desktop_profile_generates_too(self):
        cfg = Config(PROFILE_DESKTOP)
        student = trained_student(cfg)
        self.assertIsInstance(student.answer("hello world"), str)


class TestBinaryClassApi(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.student = trained_student(self.cfg)

    def test_predict_input_class_returns_probability_and_bit(self):
        features = self.student.features.build_input_features(["γεια"])
        probability, bit = self.student.predict_input_class(features)
        self.assertTrue(0.0 <= probability <= 1.0)
        self.assertIn(bit, (0, 1))

    def test_train_class_predictor_reduces_the_error(self):
        features = self.student.features.build_input_features(
            self.student.tokenizer.tokenize_typed("Τι κάνεις;"), "Τι κάνεις;")
        first = self.student.train_class_predictor(features, 1)
        for _ in range(200):
            self.student.train_class_predictor(features, 1)
        last = self.student.train_class_predictor(features, 1)
        self.assertLess(last, first)
        self.assertEqual(self.student.predict_input_class(features)[1], 1)

    def test_weak_label_recognises_questions(self):
        self.assertEqual(self.student.derive_target_class_bit("Τι κάνεις;"), 1)
        self.assertEqual(self.student.derive_target_class_bit("How are you?"), 1)
        self.assertEqual(self.student.derive_target_class_bit("Καλησπέρα."), 0)

    def test_previous_prediction_is_carried_forward(self):
        self.student.generate("Τι κάνεις;")
        self.assertIn(self.student.previous_predicted_class_bit, (0, 1))


class TestRewards(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.engine = RewardEngine(self.cfg)

    def test_teacher_agreement_rewards_overlap(self):
        good = self.engine.breakdown(
            {"teacher_tokens": ["a", "b"], "student_tokens": ["a", "b"]})
        bad = self.engine.breakdown(
            {"teacher_tokens": ["a", "b"], "student_tokens": ["x", "y"]})
        self.assertEqual(good["teacher_agreement"], 1.0)
        self.assertEqual(bad["teacher_agreement"], 0.0)

    def test_repetition_is_penalised(self):
        self.assertGreater(
            self.engine.breakdown({"student_tokens": ["a", "a", "a"]})["repetition"],
            0.5)
        self.assertEqual(
            self.engine.breakdown({"student_tokens": ["a", "b", "c"]})["repetition"],
            0.0)

    def test_empty_output_is_invalid(self):
        self.assertEqual(self.engine.breakdown({"student_tokens": []})["invalid_output"],
                         1.0)

    def test_excessive_length_is_penalised(self):
        parts = self.engine.breakdown(
            {"student_tokens": ["a"] * 20, "target_length": 5})
        self.assertGreater(parts["excessive_length"], 0.0)

    def test_class_correctness(self):
        self.assertEqual(self.engine.breakdown(
            {"predicted_class": 1, "target_class": 1})["class_correctness"], 1.0)
        self.assertEqual(self.engine.breakdown(
            {"predicted_class": 0, "target_class": 1})["class_correctness"], -1.0)

    def test_user_feedback(self):
        self.assertEqual(self.engine.breakdown({"user_feedback": 1})["user_feedback"], 1.0)
        self.assertEqual(self.engine.breakdown({"user_feedback": -1})["user_feedback"], -1.0)
        self.assertEqual(self.engine.breakdown({})["user_feedback"], 0.0)

    def test_reward_is_clipped(self):
        reward = self.engine.compute({
            "teacher_tokens": ["a"], "student_tokens": ["a"],
            "input_tokens": ["a"], "predicted_class": 1, "target_class": 1,
            "user_feedback": 1, "confidences": [1.0]})
        self.assertLessEqual(reward, self.cfg.reward_clip)
        worst = self.engine.compute({
            "student_tokens": [], "predicted_class": 0, "target_class": 1,
            "user_feedback": -1})
        self.assertGreaterEqual(worst, -self.cfg.reward_clip)

    def test_moving_average_is_tracked(self):
        for _ in range(10):
            self.engine.compute({"user_feedback": 1,
                                 "student_tokens": ["a", "b", "c"]})
        self.assertGreater(self.engine.reward_ema, 0.0)
        self.assertGreater(self.engine.average(), 0.0)
        self.assertEqual(self.engine.count, 10)

    def test_an_empty_answer_with_a_thumbs_up_is_still_not_rewarded(self):
        """invalid_output cancels the user bonus -- praise cannot buy silence."""
        self.assertLessEqual(
            self.engine.compute({"user_feedback": 1, "student_tokens": []}), 0.0)

    def test_components_are_replaceable(self):
        self.engine.register_component("relevance", lambda ctx: 0.5, weight=2.0)
        self.assertEqual(self.engine.breakdown({})["relevance"], 0.5)
        self.assertEqual(self.engine.weights["relevance"], 2.0)

    def test_a_broken_component_cannot_crash_the_bot(self):
        def explode(ctx):
            raise RuntimeError("boom")
        self.engine.register_component("relevance", explode)
        self.assertEqual(self.engine.breakdown({})["relevance"], 0.0)


class TestEndToEnd(unittest.TestCase):
    def test_learning_from_a_scripted_teacher_improves_agreement(self):
        cfg = Config()
        student = StudentModel(cfg)
        coordinator = TrainingCoordinator(cfg, student, OfflineTeacher(cfg))
        lesson = "γεια σου κόσμε πώς είσαι σήμερα"
        first = coordinator.process_turn("γεια", teacher_text=lesson)
        for _ in range(25):
            coordinator.process_turn("γεια", teacher_text=lesson)
        last = coordinator.process_turn("γεια", teacher_text=lesson)
        self.assertGreaterEqual(
            last["reward_breakdown"]["teacher_agreement"],
            first["reward_breakdown"]["teacher_agreement"])
        self.assertGreater(len(student.memory), 5)

    def test_stats_expose_everything_the_cli_needs(self):
        cfg = Config()
        coordinator = TrainingCoordinator(cfg, StudentModel(cfg), OfflineTeacher(cfg))
        coordinator.process_turn("γεια σου")
        stats = coordinator.stats()
        for key in ("tokens", "relations", "experts", "active_experts",
                    "frozen_experts", "avg_reward", "avg_error",
                    "router_usage", "turns"):
            self.assertIn(key, stats)

    def test_many_turns_stay_bounded(self):
        cfg = Config()
        cfg.max_tokens = 120
        student = StudentModel(cfg)
        coordinator = TrainingCoordinator(cfg, student, OfflineTeacher(cfg))
        for index in range(60):
            coordinator.process_turn(
                "λέξη{0} και κάτι ακόμα".format(index),
                teacher_text="απάντηση{0} με πολλές λέξεις εδώ".format(index))
        self.assertLessEqual(len(student.memory), cfg.max_tokens)
        self.assertLessEqual(len(student.pool), cfg.max_experts)
        for expert in student.pool:
            self.assertLessEqual(len(expert.replay), cfg.expert_replay_size)


if __name__ == "__main__":
    unittest.main()
