# -*- coding: utf-8 -*-
"""Selective retraining: the gate, the rollback, and the lifecycle policies."""
import random
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config
from spark_a2020a40.experts import Expert, ExpertPool, should_retrain
from spark_a2020a40.student import StudentModel
from spark_a2020a40.teacher import OfflineTeacher
from spark_a2020a40.trainer import TrainingCoordinator


def fill(expert, count=None, seed=0, separable=True):
    """Put learnable samples into an expert's replay buffer."""
    count = count or expert.cfg.retrain_min_samples * 2
    rng = random.Random(seed)
    for index in range(count):
        target = index % 2
        if separable:
            features = [1.0 if target else -1.0] * expert.neuron.n
        else:
            features = [rng.uniform(-1.0, 1.0) for _ in range(expert.neuron.n)]
        expert.remember(features, target)
    return expert


class TestShouldRetrainGate(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.expert = Expert(0, self.cfg, 4)

    def make_ready(self):
        fill(self.expert)
        self.expert.steps_since_update = self.cfg.retrain_cooldown + 1
        # Neutral state: no trigger fires.
        self.expert.error_ema = 0.0
        self.expert.reward_ema = 1.0
        self.expert.confidence_ema = 1.0
        self.expert.class_failures = [0, 0]
        return self.expert

    def test_not_enough_samples(self):
        self.expert.steps_since_update = 10 ** 6
        self.expert.error_ema = 1.0
        self.assertFalse(should_retrain(self.expert))

    def test_cooldown_not_elapsed(self):
        fill(self.expert)
        self.expert.steps_since_update = 0
        self.expert.error_ema = 1.0
        self.assertFalse(should_retrain(self.expert))

    def test_frozen_expert_is_never_retrained(self):
        self.make_ready()
        self.expert.error_ema = 1.0
        self.expert.frozen = True
        self.assertFalse(should_retrain(self.expert))

    def test_healthy_expert_is_left_alone(self):
        self.make_ready()
        self.assertFalse(should_retrain(self.expert))

    def test_trigger_high_error(self):
        self.make_ready().error_ema = self.cfg.error_threshold + 0.01
        self.assertTrue(should_retrain(self.expert))

    def test_trigger_low_reward(self):
        self.make_ready().reward_ema = self.cfg.reward_threshold - 0.01
        self.assertTrue(should_retrain(self.expert))

    def test_trigger_low_confidence(self):
        self.make_ready().confidence_ema = self.cfg.confidence_threshold - 0.01
        self.assertTrue(should_retrain(self.expert))

    def test_trigger_teacher_disagreement(self):
        self.make_ready()
        self.assertTrue(should_retrain(self.expert, {"teacher_disagreement": 0.9}))

    def test_trigger_distribution_shift(self):
        self.make_ready()
        self.assertTrue(should_retrain(self.expert, {"distribution_shift": True}))

    def test_trigger_drift(self):
        self.make_ready()
        self.assertTrue(should_retrain(
            self.expert, {"drift": 0.9, "drift_threshold": 0.5}))

    def test_trigger_repeated_class_failure(self):
        self.make_ready()
        self.expert.class_failures = [0, 7]
        self.assertTrue(should_retrain(self.expert, {"class_failure_limit": 5}))


class TestSelectiveRetrainingOnlyTouchesQualifyingExperts(unittest.TestCase):
    """Specification test 5: retraining must not update everybody."""

    def setUp(self):
        self.cfg = Config()
        self.pool = ExpertPool(self.cfg, 4, initial=1)
        while len(self.pool) < 4:
            self.pool.spawn(force=True)
        for index, expert in enumerate(sorted(self.pool, key=lambda e: e.unique_id)):
            fill(expert, seed=index)
            expert.steps_since_update = self.cfg.retrain_cooldown + 5
            expert.error_ema = 0.0
            expert.reward_ema = 1.0
            expert.confidence_ema = 1.0
        # Only expert #1 is in trouble.
        self.pool.get(1).error_ema = 0.99

    def test_only_the_failing_expert_is_retrained(self):
        reports = self.pool.retrain_due()
        self.assertEqual([report["expert"] for report in reports], [1])

    def test_the_others_keep_their_weights(self):
        snapshot = dict((expert.unique_id, list(expert.neuron.w))
                        for expert in self.pool)
        self.pool.retrain_due()
        for expert in self.pool:
            if expert.unique_id == 1:
                self.assertNotEqual(expert.neuron.w, snapshot[expert.unique_id])
            else:
                self.assertEqual(expert.neuron.w, snapshot[expert.unique_id])

    def test_last_training_step_only_moves_for_the_retrained_expert(self):
        self.pool.retrain_due(step=42)
        self.assertEqual(self.pool.get(1).last_training_step, 42)
        for expert_id in (0, 2, 3):
            self.assertEqual(self.pool.get(expert_id).last_training_step, 0)

    def test_frozen_experts_are_skipped_even_when_failing(self):
        for expert in self.pool:
            expert.error_ema = 0.99
        self.pool.get(2).frozen = True
        retrained = [report["expert"] for report in self.pool.retrain_due()]
        self.assertNotIn(2, retrained)
        self.assertIn(1, retrained)


class TestRetrainingSafety(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.expert = Expert(0, self.cfg, 4)

    def test_useful_update_is_applied(self):
        fill(self.expert, count=40, separable=True)
        report = self.expert.retrain(step=5)
        self.assertTrue(report["applied"], report)
        self.assertEqual(report["reason"], "ok")
        self.assertLess(report["after"], report["before"])

    def test_harmful_update_is_rolled_back(self):
        # Random labels: the model can only memorise, so validation degrades.
        fill(self.expert, count=60, separable=False, seed=3)
        before = list(self.expert.neuron.w)
        report = self.expert.retrain(step=5)
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "validation_regression")
        self.assertEqual(self.expert.neuron.w, before)

    def test_checkpoint_and_manual_rollback(self):
        self.expert.checkpoint()
        self.assertTrue(self.expert.has_checkpoint())
        original = list(self.expert.neuron.w)
        for _ in range(50):
            self.expert.train_online([1.0] * 4, 1)
        self.assertNotEqual(self.expert.neuron.w, original)
        self.assertTrue(self.expert.rollback())
        self.assertEqual([round(v, 6) for v in self.expert.neuron.w],
                         [round(v, 6) for v in original])

    def test_empty_buffer_is_a_no_op(self):
        report = self.expert.retrain(step=1)
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "empty_buffer")

    def test_replay_buffer_is_bounded(self):
        for index in range(self.cfg.expert_replay_size * 4):
            self.expert.remember([float(index)] * 4, index % 2)
        self.assertEqual(len(self.expert.replay), self.cfg.expert_replay_size)

    def test_freeze_requires_stability_and_quality(self):
        self.expert.error_ema = 0.01
        self.expert.reward_ema = 1.0
        self.expert.last_training_step = 0
        self.assertFalse(self.expert.maybe_freeze(10))
        self.assertTrue(self.expert.maybe_freeze(self.cfg.freeze_stability_steps + 1))
        self.assertTrue(self.expert.frozen)

    def test_frozen_expert_ignores_online_updates(self):
        self.expert.frozen = True
        before = list(self.expert.neuron.w)
        self.expert.train_online([1.0] * 4, 1)
        self.assertEqual(self.expert.neuron.w, before)


class TestPoolLifecycle(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.pool = ExpertPool(self.cfg, 4, initial=1)

    def test_expert_count_is_capped(self):
        for _ in range(self.cfg.max_experts * 3):
            self.pool.spawn(force=True)
        self.assertLessEqual(len(self.pool), self.cfg.max_experts)

    def test_spawn_needs_low_confidence_and_repeated_failures(self):
        self.assertFalse(self.pool.should_spawn(0.9, 10 ** 6))
        self.assertFalse(self.pool.should_spawn(0.0, 0))
        self.assertTrue(self.pool.should_spawn(0.0, self.cfg.spawn_failures))

    def test_runaway_spawning_is_prevented(self):
        for _ in range(1000):
            self.pool.note_failure()
            self.pool.maybe_spawn(0.0)
        self.assertLessEqual(len(self.pool), self.cfg.max_experts)

    def test_success_reduces_spawn_pressure(self):
        for _ in range(5):
            self.pool.note_failure()
        for _ in range(5):
            self.pool.note_success()
        self.assertEqual(self.pool.spawn_evidence, 0)

    def test_identical_experts_are_merged(self):
        self.pool.spawn(force=True)
        first, second = self.pool.get(0), self.pool.get(1)
        for expert in (first, second):
            for _ in range(30):
                expert.train_online([1.0, 0.5, -0.5, 0.0], 1)
                expert.train_online([-1.0, -0.5, 0.5, 0.0], 0)
            expert._update_signature([1.0, 0.5, -0.5, 0.0])
            expert.usage_count = 20
        merged = self.pool.maybe_merge()
        self.assertIsNotNone(merged)
        self.assertEqual(len(self.pool), 1)

    def test_different_experts_are_not_merged(self):
        self.pool.spawn(force=True)
        first, second = self.pool.get(0), self.pool.get(1)
        for _ in range(40):
            first.train_online([1.0, 0.0, 0.0, 0.0], 1)
            second.train_online([0.0, 0.0, 0.0, 1.0], 0)
        first._update_signature([1.0, 0.0, 0.0, 0.0])
        second._update_signature([0.0, 0.0, 0.0, 1.0])
        self.assertIsNone(self.pool.maybe_merge())
        self.assertEqual(len(self.pool), 2)

    def test_pruning_keeps_a_checkpoint_and_can_restore(self):
        self.pool.spawn(force=True)
        self.pool.step = self.cfg.retrain_cooldown * 5
        good, bad = self.pool.get(0), self.pool.get(1)
        good.reward_ema = 0.9
        good.usage_count = 100
        bad.reward_ema = -0.9
        bad.usage_count = 1
        good.signature = [1.0, 0.0, 0.0, 0.0]
        bad.signature = [0.99, 0.05, 0.0, 0.0]
        removed = self.pool.prune()
        self.assertEqual(removed, 1)
        self.assertNotIn(1, self.pool)
        self.assertIsNotNone(self.pool.restore(1))
        self.assertIn(1, self.pool)

    def test_pruning_refuses_when_nobody_covers_the_region(self):
        self.pool.spawn(force=True)
        self.pool.step = self.cfg.retrain_cooldown * 5
        self.pool.get(0).signature = [1.0, 0.0, 0.0, 0.0]
        self.pool.get(1).signature = [0.0, 0.0, 0.0, 1.0]
        self.pool.get(1).reward_ema = -0.9
        self.assertIsNone(self.pool.prune())

    def test_minimum_pool_size_is_respected(self):
        self.pool.get(0).reward_ema = -1.0
        self.pool.get(0).usage_count = 0
        self.assertIsNone(self.pool.prune())
        self.assertGreaterEqual(len(self.pool), self.cfg.min_experts)


class TestTrainerRetrainsSelectively(unittest.TestCase):
    def test_a_turn_does_not_train_every_expert(self):
        cfg = Config()
        student = StudentModel(cfg)
        while len(student.pool) < 4:
            student.pool.spawn(force=True)
        student.router.sync(student.pool.ids())
        coordinator = TrainingCoordinator(cfg, student, OfflineTeacher(cfg))
        report = coordinator.process_turn(
            "γεια σου", teacher_text="γεια σου κόσμε")
        touched = set(report["distill"]["experts"])
        self.assertLessEqual(len(touched), cfg.router_top_k)
        self.assertLess(len(touched), len(student.pool))
        for expert in student.pool:
            if expert.unique_id not in touched:
                self.assertEqual(expert.neuron.steps, 0)


if __name__ == "__main__":
    unittest.main()
