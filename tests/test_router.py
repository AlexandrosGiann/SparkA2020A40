# -*- coding: utf-8 -*-
import random
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config, PROFILE_DESKTOP
from spark_a2020a40.experts import ExpertPool
from spark_a2020a40.router import Router
from spark_a2020a40.student import StudentModel


class TestRouterSelection(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.router = Router(self.cfg, 4, [0, 1, 2])

    def test_top_1_by_default_on_tiny_android(self):
        self.assertEqual(self.cfg.router_top_k, 1)
        self.assertEqual(len(self.router.select([0.1, 0.2, 0.0, 0.0])), 1)

    def test_top_2_on_desktop_profile(self):
        cfg = Config(PROFILE_DESKTOP)
        router = Router(cfg, 4, [0, 1, 2])
        self.assertEqual(len(router.select([0.1, 0.2, 0.0, 0.0])), 2)

    def test_explicit_k(self):
        self.assertEqual(len(self.router.select([0.0] * 4, k=3)), 3)

    def test_selection_is_deterministic_for_the_same_state(self):
        first = Router(self.cfg, 4, [0, 1, 2]).select([0.5] * 4)
        second = Router(self.cfg, 4, [0, 1, 2]).select([0.5] * 4)
        self.assertEqual(first, second)

    def test_sync_adds_and_removes_arms(self):
        self.router.sync([1, 2, 5])
        self.assertEqual(sorted(self.router.arms), [1, 2, 5])

    def test_unknown_expert_is_registered_on_demand(self):
        scores = self.router.score_all([0.0] * 4, expert_ids=[9])
        self.assertIn(9, scores)


class TestRouterLearning(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.router = Router(self.cfg, 4, [0, 1])

    def test_it_learns_a_context_dependent_policy(self):
        rng = random.Random(0)
        for _ in range(1500):
            sign = rng.choice([-1.0, 1.0])
            context = [sign, 0.1, 0.0, 0.0]
            for expert_id in self.router.select(context):
                good = (sign > 0 and expert_id == 0) or (sign < 0 and expert_id == 1)
                self.router.update(expert_id, context, 1.0 if good else -1.0)
        self.assertEqual(self.router.select([1.0, 0.1, 0.0, 0.0]), [0])
        self.assertEqual(self.router.select([-1.0, 0.1, 0.0, 0.0]), [1])

    def test_reward_moves_the_arm_value(self):
        context = [1.0, 0.0, 0.0, 0.0]
        before = self.router.arms[0].value(context)
        for _ in range(200):
            self.router.update(0, context, 1.0)
        after = self.router.arms[0].value(context)
        self.assertGreater(after, before)

    def test_exploration_reaches_every_arm(self):
        router = Router(self.cfg, 2, [0, 1, 2])
        for _ in range(300):
            chosen = router.select([0.0, 0.0])
            router.update(chosen[0], [0.0, 0.0], 0.0)
        distribution = router.usage_distribution()
        self.assertTrue(all(value > 0.0 for value in distribution.values()),
                        distribution)

    def test_usage_distribution_sums_to_one(self):
        for _ in range(20):
            self.router.select([0.2, 0.2, 0.2, 0.2])
        self.assertAlmostEqual(sum(self.router.usage_distribution().values()),
                               1.0, places=9)

    def test_entropy_bounds(self):
        router = Router(self.cfg, 2, [0, 1])
        self.assertEqual(router.entropy(), 0.0)
        for _ in range(100):
            router.select([0.0, 0.0], k=2)
        self.assertAlmostEqual(router.entropy(), 1.0, places=6)

    def test_roundtrip(self):
        for _ in range(50):
            self.router.update(0, [0.3, 0.1, 0.0, 0.2], 0.5)
        restored = Router.from_dict(self.router.to_dict(), self.cfg, 4)
        self.assertAlmostEqual(restored.arms[0].value([0.3, 0.1, 0.0, 0.2]),
                               self.router.arms[0].value([0.3, 0.1, 0.0, 0.2]),
                               places=4)


class TestRouterLimitsWork(unittest.TestCase):
    """The point of routing is *not* running every expert on every token."""

    def setUp(self):
        self.cfg = Config()
        self.student = StudentModel(self.cfg)
        while len(self.student.pool) < 4:
            self.student.pool.spawn(force=True)
        self.student.router.sync(self.student.pool.ids())

    def test_only_top_k_experts_are_consulted(self):
        self.student.memory.observe_sequence(
            self.student.tokenizer.tokenize_typed("γεια σου κόσμε"))
        result = self.student.generate("γεια", max_tokens=5)
        self.assertLessEqual(len(result["experts"]), self.cfg.router_top_k)
        self.assertLess(len(result["experts"]), len(self.student.pool))

    def test_unrouted_experts_are_not_touched(self):
        pool = self.student.pool
        before = dict((expert.unique_id, expert.usage_count) for expert in pool)
        result = self.student.generate("γεια", max_tokens=5)
        touched = set(result["experts"])
        for expert in pool:
            if expert.unique_id in touched:
                self.assertGreater(expert.usage_count, before[expert.unique_id])
            else:
                self.assertEqual(expert.usage_count, before[expert.unique_id])


if __name__ == "__main__":
    unittest.main()
