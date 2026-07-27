# -*- coding: utf-8 -*-
"""Regression tests mandated by the specification for the adaptive neuron.

1. circle (inside/outside)      -> curvature MUST activate
2. linearly separable points    -> curvature MUST stay at (or near) zero
3. the six-point parallel-lines dataset -> 100% accuracy, q == 0, boundary x-y=0
4. the L1 penalty must not destroy genuinely required non-linearity
"""
import random
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.adaptive_neuron import AdaptiveQuadraticNeuron, sigmoid, softmax

# The exact dataset from the specification: two parallel lines either side of
# the y = x diagonal.
SPEC_DATASET = [
    [1, 2, 0],
    [2, 3, 0],
    [3, 4, 0],
    [4, 3, 1],
    [3, 2, 1],
    [2, 1, 1],
]


def spec_samples():
    return [([row[0], row[1]], row[2]) for row in SPEC_DATASET]


def circle_samples(count=400, radius_squared=2.25, seed=7, margin=0.15):
    rng = random.Random(seed)
    samples = []
    while len(samples) < count:
        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-3.0, 3.0)
        distance = x * x + y * y
        if abs(distance - radius_squared) < margin:
            continue  # keep a clean margin around the boundary
        samples.append(([x, y], 1 if distance < radius_squared else 0))
    return samples


def linear_samples(count=300, seed=11):
    rng = random.Random(seed)
    samples = []
    while len(samples) < count:
        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-3.0, 3.0)
        if abs(x - y) < 0.25:
            continue
        samples.append(([x, y], 1 if x > y else 0))
    return samples


class TestCase1Circle(unittest.TestCase):
    """Inside/outside a circle is not linearly separable."""

    def test_curvature_activates(self):
        samples = circle_samples()
        neuron = AdaptiveQuadraticNeuron(2, learning_rate=0.5, lambda_q=0.05, l2=0.0)
        neuron.fit(samples, epochs=300, seed=2)
        self.assertGreater(neuron.accuracy(samples), 0.95)
        self.assertGreater(neuron.curvature_magnitude(), 0.1)
        self.assertEqual(neuron.active_curvature(), 2)

    def test_curvature_signs_form_a_circle(self):
        samples = circle_samples()
        neuron = AdaptiveQuadraticNeuron(2, learning_rate=0.5, lambda_q=0.05, l2=0.0)
        neuron.fit(samples, epochs=300, seed=2)
        # Both quadratic weights negative => -(x^2 + y^2) + c > 0 inside.
        self.assertLess(neuron.q[0], 0.0)
        self.assertLess(neuron.q[1], 0.0)
        ratio = neuron.q[0] / neuron.q[1]
        self.assertAlmostEqual(ratio, 1.0, delta=0.35)

    def test_a_purely_linear_unit_cannot_solve_it(self):
        samples = circle_samples()
        linear = AdaptiveQuadraticNeuron(2, learning_rate=0.5, lambda_q=1e9, l2=0.0)
        linear.fit(samples, epochs=300, seed=2)
        self.assertEqual(linear.curvature_magnitude(), 0.0)
        self.assertLess(linear.accuracy(samples), 0.85)


class TestCase2LinearlySeparable(unittest.TestCase):
    def test_curvature_stays_small(self):
        samples = linear_samples()
        neuron = AdaptiveQuadraticNeuron(2, learning_rate=0.5, lambda_q=0.05, l2=0.0)
        neuron.fit(samples, epochs=200, seed=3)
        self.assertGreater(neuron.accuracy(samples), 0.98)
        self.assertLess(neuron.curvature_magnitude(), 0.05)


class TestCase3SpecDataset(unittest.TestCase):
    """The dataset named explicitly in the specification."""

    def setUp(self):
        self.samples = spec_samples()
        self.neuron = AdaptiveQuadraticNeuron(2, learning_rate=0.5,
                                              lambda_q=0.05, l2=0.0)
        self.neuron.fit(self.samples, epochs=3000, seed=1)

    def test_100_percent_training_accuracy(self):
        self.assertEqual(self.neuron.accuracy(self.samples), 1.0)

    def test_curvature_is_exactly_zero(self):
        # Proximal L1 gives real zeros, not 1e-9 leftovers.
        self.assertEqual(self.neuron.curvature_magnitude(), 0.0)
        self.assertEqual(self.neuron.active_curvature(), 0)
        self.assertTrue(self.neuron.is_linear())

    def test_boundary_is_x_minus_y(self):
        wx, wy = self.neuron.w
        self.assertGreater(wx, 0.0)
        self.assertLess(wy, 0.0)
        # w = (a, -a) up to scale => the boundary is x - y = 0.
        self.assertAlmostEqual(wx / (-wy), 1.0, delta=0.1)
        self.assertLess(abs(self.neuron.b), 0.5 * abs(wx))

    def test_classifies_the_diagonal_correctly(self):
        # Points on either side of y = x, unseen during training.
        self.assertEqual(self.neuron.predict([5.0, 1.0]), 1)
        self.assertEqual(self.neuron.predict([1.0, 5.0]), 0)
        self.assertEqual(self.neuron.predict([10.0, 9.0]), 1)
        self.assertEqual(self.neuron.predict([9.0, 10.0]), 0)

    def test_it_is_not_a_circle(self):
        """A circular boundary would misclassify far-away points."""
        far_positive = self.neuron.predict([100.0, 99.0])
        far_negative = self.neuron.predict([99.0, 100.0])
        self.assertEqual(far_positive, 1)
        self.assertEqual(far_negative, 0)


class TestCase4L1DoesNotKillRequiredNonLinearity(unittest.TestCase):
    def test_needed_curvature_survives_the_penalty(self):
        samples = circle_samples()
        without = AdaptiveQuadraticNeuron(2, learning_rate=0.5, lambda_q=0.0, l2=0.0)
        without.fit(samples, epochs=300, seed=2)
        with_l1 = AdaptiveQuadraticNeuron(2, learning_rate=0.5, lambda_q=0.05, l2=0.0)
        with_l1.fit(samples, epochs=300, seed=2)

        self.assertGreater(with_l1.curvature_magnitude(), 0.0)
        self.assertGreater(with_l1.accuracy(samples), 0.95)
        # The penalty may cost a little accuracy, but not much.
        self.assertGreater(with_l1.accuracy(samples), without.accuracy(samples) - 0.05)

    def test_penalty_removes_only_the_unnecessary_curvature(self):
        """A problem that is linear in x but quadratic in y keeps only q_y."""
        rng = random.Random(5)
        samples = []
        while len(samples) < 400:
            x = rng.uniform(-2.0, 2.0)
            y = rng.uniform(-2.0, 2.0)
            score = x - (y * y) + 0.5
            if abs(score) < 0.15:
                continue
            samples.append(([x, y], 1 if score > 0 else 0))
        neuron = AdaptiveQuadraticNeuron(2, learning_rate=0.5, lambda_q=0.05, l2=0.0)
        neuron.fit(samples, epochs=400, seed=6)
        self.assertGreater(neuron.accuracy(samples), 0.93)
        self.assertLess(abs(neuron.q[0]), abs(neuron.q[1]))
        self.assertGreater(abs(neuron.q[1]), 0.2)


class TestNumericHelpers(unittest.TestCase):
    def test_sigmoid_is_saturation_safe(self):
        self.assertEqual(sigmoid(1000.0), 1.0)
        self.assertEqual(sigmoid(-1000.0), 0.0)
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_softmax_sums_to_one(self):
        probabilities = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=9)
        self.assertGreater(probabilities[2], probabilities[0])

    def test_softmax_handles_extreme_scores(self):
        probabilities = softmax([1000.0, -1000.0])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=9)

    def test_clone_is_independent(self):
        neuron = AdaptiveQuadraticNeuron(3)
        neuron.train_step([1.0, 2.0, 3.0], 1)
        clone = neuron.clone()
        clone.train_step([1.0, 2.0, 3.0], 0)
        self.assertNotEqual(neuron.w, clone.w)

    def test_roundtrip_serialisation(self):
        neuron = AdaptiveQuadraticNeuron(4, lambda_q=0.02)
        for _ in range(20):
            neuron.train_step([0.5, -0.5, 1.0, 0.0], 1)
        restored = AdaptiveQuadraticNeuron.from_dict(neuron.to_dict(compact=False))
        self.assertAlmostEqual(restored.predict_proba([0.5, -0.5, 1.0, 0.0]),
                               neuron.predict_proba([0.5, -0.5, 1.0, 0.0]), places=4)

    def test_weights_are_bounded(self):
        neuron = AdaptiveQuadraticNeuron(1, learning_rate=5.0, lambda_q=0.0,
                                         max_abs_weight=3.0)
        for _ in range(5000):
            neuron.train_step([10.0], 1)
        self.assertLessEqual(max(abs(v) for v in neuron.w), 3.0)


if __name__ == "__main__":
    unittest.main()
