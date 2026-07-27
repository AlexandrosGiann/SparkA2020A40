# -*- coding: utf-8 -*-
"""The adaptive quadratic scorer -- the only "network" in the student runtime.

    z = sum(w_i * f_i) + sum(q_i * f_i**2) + b
    p = sigmoid(z)

The curvature weights ``q`` carry an L1 penalty applied as a *proximal*
(soft-thresholding) step rather than as a subgradient.  That distinction
matters: subgradient L1 leaves weights hovering at 1e-9, while the proximal
operator drives them to exactly ``0.0``.  A problem that is linearly separable
therefore ends up with a provably linear model, and only genuinely non-linear
problems keep curvature alive.

Pure standard library: no NumPy, no autograd, no tensors.
"""

import math
import random

EPS = 1e-12


def sigmoid(z):
    if z >= 0.0:
        if z > 60.0:
            return 1.0
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    if z < -60.0:
        return 0.0
    ez = math.exp(z)
    return ez / (1.0 + ez)


def softmax(scores, temperature=1.0):
    """Numerically stable softmax over a list of raw scores."""
    if not scores:
        return []
    temperature = max(1e-6, float(temperature))
    scaled = [float(s) / temperature for s in scores]
    top = max(scaled)
    exps = [math.exp(s - top) for s in scaled]
    total = sum(exps)
    if total <= 0.0:
        uniform = 1.0 / len(scores)
        return [uniform] * len(scores)
    return [e / total for e in exps]


class AdaptiveQuadraticNeuron(object):
    """A tiny adaptive unit: 2n + 1 parameters, trained online."""

    __slots__ = ("n", "w", "q", "b", "lr", "lambda_q", "l2",
                 "max_abs_weight", "_gw", "_gq", "_gb", "steps")

    def __init__(self, n_features, learning_rate=0.05, lambda_q=0.01, l2=1e-4,
                 init_scale=0.0, seed=None, max_abs_weight=25.0):
        self.n = int(n_features)
        rng = random.Random(seed)
        if init_scale:
            self.w = [rng.uniform(-init_scale, init_scale) for _ in range(self.n)]
        else:
            self.w = [0.0] * self.n
        self.q = [0.0] * self.n
        self.b = 0.0
        self.lr = float(learning_rate)
        self.lambda_q = float(lambda_q)
        self.l2 = float(l2)
        self.max_abs_weight = float(max_abs_weight)
        # AdaGrad accumulators keep the quadratic terms (whose gradients are
        # squared features) from blowing up relative to the linear terms.
        self._gw = [EPS] * self.n
        self._gq = [EPS] * self.n
        self._gb = EPS
        self.steps = 0

    # -- inference -----------------------------------------------------
    def raw_score(self, features):
        z = self.b
        w = self.w
        q = self.q
        for i in range(self.n):
            f = float(features[i])
            z += w[i] * f + q[i] * f * f
        return z

    def predict_proba(self, features):
        return sigmoid(self.raw_score(features))

    def predict(self, features, threshold=0.5):
        return 1 if self.predict_proba(features) >= threshold else 0

    def confidence(self, features):
        """Distance from the decision boundary, mapped to [0, 1]."""
        return abs(self.predict_proba(features) - 0.5) * 2.0

    # -- learning ------------------------------------------------------
    def train_step(self, features, target, lr=None, sample_weight=1.0):
        """One online logistic step.  Returns the absolute error."""
        lr = self.lr if lr is None else float(lr)
        y = 1.0 if target else 0.0
        p = self.predict_proba(features)
        g = (p - y) * float(sample_weight)
        if g > 5.0:
            g = 5.0
        elif g < -5.0:
            g = -5.0

        w = self.w
        q = self.q
        gw = self._gw
        gq = self._gq
        shrink_cap = self.max_abs_weight

        self._gb += g * g
        self.b -= lr * g / math.sqrt(self._gb)

        for i in range(self.n):
            f = float(features[i])
            f2 = f * f

            grad_w = g * f + self.l2 * w[i]
            gw[i] += grad_w * grad_w
            w[i] -= lr * grad_w / math.sqrt(gw[i])
            if w[i] > shrink_cap:
                w[i] = shrink_cap
            elif w[i] < -shrink_cap:
                w[i] = -shrink_cap

            grad_q = g * f2
            gq[i] += grad_q * grad_q
            step_q = lr / math.sqrt(gq[i])
            value = q[i] - step_q * grad_q
            # Proximal L1: soft-threshold -> exact zeros.
            threshold = step_q * self.lambda_q
            if value > threshold:
                value -= threshold
            elif value < -threshold:
                value += threshold
            else:
                value = 0.0
            if value > shrink_cap:
                value = shrink_cap
            elif value < -shrink_cap:
                value = -shrink_cap
            q[i] = value

        self.steps += 1
        return abs(p - y)

    def fit(self, samples, epochs=1, lr=None, shuffle=True, seed=0):
        """``samples`` is an iterable of ``(features, target)`` pairs."""
        data = list(samples)
        rng = random.Random(seed)
        last_loss = 0.0
        for _ in range(int(epochs)):
            if shuffle:
                rng.shuffle(data)
            total = 0.0
            for features, target in data:
                total += self.train_step(features, target, lr)
            last_loss = total / len(data) if data else 0.0
        return last_loss

    # -- diagnostics ---------------------------------------------------
    def accuracy(self, samples):
        data = list(samples)
        if not data:
            return 0.0
        correct = 0
        for features, target in data:
            if self.predict(features) == (1 if target else 0):
                correct += 1
        return float(correct) / float(len(data))

    def mean_error(self, samples):
        data = list(samples)
        if not data:
            return 0.0
        total = 0.0
        for features, target in data:
            total += abs(self.predict_proba(features) - (1.0 if target else 0.0))
        return total / float(len(data))

    def curvature_magnitude(self):
        """Max |q_i| -- zero means the unit settled on a linear boundary."""
        return max((abs(v) for v in self.q), default=0.0)

    def curvature_l1(self):
        return sum(abs(v) for v in self.q)

    def active_curvature(self, tolerance=1e-9):
        return sum(1 for v in self.q if abs(v) > tolerance)

    def is_linear(self, tolerance=1e-9):
        return self.curvature_magnitude() <= tolerance

    def signature(self):
        """Coarse fingerprint used for merge/similarity decisions."""
        return tuple(self.w) + tuple(self.q) + (self.b,)

    def cosine_similarity(self, other):
        a = self.signature()
        b = other.signature()
        if len(a) != len(b):
            return 0.0
        dot = na = nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= EPS or nb <= EPS:
            return 1.0 if (na <= EPS and nb <= EPS) else 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    # -- copying / persistence ----------------------------------------
    def clone(self):
        other = AdaptiveQuadraticNeuron(
            self.n, self.lr, self.lambda_q, self.l2, 0.0, None, self.max_abs_weight)
        other.copy_from(self)
        return other

    def copy_from(self, other):
        self.n = other.n
        self.w = list(other.w)
        self.q = list(other.q)
        self.b = other.b
        self.lr = other.lr
        self.lambda_q = other.lambda_q
        self.l2 = other.l2
        self.max_abs_weight = other.max_abs_weight
        self._gw = list(other._gw)
        self._gq = list(other._gq)
        self._gb = other._gb
        self.steps = other.steps
        return self

    def to_dict(self, compact=True):
        digits = 5 if compact else 12
        data = {
            "n": self.n,
            "w": [round(v, digits) for v in self.w],
            "q": [round(v, digits) for v in self.q],
            "b": round(self.b, digits),
            "steps": self.steps,
        }
        if not compact:
            data["gw"] = self._gw
            data["gq"] = self._gq
            data["gb"] = self._gb
        return data

    @classmethod
    def from_dict(cls, data, cfg=None):
        n = int(data.get("n", 0))
        neuron = cls(
            n,
            getattr(cfg, "learning_rate", 0.05),
            getattr(cfg, "lambda_q", 0.01),
            getattr(cfg, "l2", 1e-4),
            0.0, None,
            getattr(cfg, "max_abs_weight", 25.0),
        )
        w = list(data.get("w") or [])
        q = list(data.get("q") or [])
        if len(w) == n:
            neuron.w = [float(v) for v in w]
        if len(q) == n:
            neuron.q = [float(v) for v in q]
        neuron.b = float(data.get("b", 0.0))
        neuron.steps = int(data.get("steps", 0))
        if data.get("gw") and len(data["gw"]) == n:
            neuron._gw = [float(v) for v in data["gw"]]
        if data.get("gq") and len(data["gq"]) == n:
            neuron._gq = [float(v) for v in data["gq"]]
        if data.get("gb"):
            neuron._gb = float(data["gb"])
        return neuron

    def __repr__(self):
        return "<AdaptiveQuadraticNeuron n={0} steps={1} maxq={2:.4g}>".format(
            self.n, self.steps, self.curvature_magnitude())
