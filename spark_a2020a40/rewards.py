# -*- coding: utf-8 -*-
"""Composable reward signal for the contextual-bandit / online RL loop.

Every term is a *replaceable component*.  In particular ``relevance`` is a bag
-of-tokens overlap heuristic: it is a cheap proxy, **not** semantic
understanding, and it is registered like any other component precisely so it
can be swapped for something better without touching the training loop.

Rewards are clipped and smoothed; an unclipped reward is the fastest way to
make an online learner diverge.
"""

from .tokenizer import EOS, UNK

COMPONENT_NAMES = (
    "teacher_agreement",
    "class_correctness",
    "user_feedback",
    "relevance",
    "sequence_completion",
    "repetition",
    "invalid_output",
    "excessive_length",
    "uncertainty",
)


def _texts(tokens):
    return [t.text if hasattr(t, "text") else t for t in (tokens or ())]


# -- default components -------------------------------------------------
def teacher_agreement(ctx):
    """Recall of teacher tokens inside the student output, in [0, 1]."""
    teacher = set(_texts(ctx.get("teacher_tokens")))
    student = set(_texts(ctx.get("student_tokens")))
    if not teacher:
        return 0.0
    return float(len(teacher & student)) / float(len(teacher))


def class_correctness(ctx):
    predicted = ctx.get("predicted_class")
    target = ctx.get("target_class")
    if predicted is None or target is None:
        return 0.0
    return 1.0 if int(predicted) == int(target) else -1.0


def user_feedback(ctx):
    value = ctx.get("user_feedback")
    if value is None:
        return 0.0
    value = float(value)
    return 1.0 if value > 0 else (-1.0 if value < 0 else 0.0)


def relevance(ctx):
    """Heuristic overlap between the prompt and the answer.

    Explicitly NOT semantic understanding -- replace via
    ``RewardEngine.register_component('relevance', fn)``.
    """
    prompt = set(_texts(ctx.get("input_tokens")))
    student = _texts(ctx.get("student_tokens"))
    if not prompt or not student:
        return 0.0
    hits = sum(1 for t in student if t in prompt)
    return min(1.0, float(hits) / float(len(student)))


def sequence_completion(ctx):
    student = _texts(ctx.get("student_tokens"))
    if not student:
        return 0.0
    if student[-1] == EOS:
        return 1.0
    limit = ctx.get("max_tokens")
    if limit and len(student) < int(limit):
        return 0.5
    return 0.0


def repetition(ctx):
    student = _texts(ctx.get("student_tokens"))
    if len(student) < 2:
        return 0.0
    unique = len(set(student))
    return 1.0 - (float(unique) / float(len(student)))


def invalid_output(ctx):
    student = _texts(ctx.get("student_tokens"))
    if not student:
        return 1.0
    unknown = sum(1 for t in student if t == UNK or not t.strip())
    return min(1.0, float(unknown) / float(len(student)))


def excessive_length(ctx):
    student = _texts(ctx.get("student_tokens"))
    target = ctx.get("target_length")
    if not student or not target:
        return 0.0
    target = float(target)
    if len(student) <= target:
        return 0.0
    return min(1.0, (len(student) - target) / max(1.0, target))


def uncertainty(ctx):
    confidences = ctx.get("confidences") or ()
    if not confidences:
        return 0.0
    mean = sum(float(c) for c in confidences) / float(len(confidences))
    return max(0.0, min(1.0, 1.0 - mean))


DEFAULT_COMPONENTS = {
    "teacher_agreement": teacher_agreement,
    "class_correctness": class_correctness,
    "user_feedback": user_feedback,
    "relevance": relevance,
    "sequence_completion": sequence_completion,
    "repetition": repetition,
    "invalid_output": invalid_output,
    "excessive_length": excessive_length,
    "uncertainty": uncertainty,
}


class RewardEngine(object):
    """Weighted sum of pluggable components with clipping and EMA tracking."""

    __slots__ = ("cfg", "components", "weights", "reward_ema", "count",
                 "reward_sum", "last_breakdown")

    def __init__(self, cfg, components=None, weights=None):
        self.cfg = cfg
        self.components = dict(DEFAULT_COMPONENTS)
        if components:
            self.components.update(components)
        self.weights = cfg.reward_weights()
        self.weights.setdefault("sequence_completion", 0.3)
        if weights:
            self.weights.update(weights)
        self.reward_ema = 0.0
        self.count = 0
        self.reward_sum = 0.0
        self.last_breakdown = {}

    def register_component(self, name, fn, weight=None):
        self.components[name] = fn
        if weight is not None:
            self.weights[name] = float(weight)
        return self

    def set_weight(self, name, weight):
        self.weights[name] = float(weight)
        return self

    def breakdown(self, ctx):
        out = {}
        for name, fn in self.components.items():
            try:
                out[name] = float(fn(ctx))
            except Exception:
                # A broken custom component must not take the chatbot down.
                out[name] = 0.0
        return out

    def compute(self, ctx):
        parts = self.breakdown(ctx)
        total = 0.0
        for name, value in parts.items():
            total += self.weights.get(name, 0.0) * value
        clip = self.cfg.reward_clip
        if total > clip:
            total = clip
        elif total < -clip:
            total = -clip
        alpha = self.cfg.reward_ema_alpha
        self.reward_ema = alpha * self.reward_ema + (1.0 - alpha) * total
        self.count += 1
        self.reward_sum += total
        self.last_breakdown = parts
        return total

    def average(self):
        if self.count == 0:
            return 0.0
        return self.reward_sum / float(self.count)

    def stats(self):
        return {"count": self.count, "ema": self.reward_ema,
                "average": self.average(), "last": dict(self.last_breakdown)}

    def to_dict(self):
        return {"ema": round(self.reward_ema, 5), "count": self.count,
                "sum": round(self.reward_sum, 5),
                "weights": dict(self.weights)}

    def load_dict(self, data):
        if not data:
            return self
        self.reward_ema = float(data.get("ema", 0.0))
        self.count = int(data.get("count", 0))
        self.reward_sum = float(data.get("sum", 0.0))
        weights = data.get("weights")
        if isinstance(weights, dict):
            self.weights.update(dict((k, float(v)) for k, v in weights.items()))
        return self
