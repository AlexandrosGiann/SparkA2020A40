# -*- coding: utf-8 -*-
"""A very small contextual-bandit router.

The router decides *which* experts see a given context.  Running every expert on
every candidate token would defeat the entire purpose of the architecture, so
the router returns Top-1 (tiny_android) or Top-2 (desktop_training) only.

The policy is a linear contextual bandit with an optimism bonus
(LinUCB-flavoured but using a diagonal count approximation, so it stays pure
Python and O(n_features) per arm).
"""

import math
import random


class RouterArm(object):
    __slots__ = ("expert_id", "w", "b", "count", "reward_sum", "_g")

    def __init__(self, expert_id, n_features):
        self.expert_id = expert_id
        self.w = [0.0] * n_features
        self.b = 0.0
        self.count = 0
        self.reward_sum = 0.0
        self._g = [1e-12] * n_features

    def value(self, context):
        total = self.b
        for i in range(len(self.w)):
            total += self.w[i] * float(context[i])
        return total

    def mean_reward(self):
        if self.count == 0:
            return 0.0
        return self.reward_sum / float(self.count)

    def to_dict(self):
        return {"id": self.expert_id,
                "w": [round(v, 5) for v in self.w],
                "b": round(self.b, 5),
                "count": self.count,
                "reward_sum": round(self.reward_sum, 5)}

    @classmethod
    def from_dict(cls, data, n_features):
        arm = cls(int(data.get("id", 0)), n_features)
        w = list(data.get("w") or [])
        if len(w) == n_features:
            arm.w = [float(v) for v in w]
        arm.b = float(data.get("b", 0.0))
        arm.count = int(data.get("count", 0))
        arm.reward_sum = float(data.get("reward_sum", 0.0))
        return arm


class Router(object):
    """Selects the Top-k experts for a context and learns from the reward."""

    __slots__ = ("cfg", "n_features", "arms", "total_selections", "_rng")

    def __init__(self, cfg, n_features, expert_ids=()):
        self.cfg = cfg
        self.n_features = int(n_features)
        self.arms = {}
        self.total_selections = 0
        self._rng = random.Random(cfg.seed)
        for expert_id in expert_ids:
            self.register(expert_id)

    # -- arm bookkeeping -----------------------------------------------
    def register(self, expert_id):
        if expert_id not in self.arms:
            self.arms[expert_id] = RouterArm(expert_id, self.n_features)
        return self.arms[expert_id]

    def unregister(self, expert_id):
        return self.arms.pop(expert_id, None)

    def sync(self, expert_ids):
        wanted = set(expert_ids)
        for expert_id in wanted:
            self.register(expert_id)
        for expert_id in list(self.arms):
            if expert_id not in wanted:
                del self.arms[expert_id]
        return self

    # -- selection -----------------------------------------------------
    def _ucb_bonus(self, arm):
        total = max(1, self.total_selections)
        return self.cfg.router_explore * math.sqrt(
            math.log(total + 1.0) / float(arm.count + 1))

    def score_all(self, context, expert_ids=None):
        scores = {}
        ids = self.arms.keys() if expert_ids is None else expert_ids
        for expert_id in ids:
            arm = self.arms.get(expert_id)
            if arm is None:
                arm = self.register(expert_id)
            scores[expert_id] = arm.value(context) + self._ucb_bonus(arm)
        return scores

    def select(self, context, expert_ids=None, k=None):
        """Return the ids of the Top-k experts for this context."""
        k = self.cfg.router_top_k if k is None else int(k)
        scores = self.score_all(context, expert_ids)
        if not scores:
            return []
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        chosen = [expert_id for expert_id, _ in ordered[:max(1, k)]]
        self.total_selections += 1
        for expert_id in chosen:
            self.arms[expert_id].count += 1
        return chosen

    # -- learning ------------------------------------------------------
    def update(self, expert_id, context, reward):
        """Online ridge-free least-squares step towards the observed reward."""
        arm = self.arms.get(expert_id)
        if arm is None:
            arm = self.register(expert_id)
        reward = float(reward)
        arm.reward_sum += reward
        predicted = arm.value(context)
        error = predicted - reward
        if error > 5.0:
            error = 5.0
        elif error < -5.0:
            error = -5.0
        lr = self.cfg.router_lr
        arm.b -= lr * error
        for i in range(self.n_features):
            f = float(context[i])
            grad = error * f
            arm._g[i] += grad * grad
            arm.w[i] -= lr * grad / math.sqrt(arm._g[i])
        return arm.value(context)

    # -- diagnostics ----------------------------------------------------
    def usage_distribution(self):
        total = sum(arm.count for arm in self.arms.values())
        if total == 0:
            return dict((expert_id, 0.0) for expert_id in self.arms)
        return dict((expert_id, arm.count / float(total))
                    for expert_id, arm in self.arms.items())

    def entropy(self):
        """0 = one expert does everything, 1 = perfectly balanced routing."""
        distribution = [p for p in self.usage_distribution().values() if p > 0]
        if len(distribution) <= 1:
            return 0.0
        total = -sum(p * math.log(p) for p in distribution)
        return total / math.log(len(distribution))

    def to_dict(self):
        return {"n_features": self.n_features,
                "total_selections": self.total_selections,
                "arms": [arm.to_dict() for arm in self.arms.values()]}

    @classmethod
    def from_dict(cls, data, cfg, n_features):
        router = cls(cfg, n_features)
        for entry in (data or {}).get("arms") or []:
            arm = RouterArm.from_dict(entry, n_features)
            router.arms[arm.expert_id] = arm
        router.total_selections = int((data or {}).get("total_selections", 0))
        return router
