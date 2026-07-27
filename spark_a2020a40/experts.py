# -*- coding: utf-8 -*-
"""Many very small experts, plus the policies that create, freeze, merge,
prune and selectively retrain them.

The central rule of this module: **not every expert is updated on every
token.**  ``should_retrain`` is the single gate through which all training
passes, and it is deliberately written as a free function so that it can be
unit-tested without a pool.
"""

import math

from .adaptive_neuron import AdaptiveQuadraticNeuron
from .replay import ReplayBuffer


class Expert(object):
    """A tiny adaptive unit plus the bookkeeping the scheduler needs."""

    __slots__ = ("unique_id", "neuron", "usage_count", "success_count",
                 "error_ema", "reward_ema", "confidence_ema",
                 "last_training_step", "steps_since_update", "signature",
                 "signature_weight", "replay", "frozen", "created_at",
                 "class_failures", "_checkpoint", "cfg")

    def __init__(self, unique_id, cfg, n_features, seed=None):
        self.unique_id = unique_id
        self.cfg = cfg
        self.neuron = AdaptiveQuadraticNeuron(
            n_features, cfg.learning_rate, cfg.lambda_q, cfg.l2,
            cfg.init_scale, seed, cfg.max_abs_weight)
        self.usage_count = 0
        self.success_count = 0
        self.error_ema = 0.5
        self.reward_ema = 0.0
        self.confidence_ema = 0.5
        self.last_training_step = 0
        self.steps_since_update = 0
        self.signature = [0.0] * n_features
        self.signature_weight = 0.0
        self.replay = ReplayBuffer(cfg.expert_replay_size)
        self.frozen = False
        self.created_at = 0
        self.class_failures = [0, 0]
        self._checkpoint = None

    # -- accessors expected by the specification -----------------------
    @property
    def linear_weights(self):
        return self.neuron.w

    @property
    def curvature_weights(self):
        return self.neuron.q

    @property
    def bias(self):
        return self.neuron.b

    @property
    def active(self):
        return not self.frozen

    def success_rate(self):
        if self.usage_count == 0:
            return 0.0
        return float(self.success_count) / float(self.usage_count)

    # -- inference -----------------------------------------------------
    def score(self, features):
        return self.neuron.raw_score(features)

    def predict_proba(self, features):
        return self.neuron.predict_proba(features)

    def confidence(self, features):
        return self.neuron.confidence(features)

    # -- online statistics ---------------------------------------------
    def observe_use(self, features, confidence):
        self.usage_count += 1
        self.steps_since_update += 1
        alpha = self.cfg.reward_ema_alpha
        self.confidence_ema = alpha * self.confidence_ema + (1.0 - alpha) * float(confidence)
        self._update_signature(features)

    def observe_outcome(self, observed_error, reward, target_class=None, success=None):
        ea = self.cfg.error_ema_alpha
        ra = self.cfg.reward_ema_alpha
        self.error_ema = ea * self.error_ema + (1.0 - ea) * float(observed_error)
        self.reward_ema = ra * self.reward_ema + (1.0 - ra) * float(reward)
        if success is None:
            success = observed_error < 0.5
        if success:
            self.success_count += 1
            if target_class in (0, 1):
                self.class_failures[target_class] = 0
        elif target_class in (0, 1):
            self.class_failures[target_class] += 1
        return self.reward_ema

    def _update_signature(self, features):
        """EMA of the contexts this expert is routed to."""
        if not features or len(features) != len(self.signature):
            return
        weight = 0.98 if self.signature_weight > 0 else 0.0
        for i in range(len(self.signature)):
            self.signature[i] = weight * self.signature[i] + (1.0 - weight) * float(features[i])
        self.signature_weight = min(1.0, self.signature_weight + 0.02)

    def signature_similarity(self, other):
        a, b = self.signature, other.signature
        if len(a) != len(b):
            return 0.0
        dot = na = nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def context_distance(self, features):
        if not features or len(features) != len(self.signature):
            return 1.0
        total = 0.0
        for x, y in zip(self.signature, features):
            total += (x - y) * (x - y)
        return math.sqrt(total) / math.sqrt(len(features))

    # -- training ------------------------------------------------------
    def remember(self, features, target, reward=0.0, step=0, tag=""):
        return self.replay.add_example(features, target, reward, step, tag)

    def train_online(self, features, target, step=0):
        """A single gradient step -- used for immediate distillation."""
        if self.frozen:
            return 0.0
        error = self.neuron.train_step(features, target)
        self.last_training_step = step
        self.steps_since_update = 0
        return error

    def checkpoint(self):
        self._checkpoint = {
            "neuron": self.neuron.to_dict(compact=False),
            "error_ema": self.error_ema,
            "reward_ema": self.reward_ema,
            "confidence_ema": self.confidence_ema,
        }
        return self._checkpoint

    def has_checkpoint(self):
        return self._checkpoint is not None

    def rollback(self):
        if self._checkpoint is None:
            return False
        self.neuron = AdaptiveQuadraticNeuron.from_dict(self._checkpoint["neuron"], self.cfg)
        self.error_ema = self._checkpoint["error_ema"]
        self.reward_ema = self._checkpoint["reward_ema"]
        self.confidence_ema = self._checkpoint["confidence_ema"]
        return True

    def retrain(self, step=0, epochs=None):
        """Replay-based retraining with validation gating and rollback.

        Returns a report dict; ``applied`` is False when the update was
        rejected because it made the held-out set worse.
        """
        report = {"expert": self.unique_id, "applied": False, "reason": "",
                  "before": None, "after": None, "samples": len(self.replay)}
        if self.frozen:
            report["reason"] = "frozen"
            return report
        train, validation = self.replay.split(self.cfg.validation_fraction)
        if not train:
            report["reason"] = "empty_buffer"
            return report

        self.checkpoint()
        baseline = None
        if validation:
            baseline = self.neuron.mean_error([s.as_pair() for s in validation])
            report["before"] = baseline

        epochs = self.cfg.retrain_epochs if epochs is None else epochs
        # Mix recent and older samples so retraining does not overfit the tail.
        batch = ReplayBuffer(max(1, len(train)))
        batch.extend(train)
        mixed = batch.mixed_batch(len(train))
        self.neuron.fit([s.as_pair() for s in mixed],
                        epochs=epochs, seed=self.unique_id)

        if validation:
            after = self.neuron.mean_error([s.as_pair() for s in validation])
            report["after"] = after
            if after > baseline + self.cfg.validation_tolerance:
                self.rollback()
                report["reason"] = "validation_regression"
                return report

        self.last_training_step = step
        self.steps_since_update = 0
        report["applied"] = True
        report["reason"] = "ok"
        return report

    def maybe_freeze(self, step):
        """Freeze experts that have been stable and good for a long time."""
        if self.frozen:
            return False
        stable = (step - self.last_training_step) >= self.cfg.freeze_stability_steps
        good = (self.error_ema < self.cfg.error_threshold * 0.5
                and self.reward_ema > self.cfg.reward_threshold)
        if stable and good:
            self.frozen = True
            return True
        return False

    # -- serialisation ---------------------------------------------------
    def to_dict(self, compact=True):
        data = {
            "id": self.unique_id,
            "neuron": self.neuron.to_dict(compact=compact),
            "usage": self.usage_count,
            "success": self.success_count,
            "error_ema": round(self.error_ema, 5),
            "reward_ema": round(self.reward_ema, 5),
            "confidence_ema": round(self.confidence_ema, 5),
            "last_training_step": self.last_training_step,
            "steps_since_update": self.steps_since_update,
            "signature": [round(v, 5) for v in self.signature],
            "signature_weight": round(self.signature_weight, 5),
            "frozen": self.frozen,
            "created_at": self.created_at,
            "class_failures": list(self.class_failures),
        }
        if not compact:
            data["replay"] = self.replay.to_dict()
        return data

    @classmethod
    def from_dict(cls, data, cfg, n_features):
        expert = cls(int(data.get("id", 0)), cfg, n_features)
        expert.neuron = AdaptiveQuadraticNeuron.from_dict(data.get("neuron") or {}, cfg)
        if expert.neuron.n != n_features:
            expert.neuron = AdaptiveQuadraticNeuron(
                n_features, cfg.learning_rate, cfg.lambda_q, cfg.l2,
                cfg.init_scale, None, cfg.max_abs_weight)
        expert.usage_count = int(data.get("usage", 0))
        expert.success_count = int(data.get("success", 0))
        expert.error_ema = float(data.get("error_ema", 0.5))
        expert.reward_ema = float(data.get("reward_ema", 0.0))
        expert.confidence_ema = float(data.get("confidence_ema", 0.5))
        expert.last_training_step = int(data.get("last_training_step", 0))
        expert.steps_since_update = int(data.get("steps_since_update", 0))
        signature = list(data.get("signature") or [])
        if len(signature) == n_features:
            expert.signature = [float(v) for v in signature]
        expert.signature_weight = float(data.get("signature_weight", 0.0))
        expert.frozen = bool(data.get("frozen", False))
        expert.created_at = int(data.get("created_at", 0))
        failures = list(data.get("class_failures") or [0, 0])
        expert.class_failures = [int(failures[0]), int(failures[1])]
        if data.get("replay"):
            expert.replay = ReplayBuffer.from_dict(data["replay"], cfg.expert_replay_size)
        return expert

    def __repr__(self):
        return "<Expert {0} usage={1} err={2:.3f} rew={3:+.3f} {4}>".format(
            self.unique_id, self.usage_count, self.error_ema, self.reward_ema,
            "frozen" if self.frozen else "active")


# ----------------------------------------------------------------------
def should_retrain(expert, context_stats=None):
    """The selective-retraining gate.

    Hard requirements (all must hold):
      * enough samples in the replay buffer,
      * cooldown elapsed,
      * not frozen.

    Then at least one trigger must fire:
      * error EMA above threshold,
      * reward EMA below threshold,
      * confidence EMA below threshold,
      * significant teacher/student disagreement,
      * detected distribution drift,
      * repeated failure on a specific class.
    """
    if expert is None or expert.frozen:
        return False
    cfg = expert.cfg
    stats = context_stats or {}

    if len(expert.replay) < cfg.retrain_min_samples:
        return False
    if expert.steps_since_update < cfg.retrain_cooldown:
        return False

    if expert.error_ema > cfg.error_threshold:
        return True
    if expert.reward_ema < cfg.reward_threshold:
        return True
    if expert.confidence_ema < cfg.confidence_threshold:
        return True
    if float(stats.get("teacher_disagreement", 0.0)) > 0.5:
        return True
    if bool(stats.get("distribution_shift", False)):
        return True
    if float(stats.get("drift", 0.0)) > float(stats.get("drift_threshold", 0.5)):
        return True
    repeated = max(expert.class_failures)
    if repeated >= int(stats.get("class_failure_limit", 5)):
        return True
    return False


class ExpertPool(object):
    """A bounded collection of experts with spawn / merge / prune policies."""

    __slots__ = ("cfg", "n_features", "experts", "_next_id", "spawn_evidence",
                 "removed_checkpoints", "step")

    def __init__(self, cfg, n_features, initial=1):
        self.cfg = cfg
        self.n_features = int(n_features)
        self.experts = {}
        self._next_id = 0
        self.spawn_evidence = 0
        self.removed_checkpoints = {}
        self.step = 0
        for _ in range(max(cfg.min_experts, int(initial))):
            self.spawn(force=True)

    # -- container protocol --------------------------------------------
    def __len__(self):
        return len(self.experts)

    def __iter__(self):
        return iter(self.experts.values())

    def __contains__(self, expert_id):
        return expert_id in self.experts

    def get(self, expert_id):
        return self.experts.get(expert_id)

    def ids(self):
        return sorted(self.experts.keys())

    def active(self):
        return [e for e in self.experts.values() if not e.frozen]

    def frozen(self):
        return [e for e in self.experts.values() if e.frozen]

    # -- lifecycle -----------------------------------------------------
    def spawn(self, force=False, seed=None):
        if not force and len(self.experts) >= self.cfg.max_experts:
            return None
        if force and len(self.experts) >= self.cfg.max_experts:
            return None
        expert_id = self._next_id
        self._next_id += 1
        expert = Expert(expert_id, self.cfg, self.n_features,
                        seed if seed is not None else (self.cfg.seed + expert_id))
        expert.created_at = self.step
        self.experts[expert_id] = expert
        self.spawn_evidence = 0
        return expert

    def should_spawn(self, best_confidence, failure_streak):
        """Guard against runaway expert creation."""
        if len(self.experts) >= self.cfg.max_experts:
            return False
        if best_confidence >= self.cfg.spawn_confidence:
            return False
        if failure_streak < self.cfg.spawn_failures:
            return False
        return True

    def note_failure(self):
        self.spawn_evidence += 1
        return self.spawn_evidence

    def note_success(self):
        if self.spawn_evidence > 0:
            self.spawn_evidence -= 1
        return self.spawn_evidence

    def maybe_spawn(self, best_confidence):
        if not self.should_spawn(best_confidence, self.spawn_evidence):
            return None
        return self.spawn()

    def find_merge_pair(self):
        """Return ``(keep, drop, similarity)`` for the most redundant pair."""
        experts = [e for e in self.experts.values() if not e.frozen]
        best = None
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                a, b = experts[i], experts[j]
                weight_similarity = a.neuron.cosine_similarity(b.neuron)
                context_similarity = a.signature_similarity(b)
                score = 0.6 * weight_similarity + 0.4 * context_similarity
                if score >= self.cfg.merge_similarity:
                    if best is None or score > best[2]:
                        keep, drop = (a, b) if a.usage_count >= b.usage_count else (b, a)
                        best = (keep, drop, score)
        return best

    def merge(self, keep, drop):
        """Usage-weighted average of the two parameter sets."""
        if keep.unique_id not in self.experts or drop.unique_id not in self.experts:
            return False
        wa = float(max(1, keep.usage_count))
        wb = float(max(1, drop.usage_count))
        total = wa + wb
        keep.checkpoint()
        for i in range(self.n_features):
            keep.neuron.w[i] = (keep.neuron.w[i] * wa + drop.neuron.w[i] * wb) / total
            keep.neuron.q[i] = (keep.neuron.q[i] * wa + drop.neuron.q[i] * wb) / total
            keep.signature[i] = (keep.signature[i] * wa + drop.signature[i] * wb) / total
        keep.neuron.b = (keep.neuron.b * wa + drop.neuron.b * wb) / total
        keep.usage_count += drop.usage_count
        keep.success_count += drop.success_count
        keep.reward_ema = (keep.reward_ema * wa + drop.reward_ema * wb) / total
        keep.error_ema = (keep.error_ema * wa + drop.error_ema * wb) / total
        for sample in drop.replay.items():
            keep.replay.add(sample)
        self.removed_checkpoints[drop.unique_id] = drop.to_dict(compact=False)
        del self.experts[drop.unique_id]
        return True

    def maybe_merge(self):
        if len(self.experts) <= self.cfg.min_experts:
            return None
        pair = self.find_merge_pair()
        if pair is None:
            return None
        keep, drop, score = pair
        # Reject the merge if it visibly hurts the kept expert on its own data.
        validation = keep.replay.items()
        if validation:
            pairs = [s.as_pair() for s in validation]
            before = keep.neuron.mean_error(pairs)
            if self.merge(keep, drop):
                after = keep.neuron.mean_error(pairs)
                if after > before + self.cfg.validation_tolerance:
                    keep.rollback()
                    restored = Expert.from_dict(
                        self.removed_checkpoints.pop(drop.unique_id),
                        self.cfg, self.n_features)
                    self.experts[restored.unique_id] = restored
                    return None
                return (keep.unique_id, drop.unique_id, score)
            return None
        if self.merge(keep, drop):
            return (keep.unique_id, drop.unique_id, score)
        return None

    def prune(self):
        """Remove a chronically bad, barely used expert (checkpoint kept)."""
        if len(self.experts) <= self.cfg.min_experts:
            return None
        candidates = []
        for expert in self.experts.values():
            if expert.frozen:
                continue
            if expert.usage_count >= self.cfg.prune_min_usage:
                continue
            if expert.reward_ema > self.cfg.prune_reward:
                continue
            if (self.step - expert.created_at) < self.cfg.retrain_cooldown * 2:
                continue
            candidates.append(expert)
        if not candidates:
            return None
        # Only drop it if someone else already covers the same region.
        candidates.sort(key=lambda e: e.reward_ema)
        victim = candidates[0]
        covered = False
        for other in self.experts.values():
            if other.unique_id == victim.unique_id:
                continue
            if (other.reward_ema > victim.reward_ema
                    and other.signature_similarity(victim) > 0.7):
                covered = True
                break
        if not covered:
            return None
        self.removed_checkpoints[victim.unique_id] = victim.to_dict(compact=False)
        del self.experts[victim.unique_id]
        return victim.unique_id

    def restore(self, expert_id):
        data = self.removed_checkpoints.get(expert_id)
        if data is None or len(self.experts) >= self.cfg.max_experts:
            return None
        expert = Expert.from_dict(data, self.cfg, self.n_features)
        self.experts[expert.unique_id] = expert
        return expert

    # -- selective retraining -------------------------------------------
    def retrain_due(self, context_stats=None, step=None):
        """Retrain **only** the experts that pass ``should_retrain``."""
        if step is not None:
            self.step = step
        reports = []
        for expert in list(self.experts.values()):
            if should_retrain(expert, context_stats):
                reports.append(expert.retrain(self.step))
        return reports

    # -- stats / serialisation -------------------------------------------
    def stats(self):
        experts = list(self.experts.values())
        count = len(experts)
        if count == 0:
            return {"count": 0, "active": 0, "frozen": 0,
                    "avg_reward": 0.0, "avg_error": 0.0, "avg_confidence": 0.0}
        return {
            "count": count,
            "active": sum(1 for e in experts if not e.frozen),
            "frozen": sum(1 for e in experts if e.frozen),
            "avg_reward": sum(e.reward_ema for e in experts) / count,
            "avg_error": sum(e.error_ema for e in experts) / count,
            "avg_confidence": sum(e.confidence_ema for e in experts) / count,
            "total_usage": sum(e.usage_count for e in experts),
        }

    def to_dict(self, compact=True):
        return {
            "n_features": self.n_features,
            "next_id": self._next_id,
            "step": self.step,
            "spawn_evidence": self.spawn_evidence,
            "experts": [e.to_dict(compact) for e in self.experts.values()],
        }

    @classmethod
    def from_dict(cls, data, cfg, n_features):
        pool = cls.__new__(cls)
        pool.cfg = cfg
        pool.n_features = int((data or {}).get("n_features", n_features)) or n_features
        if pool.n_features != n_features:
            pool.n_features = n_features
        pool.experts = {}
        pool.removed_checkpoints = {}
        pool._next_id = 0
        pool.spawn_evidence = int((data or {}).get("spawn_evidence", 0))
        pool.step = int((data or {}).get("step", 0))
        for entry in (data or {}).get("experts") or []:
            expert = Expert.from_dict(entry, cfg, pool.n_features)
            pool.experts[expert.unique_id] = expert
        pool._next_id = int((data or {}).get("next_id", 0))
        if pool.experts:
            pool._next_id = max(pool._next_id, max(pool.experts) + 1)
        while len(pool.experts) < cfg.min_experts:
            pool.spawn(force=True)
        return pool
