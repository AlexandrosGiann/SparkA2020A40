# -*- coding: utf-8 -*-
"""The training coordinator: the twelve-step teacher/student turn.

 1. receive input                    7. build positive/negative candidates
 2. extract input features           8. supervised distillation update
 3. predict input_class_bit          9. student generation
 4. ask the Ollama teacher          10. compute reward
 5. receive teacher response        11. update ONLY the routed experts
 6. tokenize teacher response       12. persist compact memory

Step 4/5 are optional: when the teacher is unreachable the turn continues from
step 9 and the student still answers, learns from reward, and saves.
"""

import random

from .experts import should_retrain
from .features import N_FEATURES
from .rewards import RewardEngine
from .teacher import TeacherClient
from .tokenizer import EOS

NEGATIVES_PER_POSITIVE = 3


class TrainingCoordinator(object):
    """Owns the turn loop, the reward plumbing and expert lifecycle policy."""

    __slots__ = ("cfg", "student", "teacher", "rewards", "persistence",
                 "turns", "_rng", "_failure_streak", "last_turn",
                 "_since_save", "_since_maintenance", "debug")

    def __init__(self, cfg, student, teacher=None, rewards=None, persistence=None):
        self.cfg = cfg
        self.student = student
        self.teacher = teacher if teacher is not None else TeacherClient(cfg)
        self.rewards = rewards if rewards is not None else RewardEngine(cfg)
        self.persistence = persistence
        self.turns = 0
        self._rng = random.Random(cfg.seed + 1)
        self._failure_streak = 0
        self.last_turn = None
        self._since_save = 0
        self._since_maintenance = 0
        self.debug = cfg.debug

    # ==================================================================
    # distillation
    # ==================================================================
    def build_training_pairs(self, input_texts, teacher_texts):
        """Positive = the teacher's actual next token; negatives = plausible
        alternatives drawn from the student's own candidate set."""
        student = self.student
        pairs = []
        output_texts = []
        sequence = list(teacher_texts) + [EOS]
        class_bit = student.previous_predicted_class_bit
        for positive in sequence:
            candidates = student.candidate_tokens(input_texts, output_texts)
            negatives = [c for c in candidates if c != positive]
            self._rng.shuffle(negatives)
            negatives = negatives[:NEGATIVES_PER_POSITIVE]
            for token, target in [(positive, 1)] + [(n, 0) for n in negatives]:
                vector = student.features.build_token_features(
                    token,
                    input_tokens=input_texts,
                    output_tokens=output_texts,
                    input_class_bit=class_bit,
                    previous_predicted_class_bit=student.previous_predicted_class_bit,
                    learn=True,
                )
                pairs.append((vector, target, token))
            if positive == EOS:
                break
            output_texts.append(positive)
        return pairs

    def distill(self, input_texts, teacher_texts, experts):
        """Supervised update of the routed experts only."""
        if not experts:
            return {"pairs": 0, "mean_error": 0.0, "experts": []}
        pairs = self.build_training_pairs(input_texts, teacher_texts)
        total_error = 0.0
        step = self.student.step
        for vector, target, token in pairs:
            for expert in experts:
                if expert.frozen:
                    continue
                error = expert.train_online(vector, target, step)
                total_error += error
                expert.remember(vector, target, 0.0, step, token)
            self.student.memory.update_error(
                token, 1.0 if target == 0 else 0.0, self.cfg.error_ema_alpha)
        divisor = float(max(1, len(pairs) * max(1, len(experts))))
        return {
            "pairs": len(pairs),
            "mean_error": total_error / divisor,
            "experts": [e.unique_id for e in experts],
        }

    # ==================================================================
    # drift detection
    # ==================================================================
    def detect_drift(self, input_features):
        """Mean absolute normalised deviation of the current input.

        The normaliser already produces values in roughly [-1, 1]; a sustained
        excursion beyond ~0.6 means the incoming distribution moved.
        """
        if not input_features:
            return 0.0
        total = 0.0
        for value in input_features:
            total += abs(float(value))
        return total / float(len(input_features))

    # ==================================================================
    # the turn
    # ==================================================================
    def process_turn(self, user_input, user_feedback=None, use_teacher=True,
                     teacher_text=None):
        student = self.student
        cfg = self.cfg
        student.step += 1
        self.turns += 1
        report = {"input": user_input, "step": student.step}

        # 1-2. input + features -----------------------------------------
        input_tokens = student.tokenizer.tokenize_typed(user_input)
        input_texts = [t.text for t in input_tokens]
        input_features = student.features.build_input_features(
            input_tokens, user_input, learn=True)

        # 3. predict the binary input class (never using the label) -------
        probability, predicted_class = student.predict_input_class(input_features)
        target_class = student.derive_target_class_bit(user_input)
        report["input_features"] = input_features
        report["class"] = {"predicted": predicted_class,
                           "probability": probability,
                           "target": target_class}
        # The label is used *here only*, for the class head, after prediction.
        student.train_class_predictor(input_features, target_class)

        # 4-6. teacher --------------------------------------------------
        teacher_texts = []
        if teacher_text is None and use_teacher and self.teacher.is_available():
            teacher_text = self.teacher.generate(user_input)
        report["teacher_available"] = teacher_text is not None
        report["teacher_text"] = teacher_text
        if teacher_text:
            teacher_tokens = student.tokenizer.tokenize_typed(teacher_text)
            teacher_texts = [t.text for t in teacher_tokens]
            student.memory.observe_sequence(input_tokens, class_bit=target_class)
            # anchor=True wraps the answer in <bos>...<eos>, which is what
            # teaches the generator where an answer starts and stops.
            student.memory.observe_sequence(teacher_tokens, class_bit=target_class,
                                            anchor=True)
            student.memory.observe_association(input_texts, teacher_texts)

        # routing happens once per turn, not once per candidate ----------
        experts = student.route(input_features)
        report["experts"] = [e.unique_id for e in experts]

        # 7-8. candidates + distillation --------------------------------
        if teacher_texts:
            report["distill"] = self.distill(input_texts, teacher_texts, experts)
        else:
            report["distill"] = {"pairs": 0, "mean_error": 0.0, "experts": []}

        # 9. student generation -----------------------------------------
        generation = student.generate(user_input, learn_norm=True)
        report["student_text"] = generation["text"]
        report["student_tokens"] = generation["tokens"]

        # 10. reward ----------------------------------------------------
        reward_ctx = {
            "input_tokens": input_texts,
            "teacher_tokens": teacher_texts,
            "student_tokens": generation["tokens"],
            "predicted_class": predicted_class,
            "target_class": target_class,
            "user_feedback": user_feedback,
            "confidences": generation["confidences"],
            "max_tokens": cfg.max_generated_tokens,
            "target_length": len(teacher_texts) or cfg.max_generated_tokens // 2,
        }
        reward = self.rewards.compute(reward_ctx)
        report["reward"] = reward
        report["reward_breakdown"] = dict(self.rewards.last_breakdown)

        agreement = self.rewards.last_breakdown.get("teacher_agreement", 0.0)
        disagreement = (1.0 - agreement) if teacher_texts else 0.0
        observed_error = max(0.0, min(1.0, 0.5 - reward / (2.0 * cfg.reward_clip)))

        # 11. update ONLY the routed experts + the router ----------------
        for expert in experts:
            expert.observe_outcome(observed_error, reward, target_class)
            self.student.router.update(expert.unique_id, input_features, reward)
        if reward < 0:
            self._failure_streak += 1
            student.pool.note_failure()
        else:
            self._failure_streak = 0
            student.pool.note_success()

        drift = self.detect_drift(input_features)
        context_stats = {
            "teacher_disagreement": disagreement,
            "drift": drift,
            "drift_threshold": 0.6,
            "distribution_shift": drift > 0.85,
            "class_failure_limit": 5,
        }
        report["context_stats"] = context_stats

        student.pool.step = student.step
        retrained = []
        for expert in experts:
            if should_retrain(expert, context_stats):
                retrained.append(expert.retrain(student.step))
        report["retrained"] = retrained

        # expert lifecycle -----------------------------------------------
        self._since_maintenance += 1
        if self._since_maintenance >= 10:
            self._since_maintenance = 0
            report["lifecycle"] = self.maintain(generation["confidences"])
        else:
            report["lifecycle"] = {}

        # 12. compact persistence ---------------------------------------
        self._since_save += 1
        if self.persistence is not None and self._since_save >= cfg.autosave_every:
            self._since_save = 0
            report["saved"] = bool(self.save())
        else:
            report["saved"] = False

        self.last_turn = report
        return report

    # ==================================================================
    # lifecycle + feedback
    # ==================================================================
    def maintain(self, confidences=()):
        student = self.student
        actions = {}
        best_confidence = max(confidences) if confidences else 0.0
        spawned = student.pool.maybe_spawn(best_confidence)
        if spawned is not None:
            actions["spawned"] = spawned.unique_id
        merged = student.pool.maybe_merge()
        if merged is not None:
            actions["merged"] = {"kept": merged[0], "dropped": merged[1],
                                 "similarity": merged[2]}
        pruned = student.pool.prune()
        if pruned is not None:
            actions["pruned"] = pruned
        frozen = [e.unique_id for e in student.pool if e.maybe_freeze(student.step)]
        if frozen:
            actions["frozen"] = frozen
        student.router.sync(student.pool.ids())
        return actions

    def apply_feedback(self, value):
        """``:feedback +1`` / ``:feedback -1`` from the CLI."""
        if self.last_turn is None:
            return None
        reward = self.cfg.w_user * (1.0 if value > 0 else -1.0)
        reward = max(-self.cfg.reward_clip, min(self.cfg.reward_clip, reward))
        features = self.last_turn.get("input_features")
        for expert_id in self.last_turn.get("experts", []):
            expert = self.student.pool.get(expert_id)
            if expert is None:
                continue
            expert.observe_outcome(0.0 if value > 0 else 1.0, reward)
            if features:
                self.student.router.update(expert_id, features, reward)
        self.rewards.reward_ema = (self.cfg.reward_ema_alpha * self.rewards.reward_ema
                                   + (1.0 - self.cfg.reward_ema_alpha) * reward)
        return reward

    def force_retrain(self, ignore_gate=False):
        """``:retrain`` -- still selective unless explicitly overridden."""
        reports = []
        stats = {"teacher_disagreement": 0.0, "drift": 0.0,
                 "drift_threshold": 0.6, "distribution_shift": False}
        for expert in list(self.student.pool):
            if ignore_gate or should_retrain(expert, stats):
                reports.append(expert.retrain(self.student.step))
        return reports

    # ==================================================================
    def save(self):
        if self.persistence is None:
            return False
        return self.persistence.save(self.build_state())

    def build_state(self):
        state = self.student.to_dict(compact=self.cfg.compact_mode)
        state["rewards"] = self.rewards.to_dict()
        state["turns"] = self.turns
        return state

    def load_state(self, state):
        if not state:
            return self
        self.student.load_dict(state)
        self.rewards.load_dict(state.get("rewards"))
        self.turns = int(state.get("turns", 0))
        return self

    # ==================================================================
    def stats(self):
        student = self.student
        pool_stats = student.pool.stats()
        return {
            "tokens": len(student.memory),
            "relations": student.memory.total_relations(),
            "experts": pool_stats["count"],
            "active_experts": pool_stats["active"],
            "frozen_experts": pool_stats["frozen"],
            "avg_reward": self.rewards.average(),
            "reward_ema": self.rewards.reward_ema,
            "avg_error": pool_stats["avg_error"],
            "avg_confidence": pool_stats["avg_confidence"],
            "router_usage": student.router.usage_distribution(),
            "router_entropy": student.router.entropy(),
            "turns": self.turns,
            "steps": student.step,
            "generations": student.generations,
            "teacher": self.teacher.status(),
        }
