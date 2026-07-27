# -*- coding: utf-8 -*-
"""The student: routing, scoring and generation.

Everything here runs with the standard library only, so the phone can keep
answering after the laptop running Ollama has been switched off.
"""

import random

from .adaptive_neuron import AdaptiveQuadraticNeuron, softmax
from .experts import ExpertPool
from .features import FeatureExtractor, N_FEATURES, N_INPUT_FEATURES
from .memory import TokenMemory
from .router import Router
from .tokenizer import BOS, EOS, UNK, Tokenizer


class StudentModel(object):
    """Router + expert pool + symbolic memory + binary class predictor."""

    __slots__ = ("cfg", "tokenizer", "memory", "features", "pool", "router",
                 "class_predictor", "previous_predicted_class_bit",
                 "step", "_rng", "generations")

    def __init__(self, cfg, memory=None):
        self.cfg = cfg
        self.tokenizer = Tokenizer(cfg.casefold, cfg.max_token_chars)
        self.memory = memory if memory is not None else TokenMemory(cfg)
        self.features = FeatureExtractor(cfg, self.memory)
        self.pool = ExpertPool(cfg, N_FEATURES, initial=max(1, cfg.min_experts))
        self.router = Router(cfg, N_INPUT_FEATURES, self.pool.ids())
        self.class_predictor = AdaptiveQuadraticNeuron(
            N_INPUT_FEATURES, cfg.learning_rate, cfg.lambda_q, cfg.l2,
            cfg.init_scale, cfg.seed, cfg.max_abs_weight)
        self.previous_predicted_class_bit = 0
        self.step = 0
        self.generations = 0
        self._rng = random.Random(cfg.seed)

    # ==================================================================
    # binary input class
    # ==================================================================
    def predict_input_class(self, features):
        """``features`` are *input* features -- never token labels.

        Returns ``(probability, bit)``.
        """
        probability = self.class_predictor.predict_proba(features)
        return probability, (1 if probability >= 0.5 else 0)

    def train_class_predictor(self, features, target_class_bit):
        """The only place a class label may legally appear."""
        return self.class_predictor.train_step(features, 1 if target_class_bit else 0)

    def classify_text(self, text, learn=False):
        tokens = self.tokenizer.tokenize_typed(text)
        input_features = self.features.build_input_features(tokens, text, learn=learn)
        return self.predict_input_class(input_features)

    def derive_target_class_bit(self, text):
        """A deterministic weak label for the binary class.

        It is *not* a part of speech and *not* a POS tag -- it is a stable
        partition of inputs (here: interrogative / directive vs. declarative)
        that the student is asked to predict.  Replaceable without touching the
        training loop.
        """
        normalized = self.tokenizer.normalize(text)
        if "?" in normalized or ";" in normalized:
            return 1
        starters = ("τι", "πώς", "πως", "γιατί", "γιατι", "ποιος", "ποια", "ποιο",
                    "πού", "που", "πότε", "ποτε", "what", "how", "why", "who",
                    "where", "when", "which", "can", "do", "does", "is", "are")
        tokens = self.tokenizer.tokenize(normalized)
        if tokens and tokens[0] in starters:
            return 1
        return 0

    # ==================================================================
    # candidate generation and scoring
    # ==================================================================
    def candidate_tokens(self, input_texts, output_texts):
        """A bounded candidate set -- we never score the whole vocabulary."""
        limit = self.cfg.max_candidates
        scored = {}

        def offer(token, weight):
            if token in (BOS, UNK):
                return
            scored[token] = scored.get(token, 0.0) + weight

        previous = output_texts[-1] if output_texts else (
            input_texts[-1] if input_texts else None)
        if previous is not None:
            for token, count in self.memory.successors(previous, 1).items():
                offer(token, 3.0 * count)
        if len(output_texts) >= 2:
            for token, count in self.memory.successors(output_texts[-2], 2).items():
                offer(token, 1.5 * count)
        if len(output_texts) >= 3:
            for token, count in self.memory.successors(output_texts[-3], 3).items():
                offer(token, 1.0 * count)
        for text in input_texts:
            for token, count in self.memory.successors(text, 1).items():
                offer(token, 1.0 * count)
        if len(scored) < limit:
            # Back off to the globally most common tokens.
            common = sorted(self.memory.tokens.items(),
                            key=lambda kv: -kv[1].count)
            for text, record in common:
                if len(scored) >= limit:
                    break
                if record.kind == "special" and text != EOS:
                    continue
                offer(text, 0.1 * self.memory.commonality(text))
        offer(EOS, 0.5)

        ordered = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        return [token for token, _ in ordered[:limit]]

    def route(self, input_features):
        """Pick the Top-k experts for this context (never the whole pool)."""
        self.router.sync(self.pool.ids())
        expert_ids = self.router.select(input_features)
        return [self.pool.get(i) for i in expert_ids if self.pool.get(i) is not None]

    def score_candidates(self, candidates, experts, input_texts, output_texts,
                         input_class_bit, learn_norm=True):
        """Return ``(scores, feature_vectors, confidences)`` aligned with
        ``candidates``."""
        scores = []
        vectors = []
        confidences = []
        recent = output_texts[-self.cfg.repetition_window:] if output_texts else []
        for candidate in candidates:
            vector = self.features.build_token_features(
                candidate,
                input_tokens=input_texts,
                output_tokens=output_texts,
                input_class_bit=input_class_bit,
                previous_predicted_class_bit=self.previous_predicted_class_bit,
                learn=learn_norm,
            )
            vectors.append(vector)
            if experts:
                total = 0.0
                confidence = 0.0
                for expert in experts:
                    total += expert.score(vector)
                    confidence += expert.confidence(vector)
                total /= float(len(experts))
                confidence /= float(len(experts))
            else:
                total = 0.0
                confidence = 0.0
            # Repetition is discouraged in the score, not by a hard ban, so the
            # model can still repeat a token when the evidence is strong.
            repeats = recent.count(candidate)
            if repeats:
                total -= 1.25 * repeats
            scores.append(total)
            confidences.append(confidence)
        return scores, vectors, confidences

    # ==================================================================
    # generation
    # ==================================================================
    def generate(self, prompt, max_tokens=None, greedy=False, learn_norm=False):
        """Produce an answer using only local state.

        Returns a dict with the text, tokens, per-step confidences and the ids
        of the experts that were actually consulted.
        """
        max_tokens = self.cfg.max_generated_tokens if max_tokens is None else int(max_tokens)
        input_tokens = self.tokenizer.tokenize_typed(prompt)
        input_texts = [t.text for t in input_tokens]
        input_features = self.features.build_input_features(
            input_tokens, prompt, learn=learn_norm)
        probability, class_bit = self.predict_input_class(input_features)
        experts = self.route(input_features)

        output_texts = []
        confidences = []
        used_features = []
        for _ in range(max_tokens):
            candidates = self.candidate_tokens(input_texts, output_texts)
            if not candidates:
                break
            scores, vectors, step_confidences = self.score_candidates(
                candidates, experts, input_texts, output_texts, class_bit,
                learn_norm=learn_norm)
            index = self._choose(scores, greedy)
            token = candidates[index]
            confidences.append(step_confidences[index])
            used_features.append(vectors[index])
            if token == EOS:
                break
            output_texts.append(token)

        for expert in experts:
            if used_features:
                expert.observe_use(used_features[-1], confidences[-1] if confidences else 0.0)
            else:
                expert.observe_use(input_features + [0.0] * (len(expert.signature) - len(input_features))
                                   if len(input_features) < len(expert.signature) else input_features,
                                   0.0)

        self.previous_predicted_class_bit = class_bit
        self.generations += 1
        text = self.tokenizer.detokenize(output_texts)
        return {
            "text": text if text else "",
            "tokens": output_texts,
            "input_tokens": input_texts,
            "input_features": input_features,
            "class_bit": class_bit,
            "class_probability": probability,
            "experts": [e.unique_id for e in experts],
            "confidences": confidences,
            "features": used_features,
            "empty": not bool(output_texts),
        }

    def _choose(self, scores, greedy):
        if not scores:
            return 0
        if greedy or self.cfg.temperature <= 0.0:
            best = 0
            for i in range(1, len(scores)):
                if scores[i] > scores[best]:
                    best = i
            return best
        probabilities = softmax(scores, self.cfg.temperature)
        roll = self._rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if roll <= cumulative:
                return index
        return len(scores) - 1

    def answer(self, prompt):
        """Convenience wrapper returning plain text."""
        result = self.generate(prompt)
        if result["empty"]:
            return "Δεν ξέρω ακόμα. / I do not know yet."
        return result["text"]

    # ==================================================================
    # serialisation
    # ==================================================================
    def to_dict(self, compact=True):
        return {
            "memory": self.memory.to_dict(compact),
            "features": self.features.to_dict(),
            "pool": self.pool.to_dict(compact),
            "router": self.router.to_dict(),
            "class_predictor": self.class_predictor.to_dict(compact=False),
            "previous_predicted_class_bit": self.previous_predicted_class_bit,
            "step": self.step,
            "generations": self.generations,
        }

    def load_dict(self, data):
        if not data:
            return self
        if data.get("memory"):
            self.memory = TokenMemory.from_dict(self.cfg, data["memory"])
            self.features.memory = self.memory
        self.features.load_dict(data.get("features"))
        if data.get("pool"):
            self.pool = ExpertPool.from_dict(data["pool"], self.cfg, N_FEATURES)
        if data.get("router"):
            self.router = Router.from_dict(data["router"], self.cfg, N_INPUT_FEATURES)
        if data.get("class_predictor"):
            self.class_predictor = AdaptiveQuadraticNeuron.from_dict(
                data["class_predictor"], self.cfg)
        self.previous_predicted_class_bit = int(
            data.get("previous_predicted_class_bit", 0))
        self.step = int(data.get("step", 0))
        self.generations = int(data.get("generations", 0))
        self.router.sync(self.pool.ids())
        return self
