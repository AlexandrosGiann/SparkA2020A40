# -*- coding: utf-8 -*-
"""The student: routing, scoring and generation.

Everything here runs with the standard library only, so the phone can keep
answering after the laptop running Ollama has been switched off.
"""

import random

from .adaptive_neuron import AdaptiveQuadraticNeuron, softmax
from .experts import ExpertPool
from .features import FeatureExtractor, N_FEATURES, N_INPUT_FEATURES
from .markov import MarkovScorer
from .memory import TokenMemory
from .router import Router
from .tokenizer import (BOS, EOS, LANG_NEUTRAL, UNK, Tokenizer,
                        text_language, token_language)


class StudentModel(object):
    """Router + expert pool + symbolic memory + binary class predictor."""

    __slots__ = ("cfg", "tokenizer", "memory", "features", "pool", "router",
                 "class_predictor", "previous_predicted_class_bit",
                 "step", "_rng", "generations", "markov")

    def __init__(self, cfg, memory=None):
        self.cfg = cfg
        self.tokenizer = Tokenizer(cfg.casefold, cfg.max_token_chars)
        self.memory = memory if memory is not None else TokenMemory(cfg)
        self.features = FeatureExtractor(cfg, self.memory)
        self.markov = MarkovScorer(cfg, self.memory)
        self._bind_memory(self.memory)
        self.pool = ExpertPool(cfg, N_FEATURES, initial=max(1, cfg.min_experts))
        self.router = Router(cfg, N_INPUT_FEATURES, self.pool.ids())
        self.class_predictor = AdaptiveQuadraticNeuron(
            N_INPUT_FEATURES, cfg.learning_rate, cfg.lambda_q, cfg.l2,
            cfg.init_scale, cfg.seed, cfg.max_abs_weight)
        self.previous_predicted_class_bit = 0
        self.step = 0
        self.generations = 0
        self._rng = random.Random(cfg.seed)

    def _bind_memory(self, memory):
        """Point every collaborator at the same TokenMemory instance.

        Replacing ``self.memory`` on load while leaving a collaborator holding
        the old object is a silent, catastrophic bug: the model keeps all its
        learned state but generates from an empty memory, so it looks like it
        forgot everything on restart.  One method, one place to get it right.
        """
        self.memory = memory
        self.features.memory = memory
        self.markov.memory = memory
        return memory

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
        """A bounded candidate set driven by the n-gram evidence.

        Delegated to :class:`MarkovScorer` so that generation and distillation
        always see the same candidates.
        """
        return self.markov.candidates(input_texts, output_texts,
                                      self.cfg.max_candidates)

    def route(self, input_features):
        """Pick the Top-k experts for this context (never the whole pool)."""
        self.router.sync(self.pool.ids())
        expert_ids = self.router.select(input_features)
        return [self.pool.get(i) for i in expert_ids if self.pool.get(i) is not None]

    @staticmethod
    def emitted_ngrams(output_texts, size):
        """The n-grams already produced in this answer."""
        if size < 2 or len(output_texts) < size:
            return set()
        return set(tuple(output_texts[i:i + size])
                   for i in range(len(output_texts) - size + 1))

    def score_candidates(self, candidates, experts, input_texts, output_texts,
                         input_class_bit, learn_norm=True, seen_ngrams=None):
        """Return ``(scores, feature_vectors, confidences)`` aligned with
        ``candidates``.

        The score is a log-linear blend:

            w_markov · log S(token | w₋₂ w₋₁)      -- word order
          + w_assoc  · log(1 + gain·P(token | question))  -- topic (a bonus)
          + w_expert · expert_correction           -- learned re-ranking

        The Markov term is the backbone. The experts only *correct* it, which
        is why a freshly initialised pool no longer produces word salad: with
        zero-weight experts the blend degrades gracefully to a plain trigram
        model instead of to noise.
        """
        cfg = self.cfg
        scores = []
        vectors = []
        confidences = []
        recent = output_texts[-cfg.repetition_window:] if output_texts else []
        target_language = LANG_NEUTRAL
        if cfg.w_language > 0.0:
            target_language = text_language(input_texts)
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

            total = cfg.w_markov * self.markov.log_score(candidate, output_texts)
            total += cfg.w_assoc * self.markov.association_bonus(
                candidate, input_texts)

            if experts:
                correction = 0.0
                confidence = 0.0
                for expert in experts:
                    # Squashed to [-1, 1] so an unbounded logit cannot drown
                    # the language model.
                    correction += 2.0 * (expert.predict_proba(vector) - 0.5)
                    confidence += expert.confidence(vector)
                count = float(len(experts))
                total += cfg.w_expert * (correction / count)
                confidence /= count
            else:
                confidence = 0.0

            if candidate == EOS and output_texts:
                total += cfg.eos_pressure * self._stop_pressure(len(output_texts))
            elif seen_ngrams:
                size = cfg.no_repeat_ngram
                if len(output_texts) >= size - 1:
                    ngram = tuple(output_texts[-(size - 1):]) + (candidate,)
                    if ngram in seen_ngrams:
                        total -= cfg.w_no_repeat

            if target_language:
                candidate_language = self.memory.language(candidate)
                if candidate_language and candidate_language != target_language:
                    total -= cfg.w_language

            # Repetition is discouraged in the score, not by a hard ban.
            repeats = recent.count(candidate)
            if repeats:
                total -= 2.0 * repeats

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

        cfg_ngram = self.cfg.no_repeat_ngram
        output_texts = []
        confidences = []
        used_features = []
        finished = False
        for _ in range(max_tokens):
            candidates = self.candidate_tokens(input_texts, output_texts)
            if not candidates:
                break
            seen = self.emitted_ngrams(output_texts, cfg_ngram)
            scores, vectors, step_confidences = self.score_candidates(
                candidates, experts, input_texts, output_texts, class_bit,
                learn_norm=learn_norm,
                seen_ngrams=seen)
            index = self._choose(scores, greedy)
            token = candidates[index]
            confidences.append(step_confidences[index])
            used_features.append(vectors[index])
            if token == EOS:
                finished = True
                break
            if output_texts and self._is_debris(token, output_texts, seen):
                # The winner is only held up by a backed-off unigram, or it
                # would repeat a phrase we already emitted.  Either way the
                # learned material is exhausted -- stop rather than ramble,
                # and back up to the last complete sentence so the answer does
                # not end mid-clause.
                output_texts = self._trim_to_sentence(output_texts)
                finished = True
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
            "finished": finished,
        }

    #: Tokens that close a sentence in Greek or English.
    SENTENCE_END = (".", "!", "?", ";", "·", "...", "…")

    def _trim_to_sentence(self, output_texts):
        """Drop a trailing incomplete clause, if a complete one precedes it."""
        for index in range(len(output_texts) - 1, 0, -1):
            if output_texts[index] in self.SENTENCE_END:
                return output_texts[:index + 1]
        return output_texts

    def _is_debris(self, token, output_texts, seen_ngrams):
        """True when continuing with ``token`` would be padding, not language."""
        size = self.cfg.no_repeat_ngram
        if seen_ngrams and size >= 2 and len(output_texts) >= size - 1:
            ngram = tuple(output_texts[-(size - 1):]) + (token,)
            if ngram in seen_ngrams:
                return True
        floor = self.cfg.min_continuation_evidence
        if floor > 0.0 and self.markov.score(token, output_texts) < floor:
            return True
        return False

    def _stop_pressure(self, produced):
        """How hard to push for <eos>, in [0, ~1+].

        Two situations need it.  Offline-only models have never seen an
        anchored answer, so nothing tells them where to stop.  Trained models
        can still fall into a Markov cycle -- the chain re-enters a phrase it
        has already emitted and loops to the token cap.  In both cases the
        model's *own* statistics say what a normal answer looks like, so
        overshooting them is the signal to wrap up.
        """
        mean = self.memory.mean_answer_length()
        if mean <= 0.0:
            # No anchored answers at all: aim for half the hard cap.
            target = max(4, self.cfg.max_generated_tokens // 2)
            ratio = produced / float(target)
            return ratio * ratio
        # Learned answers exist: no pressure at all up to the usual length,
        # then a quadratic ramp so a runaway loop is cut off.
        if produced <= mean:
            return 0.0
        ratio = (produced - mean) / max(1.0, mean)
        return ratio * ratio

    def _has_stop_evidence(self):
        """True once the model has seen at least one anchored answer."""
        record = self.memory.get(EOS)
        return record is not None and record.count > 0

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
            self._bind_memory(TokenMemory.from_dict(self.cfg, data["memory"]))
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
