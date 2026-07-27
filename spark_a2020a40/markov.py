# -*- coding: utf-8 -*-
"""Backoff Markov scorer -- the backbone of generation.

The experts are good at *judging* a token (is this a plausible continuation
given eighteen features?) but they are hopeless at *ordering* one, because
ordering is a property of the corpus, not of a classifier with near-zero
initial weights.  The n-gram statistics already in :class:`TokenMemory` know
the ordering exactly; this module turns them into a probability instead of
burying them as three features among eighteen.

Scoring uses **stupid backoff** (Brants et al. 2007): cheap, needs no
discounting mass, and behaves well on tiny corpora.

    S(w | w₋₂ w₋₁) = c(w₋₂ w₋₁ w) / c(w₋₂ w₋₁)      when the trigram exists
                   = α · S(w | w₋₁)                   otherwise
    S(w | w₋₁)     = c(w₋₁ w) / c(w₋₁)                when the bigram exists
                   = α · S(w)                         otherwise
    S(w)           = unigram frequency, floored

It is a scoring function, not a normalised distribution -- which is exactly
what stupid backoff is designed to be.
"""

import math

from .tokenizer import BOS, EOS, UNK

#: Tokens that must never be generated.
BANNED = (BOS, UNK)


class MarkovScorer(object):
    """Backoff n-gram scoring plus topic conditioning on the question."""

    __slots__ = ("cfg", "memory")

    def __init__(self, cfg, memory):
        self.cfg = cfg
        self.memory = memory

    # ------------------------------------------------------------------
    def history_context(self, output_texts):
        """The (w₋₂, w₋₁) pair, padded with ``<bos>`` at the start."""
        if len(output_texts) >= 2:
            return output_texts[-2], output_texts[-1]
        if len(output_texts) == 1:
            return BOS, output_texts[0]
        return BOS, BOS

    # ------------------------------------------------------------------
    def unigram(self, candidate):
        record = self.memory.get(candidate)
        if record is None or self.memory.total_observations <= 0:
            return self.cfg.markov_floor
        probability = float(record.count) / float(self.memory.total_observations)
        return max(probability, self.cfg.markov_floor)

    def score(self, candidate, output_texts):
        """Stupid-backoff score in ``(0, 1]``."""
        alpha = self.cfg.backoff_alpha
        order = self.cfg.markov_order
        first, second = self.history_context(output_texts)

        if order >= 3:
            trigram = self.memory.context_score(first, second, candidate)
            if trigram > 0.0:
                return trigram

        if order >= 2:
            previous = output_texts[-1] if output_texts else BOS
            bigram = self.memory.pos_score(1, previous, candidate)
            if bigram > 0.0:
                return (alpha if order >= 3 else 1.0) * bigram

        penalty = alpha ** (order - 1) if order > 1 else 1.0
        # The floor is applied here too, so it really is a floor: the backoff
        # penalty must not push a known token below an unknown one.
        return max(penalty * self.unigram(candidate), self.cfg.markov_floor)

    def log_score(self, candidate, output_texts):
        return math.log(max(self.score(candidate, output_texts),
                            self.cfg.markov_floor))

    # ------------------------------------------------------------------
    def association_bonus(self, candidate, input_texts):
        """Topic conditioning: how strongly the question predicts this token.

        Without this term the chain would start every answer from ``<bos>``
        and therefore reply identically to every question.

        This is a **bonus**, ``log(1 + gain·p)``, not a log-probability.  That
        distinction is not cosmetic: as a log-probability it silently punished
        every token the question had never been seen with -- including
        ``<eos>``, which by construction never appears in an association table.
        The result was answers that could never end.  As a bonus, absent
        evidence scores exactly 0 and only positive evidence moves the score.
        """
        probability = self.memory.association_score(input_texts, candidate)
        if probability <= 0.0:
            return 0.0
        return math.log(1.0 + self.cfg.assoc_gain * probability)

    # ------------------------------------------------------------------
    def candidates(self, input_texts, output_texts, limit):
        """A small, evidence-driven candidate set.

        Sources, in order of trust: the trigram continuation, the bigram
        continuation, tokens associated with the question, and -- only to fill
        the remaining slots -- the globally most common tokens.
        """
        scored = {}

        def offer(token, weight):
            if token in BANNED or not token:
                return
            if token not in self.memory.tokens:
                return
            scored[token] = scored.get(token, 0.0) + weight

        first, second = self.history_context(output_texts)
        for token, count in self.memory.context_successors(first, second).items():
            offer(token, 100.0 * count)

        previous = output_texts[-1] if output_texts else BOS
        for token, count in self.memory.successors(previous, 1).items():
            offer(token, 10.0 * count)

        for token, count in self.memory.association_candidates(input_texts).items():
            offer(token, 1.0 * count)

        if len(scored) < limit:
            common = sorted(self.memory.tokens.items(), key=lambda kv: -kv[1].count)
            for text, record in common:
                if len(scored) >= limit:
                    break
                if record.kind == "special" and text != EOS:
                    continue
                offer(text, 0.01 * self.memory.commonality(text))

        # An answer must always be allowed to end.
        if output_texts:
            offer(EOS, 0.5)

        ordered = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        return [token for token, _ in ordered[:limit]]

    # ------------------------------------------------------------------
    def perplexity(self, token_sequences):
        """Diagnostic: lower is better.  Used by the benchmark and tests."""
        total_log = 0.0
        count = 0
        for tokens in token_sequences:
            history = []
            for token in list(tokens) + [EOS]:
                total_log += self.log_score(token, history)
                count += 1
                history.append(token)
        if count == 0:
            return float("inf")
        return math.exp(-total_log / float(count))
