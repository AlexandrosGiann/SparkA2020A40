# -*- coding: utf-8 -*-
"""Compact, *bounded* symbolic token memory (the BitTree successor).

Differences from the legacy dictionary model:

* the hard ``MAX_TOTAL_ITEMS = 110`` wall -- which permanently stopped the model
  from learning after roughly fifteen sentences -- is replaced by an LRU
  eviction policy with a configurable id width;
* relation dictionaries are bounded per token.  The old ``add_relation`` grew
  ``relations`` without limit even after the token cap was hit, which was an
  unbounded memory leak on a 1 GB phone;
* ids are stored as integers and rendered as binary strings only on export,
  which is roughly seven times cheaper in RAM.
"""

from .tokenizer import EOS, UNK, BOS

MAX_ORDER = 3

#: Separator used to flatten a multi-token context into a JSON-safe dict key.
CONTEXT_SEP = "\u0001"


def context_key(previous_two):
    """``("γεια", "σου") -> "γεια\x01σου"``."""
    return CONTEXT_SEP.join(previous_two)


class TokenRecord(object):
    __slots__ = ("tid", "kind", "count", "pos", "cls0", "cls1",
                 "error_ema", "last_used")

    def __init__(self, tid, kind="word"):
        self.tid = tid
        self.kind = kind
        self.count = 0
        # pos[0] -> successors at distance 1, pos[1] -> distance 2, ...
        self.pos = [{}, {}, {}]
        self.cls0 = 0
        self.cls1 = 0
        self.error_ema = 0.5
        self.last_used = 0

    def total_relations(self):
        return sum(len(d) for d in self.pos)

    def class_bias(self):
        total = self.cls0 + self.cls1
        if total == 0:
            return 0.5
        return float(self.cls1) / float(total)


class TokenMemory(object):
    """Bounded token store with positional statistics of order 1..3."""

    __slots__ = ("cfg", "tokens", "_free_ids", "_next_id", "_clock",
                 "_max_count", "total_observations", "contexts", "associations")

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokens = {}
        self._free_ids = []
        self._next_id = 0
        self._clock = 0
        self._max_count = 1
        self.total_observations = 0
        # Order-2 contexts: (w-2, w-1) -> {successor: count}.  These are real
        # trigram statistics; ``pos[1]``/``pos[2]`` are only skip-grams and
        # cannot answer "what follows *this pair*".
        self.contexts = {}
        # Topic conditioning: input token -> {response token: count}.  Without
        # it a pure Markov chain starts every answer from <bos> and therefore
        # says the same thing regardless of the question.
        self.associations = {}
        for special in (UNK, BOS, EOS):
            self.ensure(special, kind="special")

    # -- capacity ------------------------------------------------------
    def __len__(self):
        return len(self.tokens)

    def _allocate_id(self):
        if self._free_ids:
            return self._free_ids.pop()
        if self._next_id >= self.cfg.id_capacity():
            return None
        tid = self._next_id
        self._next_id += 1
        return tid

    def _tick(self):
        self._clock += 1
        return self._clock

    def ensure(self, text, kind="word"):
        """Return the record for ``text``, creating (and evicting) as needed."""
        rec = self.tokens.get(text)
        if rec is not None:
            rec.last_used = self._tick()
            return rec
        if len(self.tokens) >= self.cfg.max_tokens:
            self._evict()
        tid = self._allocate_id()
        if tid is None:
            self._evict(force=True)
            tid = self._allocate_id()
            if tid is None:
                return self.tokens.get(UNK)
        rec = TokenRecord(tid, kind)
        rec.last_used = self._tick()
        self.tokens[text] = rec
        return rec

    def get(self, text):
        return self.tokens.get(text)

    def _evict(self, force=False):
        """Drop the least-recently-used, least-common non-special tokens.

        High-frequency tokens are protected outright: they are exactly the
        ones that make the symbolic memory useful, and a pure recency policy
        would happily throw away "γεια" to make room for a hapax.  The
        protection is lifted only when *every* remaining token is protected.
        """
        protected = (UNK, BOS, EOS)
        floor = max(2, int(self._max_count * 0.25))
        candidates = []
        fallback = []
        for text, rec in self.tokens.items():
            if text in protected:
                continue
            fallback.append((rec.last_used, text))
            if rec.count < floor:
                candidates.append((rec.last_used, text))
        if not candidates:
            candidates = fallback
        if not candidates:
            return 0
        candidates.sort()
        batch = max(1, min(self.cfg.eviction_batch, len(candidates)))
        removed = 0
        for _, text in candidates[:batch]:
            rec = self.tokens.pop(text, None)
            if rec is None:
                continue
            self._free_ids.append(rec.tid)
            removed += 1
            if not force and removed >= batch:
                break
        return removed

    def _bound_relations(self, rec):
        limit = self.cfg.max_relations_per_token
        for order in range(MAX_ORDER):
            table = rec.pos[order]
            if len(table) <= limit:
                continue
            # Keep the strongest associations only.
            keep = sorted(table.items(), key=lambda kv: kv[1], reverse=True)[:limit]
            rec.pos[order] = dict(keep)

    # -- learning ------------------------------------------------------
    def observe(self, text, kind="word", amount=1, class_bit=None):
        rec = self.ensure(text, kind)
        if rec is None:
            return None
        rec.count += amount
        if rec.count > self._max_count:
            self._max_count = rec.count
        if class_bit is not None:
            if class_bit:
                rec.cls1 += 1
            else:
                rec.cls0 += 1
        self.total_observations += amount
        return rec

    def observe_sequence(self, tokens, class_bit=None, weight=1, anchor=False):
        """Record counts, order-1/2/3 relations and order-2 contexts.

        ``anchor=True`` wraps the sequence in ``<bos> ... <eos>`` so the model
        learns how an answer *starts* and *stops*.  Without it the generator
        has no idea when to shut up -- which is exactly what the first version
        of this method got wrong.
        """
        texts = [t.text if hasattr(t, "text") else t for t in tokens]
        kinds = [t.kind if hasattr(t, "kind") else "word" for t in tokens]
        if anchor:
            texts = [BOS] + texts + [EOS]
            kinds = ["special"] + kinds + ["special"]
        for text, kind in zip(texts, kinds):
            self.observe(text, kind, weight, class_bit)
        for index in range(len(texts)):
            source = self.tokens.get(texts[index])
            if source is None:
                continue
            for order in range(1, MAX_ORDER + 1):
                target_index = index + order
                if target_index >= len(texts):
                    break
                target = texts[target_index]
                if target not in self.tokens:
                    continue
                table = source.pos[order - 1]
                table[target] = table.get(target, 0) + weight
            self._bound_relations(source)

        # Order-2 contexts (true trigrams).
        for index in range(len(texts) - 2):
            self.observe_context(texts[index], texts[index + 1],
                                 texts[index + 2], weight)
        return self

    # -- trigram contexts ----------------------------------------------
    def observe_context(self, first, second, successor, weight=1):
        key = context_key((first, second))
        table = self.contexts.get(key)
        if table is None:
            if len(self.contexts) >= self.cfg.max_contexts:
                self._evict_contexts()
            table = {}
            self.contexts[key] = table
        table[successor] = table.get(successor, 0) + weight
        limit = self.cfg.max_successors_per_context
        if len(table) > limit:
            keep = sorted(table.items(), key=lambda kv: kv[1], reverse=True)[:limit]
            self.contexts[key] = dict(keep)
        return self.contexts[key]

    def _evict_contexts(self):
        """Drop the weakest contexts -- the table is bounded like everything else."""
        if not self.contexts:
            return 0
        scored = sorted(self.contexts.items(),
                        key=lambda kv: sum(kv[1].values()))
        batch = max(1, self.cfg.eviction_batch)
        for key, _ in scored[:batch]:
            del self.contexts[key]
        return batch

    def context_successors(self, first, second):
        return self.contexts.get(context_key((first, second)), {})

    def context_score(self, first, second, candidate):
        """P(candidate | first, second) in [0, 1]."""
        table = self.contexts.get(context_key((first, second)))
        if not table:
            return 0.0
        hit = table.get(candidate)
        if not hit:
            return 0.0
        total = 0
        for value in table.values():
            total += value
        return (float(hit) / float(total)) if total > 0 else 0.0

    # -- prompt/response associations ----------------------------------
    def observe_association(self, input_texts, response_texts, weight=1):
        """Remember which answer tokens follow which question tokens."""
        limit = self.cfg.max_associations_per_token
        unique_response = []
        seen = set()
        for text in response_texts:
            if text not in seen:
                seen.add(text)
                unique_response.append(text)
        for source in set(input_texts):
            if source in (BOS, EOS, UNK):
                continue
            table = self.associations.get(source)
            if table is None:
                if len(self.associations) >= self.cfg.max_tokens:
                    self.associations.pop(next(iter(self.associations)), None)
                table = {}
                self.associations[source] = table
            for target in unique_response:
                if target in (BOS, UNK):
                    continue
                table[target] = table.get(target, 0) + weight
            if len(table) > limit:
                keep = sorted(table.items(), key=lambda kv: kv[1],
                              reverse=True)[:limit]
                self.associations[source] = dict(keep)
        return self

    def association_score(self, input_texts, candidate):
        """Mean P(candidate | question token) over the question, in [0, 1]."""
        if not input_texts:
            return 0.0
        total = 0.0
        seen = 0
        for source in set(input_texts):
            table = self.associations.get(source)
            if not table:
                continue
            hit = table.get(candidate)
            denominator = 0
            for value in table.values():
                denominator += value
            if denominator > 0:
                total += float(hit or 0) / float(denominator)
                seen += 1
        return (total / float(seen)) if seen else 0.0

    def association_candidates(self, input_texts):
        out = {}
        for source in set(input_texts):
            for target, count in (self.associations.get(source) or {}).items():
                out[target] = out.get(target, 0) + count
        return out

    def update_error(self, text, observed_error, alpha):
        rec = self.tokens.get(text)
        if rec is None:
            return 0.5
        rec.error_ema = alpha * rec.error_ema + (1.0 - alpha) * float(observed_error)
        return rec.error_ema

    # -- queries used by the feature extractor -------------------------
    def commonality(self, text):
        """Frequency normalised to [0, 1] -- never a raw count.

        The legacy scorer added a raw count to a raw relation weight, so the
        counter always dominated once the model had seen a few hundred tokens.
        """
        rec = self.tokens.get(text)
        if rec is None or self._max_count <= 0:
            return 0.0
        return float(rec.count) / float(self._max_count)

    def pos_score(self, order, previous_text, candidate_text):
        """P(candidate | previous at distance ``order``) in [0, 1]."""
        if previous_text is None or not (1 <= order <= MAX_ORDER):
            return 0.0
        rec = self.tokens.get(previous_text)
        if rec is None:
            return 0.0
        table = rec.pos[order - 1]
        hit = table.get(candidate_text)
        if not hit:
            return 0.0
        total = 0
        for value in table.values():
            total += value
        if total <= 0:
            return 0.0
        return float(hit) / float(total)

    def successors(self, previous_text, order=1):
        rec = self.tokens.get(previous_text)
        if rec is None:
            return {}
        return rec.pos[order - 1]

    def error_probability(self, text):
        rec = self.tokens.get(text)
        return 0.5 if rec is None else rec.error_ema

    def class_bias(self, text):
        rec = self.tokens.get(text)
        return 0.5 if rec is None else rec.class_bias()

    def binary_id(self, text):
        """Legacy-style fixed-width binary id (kept for compatibility)."""
        rec = self.tokens.get(text)
        if rec is None:
            return None
        return format(rec.tid, "0" + str(self.cfg.id_bits) + "b")

    def total_relations(self):
        return sum(rec.total_relations() for rec in self.tokens.values())

    # -- maintenance ---------------------------------------------------
    def compact(self):
        """Purge relation, context and association entries pointing at
        tokens that have since been evicted."""
        alive = self.tokens
        removed = 0
        for rec in alive.values():
            for order in range(MAX_ORDER):
                table = rec.pos[order]
                dead = [k for k in table if k not in alive]
                for key in dead:
                    del table[key]
                    removed += 1
        for key in list(self.contexts):
            parts = key.split(CONTEXT_SEP)
            if any(part not in alive for part in parts):
                del self.contexts[key]
                removed += 1
                continue
            table = self.contexts[key]
            for successor in [k for k in table if k not in alive]:
                del table[successor]
                removed += 1
            if not table:
                del self.contexts[key]
        for source in list(self.associations):
            if source not in alive:
                del self.associations[source]
                removed += 1
                continue
            table = self.associations[source]
            for target in [k for k in table if k not in alive]:
                del table[target]
                removed += 1
            if not table:
                del self.associations[source]
        return removed

    def total_contexts(self):
        return len(self.contexts)

    # -- serialisation -------------------------------------------------
    def to_dict(self, compact=False):
        tokens = {}
        for text, rec in self.tokens.items():
            entry = {
                "id": rec.tid,
                "k": rec.kind,
                "c": rec.count,
                "p1": rec.pos[0],
                "p2": rec.pos[1],
                "p3": rec.pos[2],
                "c0": rec.cls0,
                "c1": rec.cls1,
                "e": round(rec.error_ema, 4),
                "u": rec.last_used,
            }
            if compact:
                entry = dict((k, v) for k, v in entry.items() if v not in ({}, 0))
                entry["id"] = rec.tid
            tokens[text] = entry
        return {
            "tokens": tokens,
            "contexts": self.contexts,
            "associations": self.associations,
            "next_id": self._next_id,
            "free_ids": sorted(self._free_ids),
            "clock": self._clock,
            "max_count": self._max_count,
            "total_observations": self.total_observations,
        }

    @classmethod
    def from_dict(cls, cfg, data):
        memory = cls(cfg)
        memory.tokens = {}
        for text, entry in (data.get("tokens") or {}).items():
            rec = TokenRecord(int(entry.get("id", 0)), entry.get("k", "word"))
            rec.count = int(entry.get("c", 0))
            rec.pos = [
                dict(entry.get("p1") or {}),
                dict(entry.get("p2") or {}),
                dict(entry.get("p3") or {}),
            ]
            rec.cls0 = int(entry.get("c0", 0))
            rec.cls1 = int(entry.get("c1", 0))
            rec.error_ema = float(entry.get("e", 0.5))
            rec.last_used = int(entry.get("u", 0))
            memory.tokens[text] = rec
        memory.contexts = dict(
            (key, dict(value)) for key, value in (data.get("contexts") or {}).items())
        memory.associations = dict(
            (key, dict(value)) for key, value in (data.get("associations") or {}).items())
        memory._next_id = int(data.get("next_id", len(memory.tokens)))
        memory._free_ids = list(data.get("free_ids") or [])
        memory._clock = int(data.get("clock", 0))
        memory._max_count = max(1, int(data.get("max_count", 1)))
        memory.total_observations = int(data.get("total_observations", 0))
        for special in (UNK, BOS, EOS):
            memory.ensure(special, kind="special")
        return memory
