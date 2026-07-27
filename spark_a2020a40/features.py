# -*- coding: utf-8 -*-
"""Feature extraction with a structural guarantee against label leakage.

``build_token_features`` does not accept a ``target_class_bit`` parameter *at
all*.  Passing one raises :class:`LabelLeakageError` instead of silently
training on the answer, so the leak cannot be reintroduced by a later edit
without a test failing.  The previous timestep's *prediction* is a legitimate
context feature and is exposed separately as
``previous_predicted_class_bit``.
"""

import unicodedata

from .tokenizer import normalize_text

FEATURE_NAMES = (
    "commonality_in_data",
    "length_token",
    "length_input",
    "length_output",
    "common_pos_1",
    "common_pos_2",
    "common_pos_3",
    "input_class_bit",
    "previous_predicted_class_bit",
    "error_probability",
    "repeats_in_input",
    "repeats_in_output",
    "repeats_in_window",
    "ord_sum",
    "ord_mean",
    "ord_weighted_sum",
    "ord_weighted_sum_mod_257",
    "ord_weighted_sum_mod_263",
)

INPUT_FEATURE_NAMES = (
    "input_length",
    "input_mean_commonality",
    "input_question",
    "input_greek_ratio",
    "input_latin_ratio",
    "input_digit_ratio",
    "input_punct_ratio",
    "input_unique_ratio",
    "input_mean_error",
    "input_ord_mod_257",
)

N_FEATURES = len(FEATURE_NAMES)
N_INPUT_FEATURES = len(INPUT_FEATURE_NAMES)

#: Names that must never be passed as an input feature of the same example.
LABEL_KEYS = frozenset((
    "target_class_bit", "target", "label", "y", "gold", "gold_class",
    "true_class", "ground_truth", "target_token",
))


class LabelLeakageError(ValueError):
    """Raised when a training label is offered as an input feature."""


def _reject_labels(kwargs, where):
    if not kwargs:
        return
    leaked = sorted(k for k in kwargs if k in LABEL_KEYS)
    if leaked:
        raise LabelLeakageError(
            "{0}() refuses label(s) {1} as input features; use "
            "previous_predicted_class_bit for the previous timestep instead".format(
                where, ", ".join(leaked)))
    raise TypeError("{0}() got unexpected keyword argument(s): {1}".format(
        where, ", ".join(sorted(kwargs))))


# ----------------------------------------------------------------------
def ord_features(text):
    """Return the ord-derived family described in the specification.

    ``ord_sum`` is kept exactly as specified (a plain sum over the NFC +
    casefolded text).  Because a plain sum collides on anagrams, the
    position-weighted variants are provided alongside it -- but as *extra*
    features, never as a replacement.
    """
    norm = normalize_text(text, casefold=True)
    if not norm:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    codes = [ord(ch) for ch in norm]
    ord_sum = 0
    weighted = 0
    for index, code in enumerate(codes):
        ord_sum += code
        weighted += (index + 1) * code
    ord_mean = float(ord_sum) / float(len(codes))
    return (
        float(ord_sum),
        ord_mean,
        float(weighted),
        float(weighted % 257),
        float(weighted % 263),
    )


class RunningNormalizer(object):
    """Online per-feature standardisation (Welford), O(1) memory.

    Without it a feature such as ``ord_sum`` (thousands) would drown
    ``common_pos_1`` (0..1) purely because of its numeric scale.
    """

    __slots__ = ("n", "mean", "m2", "warmup", "clip", "frozen", "size")

    def __init__(self, size, warmup=8, clip=4.0):
        self.size = size
        self.n = 0
        self.mean = [0.0] * size
        self.m2 = [0.0] * size
        self.warmup = max(2, int(warmup))
        self.clip = float(clip)
        self.frozen = False

    def observe(self, values):
        if self.frozen:
            return
        self.n += 1
        count = float(self.n)
        for i in range(self.size):
            value = float(values[i])
            delta = value - self.mean[i]
            self.mean[i] += delta / count
            self.m2[i] += delta * (value - self.mean[i])

    def transform(self, values):
        out = [0.0] * self.size
        if self.n < self.warmup:
            for i in range(self.size):
                value = float(values[i])
                out[i] = value / (1.0 + abs(value))
            return out
        denom = float(self.n - 1)
        for i in range(self.size):
            variance = self.m2[i] / denom if denom > 0 else 0.0
            std = variance ** 0.5
            if std < 1e-9:
                out[i] = 0.0
                continue
            z = (float(values[i]) - self.mean[i]) / std
            if z > self.clip:
                z = self.clip
            elif z < -self.clip:
                z = -self.clip
            out[i] = z / self.clip
        return out

    def observe_and_transform(self, values):
        self.observe(values)
        return self.transform(values)

    def to_dict(self):
        return {"n": self.n,
                "mean": [round(v, 6) for v in self.mean],
                "m2": [round(v, 6) for v in self.m2],
                "size": self.size}

    @classmethod
    def from_dict(cls, data, warmup=8, clip=4.0):
        size = int(data.get("size", 0)) or len(data.get("mean") or [])
        norm = cls(size or 1, warmup, clip)
        norm.n = int(data.get("n", 0))
        mean = list(data.get("mean") or [])
        m2 = list(data.get("m2") or [])
        if len(mean) == norm.size:
            norm.mean = [float(v) for v in mean]
        if len(m2) == norm.size:
            norm.m2 = [float(v) for v in m2]
        return norm


class FeatureExtractor(object):
    """Builds normalised feature vectors for candidate tokens and inputs."""

    __slots__ = ("cfg", "memory", "token_norm", "input_norm")

    def __init__(self, cfg, memory):
        self.cfg = cfg
        self.memory = memory
        self.token_norm = RunningNormalizer(N_FEATURES, cfg.normalizer_warmup, cfg.clip_sigma)
        self.input_norm = RunningNormalizer(N_INPUT_FEATURES, cfg.normalizer_warmup, cfg.clip_sigma)

    # -- names ---------------------------------------------------------
    @staticmethod
    def feature_names():
        return FEATURE_NAMES

    @staticmethod
    def input_feature_names():
        return INPUT_FEATURE_NAMES

    # -- candidate token features --------------------------------------
    def build_token_features(self, candidate, input_tokens=(), output_tokens=(),
                             input_class_bit=0, previous_predicted_class_bit=0,
                             normalize=True, learn=True, **forbidden):
        """Raw or normalised feature vector for one candidate next token.

        There is deliberately **no** ``target_class_bit`` parameter.
        """
        _reject_labels(forbidden, "build_token_features")

        candidate = candidate if isinstance(candidate, str) else str(candidate)
        input_texts = [t.text if hasattr(t, "text") else t for t in input_tokens]
        output_texts = [t.text if hasattr(t, "text") else t for t in output_tokens]

        memory = self.memory
        commonality = memory.commonality(candidate)

        prev1 = output_texts[-1] if len(output_texts) >= 1 else (
            input_texts[-1] if input_texts else None)
        prev2 = output_texts[-2] if len(output_texts) >= 2 else None
        prev3 = output_texts[-3] if len(output_texts) >= 3 else None
        if prev2 is None and input_texts:
            needed = 2 - len(output_texts)
            if 0 < needed <= len(input_texts):
                prev2 = input_texts[-needed]
        if prev3 is None and input_texts:
            needed = 3 - len(output_texts)
            if 0 < needed <= len(input_texts):
                prev3 = input_texts[-needed]

        pos1 = memory.pos_score(1, prev1, candidate)
        pos2 = memory.pos_score(2, prev2, candidate)
        pos3 = memory.pos_score(3, prev3, candidate)

        window = self.cfg.context_window
        recent = output_texts[-window:] if window > 0 else output_texts

        raw = [
            commonality,
            float(len(candidate)),
            float(len(input_texts)),
            float(len(output_texts)),
            pos1,
            pos2,
            pos3,
            1.0 if input_class_bit else 0.0,
            1.0 if previous_predicted_class_bit else 0.0,
            memory.error_probability(candidate),
            float(input_texts.count(candidate)),
            float(output_texts.count(candidate)),
            float(recent.count(candidate)),
        ]
        raw.extend(ord_features(candidate))

        if not normalize:
            return raw
        if learn:
            return self.token_norm.observe_and_transform(raw)
        return self.token_norm.transform(raw)

    # -- whole-input features ------------------------------------------
    def build_input_features(self, input_tokens, raw_text=None,
                             normalize=True, learn=True, **forbidden):
        _reject_labels(forbidden, "build_input_features")

        texts = [t.text if hasattr(t, "text") else t for t in input_tokens]
        n = len(texts)
        joined = raw_text if raw_text is not None else " ".join(texts)
        joined = normalize_text(joined, casefold=True)

        greek = latin = digits = punct = 0
        for ch in joined:
            if ch.isspace():
                continue
            category = unicodedata.category(ch)
            if category[0] == "N":
                digits += 1
            elif category[0] in ("P", "S"):
                punct += 1
            elif category[0] in ("L", "M"):
                if "GREEK" in unicodedata.name(ch, ""):
                    greek += 1
                else:
                    latin += 1
        total_chars = max(1, greek + latin + digits + punct)

        commonalities = [self.memory.commonality(t) for t in texts]
        errors = [self.memory.error_probability(t) for t in texts]
        weighted = 0
        for index, ch in enumerate(joined):
            weighted += (index + 1) * ord(ch)

        raw = [
            float(n),
            (sum(commonalities) / n) if n else 0.0,
            1.0 if ("?" in joined or ";" in joined) else 0.0,
            float(greek) / total_chars,
            float(latin) / total_chars,
            float(digits) / total_chars,
            float(punct) / total_chars,
            (float(len(set(texts))) / n) if n else 0.0,
            (sum(errors) / n) if n else 0.5,
            float(weighted % 257),
        ]
        if not normalize:
            return raw
        if learn:
            return self.input_norm.observe_and_transform(raw)
        return self.input_norm.transform(raw)

    # -- serialisation ---------------------------------------------------
    def to_dict(self):
        return {"token_norm": self.token_norm.to_dict(),
                "input_norm": self.input_norm.to_dict()}

    def load_dict(self, data):
        if not data:
            return self
        if data.get("token_norm"):
            self.token_norm = RunningNormalizer.from_dict(
                data["token_norm"], self.cfg.normalizer_warmup, self.cfg.clip_sigma)
        if data.get("input_norm"):
            self.input_norm = RunningNormalizer.from_dict(
                data["input_norm"], self.cfg.normalizer_warmup, self.cfg.clip_sigma)
        return self
