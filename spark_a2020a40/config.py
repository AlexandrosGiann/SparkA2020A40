# -*- coding: utf-8 -*-
"""Central configuration with two profiles.

``tiny_android``      -- the default and the primary priority.  Everything is
                         bounded aggressively so the process stays inside a few
                         megabytes of Python heap.
``desktop_training``  -- roomier limits for distillation runs on a laptop.

Every value can be overridden from the environment with the ``SPARK_`` prefix,
which finally removes the hard-coded ``192.168.1.29`` that used to be pasted in
three different places.
"""

import os

PROFILE_TINY = "tiny_android"
PROFILE_DESKTOP = "desktop_training"
PROFILES = (PROFILE_TINY, PROFILE_DESKTOP)


class ConfigError(ValueError):
    pass


def _env(name, default, cast):
    raw = os.environ.get("SPARK_" + name)
    if raw is None or raw == "":
        return default
    try:
        if cast is bool:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return cast(raw)
    except (TypeError, ValueError):
        return default


class Config(object):
    """Plain attribute bag -- no dataclasses, works on very old Python 3."""

    __slots__ = (
        "profile",
        # --- memory -----------------------------------------------------
        "id_bits", "max_tokens", "max_relations_per_token", "memory_file",
        "backup_suffix", "compact_mode", "eviction_batch",
        # --- tokenizer --------------------------------------------------
        "max_token_chars", "casefold",
        # --- markov ------------------------------------------------------
        "markov_order", "backoff_alpha", "max_contexts",
        "max_successors_per_context", "max_associations_per_token",
        "w_markov", "w_assoc", "w_expert", "markov_floor", "assoc_gain",
        "eos_pressure", "w_language", "match_language", "teacher_system",
        "no_repeat_ngram", "w_no_repeat", "min_continuation_evidence",
        # --- features ---------------------------------------------------
        "context_window", "normalizer_warmup", "clip_sigma",
        # --- neuron -----------------------------------------------------
        "learning_rate", "lambda_q", "l2", "init_scale", "max_abs_weight",
        # --- experts ----------------------------------------------------
        "max_experts", "min_experts", "expert_replay_size",
        "retrain_cooldown", "retrain_min_samples", "retrain_epochs",
        "error_threshold", "reward_threshold", "confidence_threshold",
        "spawn_confidence", "spawn_failures", "merge_similarity",
        "prune_min_usage", "prune_reward", "validation_fraction",
        "validation_tolerance", "freeze_stability_steps",
        # --- router -----------------------------------------------------
        "router_top_k", "router_lr", "router_explore",
        # --- rewards ----------------------------------------------------
        "reward_clip", "reward_ema_alpha", "error_ema_alpha",
        "w_teacher", "w_class", "w_user", "w_relevance",
        "w_repetition", "w_invalid", "w_length", "w_uncertainty",
        # --- generation -------------------------------------------------
        "max_generated_tokens", "max_candidates", "temperature",
        "repetition_window",
        # --- teacher ----------------------------------------------------
        "ollama_host", "ollama_port", "ollama_model", "teacher_timeout",
        "teacher_probe_timeout", "teacher_recheck_seconds", "teacher_enabled",
        # --- misc -------------------------------------------------------
        "seed", "debug", "autosave_every",
    )

    def __init__(self, profile=PROFILE_TINY):
        if profile not in PROFILES:
            raise ConfigError("unknown profile: " + repr(profile))
        self.profile = profile
        tiny = (profile == PROFILE_TINY)

        # memory ---------------------------------------------------------
        self.id_bits = _env("ID_BITS", 12 if tiny else 16, int)
        self.max_tokens = _env("MAX_TOKENS", 1500 if tiny else 20000, int)
        self.max_relations_per_token = _env("MAX_RELATIONS", 12 if tiny else 48, int)
        self.memory_file = _env("MEMORY_FILE", "spark_memory.json", str)
        self.backup_suffix = ".bak"
        self.compact_mode = _env("COMPACT", tiny, bool)
        self.eviction_batch = 16 if tiny else 128

        # tokenizer ------------------------------------------------------
        self.max_token_chars = _env("MAX_TOKEN_CHARS", 32, int)
        self.casefold = _env("CASEFOLD", True, bool)

        # markov ---------------------------------------------------------
        self.markov_order = _env("MARKOV_ORDER", 3, int)
        self.backoff_alpha = _env("BACKOFF_ALPHA", 0.4, float)
        self.max_contexts = _env("MAX_CONTEXTS", 800 if tiny else 12000, int)
        self.max_successors_per_context = _env("MAX_SUCCESSORS", 6 if tiny else 24, int)
        self.max_associations_per_token = _env("MAX_ASSOCIATIONS", 10 if tiny else 40, int)
        self.w_markov = _env("W_MARKOV", 1.0, float)
        self.w_assoc = _env("W_ASSOC", 3.0, float)
        self.w_expert = _env("W_EXPERT", 0.8, float)
        self.markov_floor = _env("MARKOV_FLOOR", 1e-6, float)
        self.assoc_gain = _env("ASSOC_GAIN", 20.0, float)
        self.eos_pressure = _env("EOS_PRESSURE", 20.0, float)
        # Standard no-repeat-ngram decoding constraint.  The plain repetition
        # window only sees the last few tokens, so a Markov cycle longer than
        # the window (a repeated *phrase*) slips straight past it.
        self.no_repeat_ngram = _env("NO_REPEAT_NGRAM", 3, int)
        self.w_no_repeat = _env("W_NO_REPEAT", 25.0, float)
        # "If I have nothing solid to say next, stop."  Once the chosen
        # continuation is only supported by a backed-off unigram, the model has
        # run out of learned material; carrying on just emits plausible-looking
        # debris.  Stupid-backoff scores a real bigram/trigram well above this.
        self.min_continuation_evidence = _env("MIN_EVIDENCE", 0.02, float)
        # Answer in the language of the question.  Two independent levers:
        #
        #   match_language -- strip foreign-language asides from the teacher's
        #       reply *before* it reaches the memory.  This is the one that
        #       works: measured on a bilingual corpus it takes language purity
        #       from 70% to 100% with no loops.
        #   w_language -- penalise off-language candidates during generation.
        #       Measured at 85% purity, but it fights the n-gram evidence
        #       mid-phrase and caused loops (average answer 23 tokens instead
        #       of 7).  Off by default; raise it only to clean up a memory that
        #       was already polluted before match_language existed.
        self.match_language = _env("MATCH_LANGUAGE", True, bool)
        self.w_language = _env("W_LANGUAGE", 0.0, float)
        self.teacher_system = _env(
            "TEACHER_SYSTEM",
            "You are a concise assistant. Reply ONLY in {language}. "
            "Give the answer itself and nothing else: no translations, no "
            "parenthetical glosses, no definitions or explanations of your own "
            "words, no notes about what language you are using. "
            "Answer in one or two short sentences.", str)

        # features -------------------------------------------------------
        self.context_window = _env("CONTEXT_WINDOW", 12 if tiny else 32, int)
        self.normalizer_warmup = 8
        self.clip_sigma = 4.0

        # neuron ---------------------------------------------------------
        self.learning_rate = _env("LR", 0.05, float)
        self.lambda_q = _env("LAMBDA_Q", 0.05, float)
        self.l2 = _env("L2", 0.0001, float)
        self.init_scale = 0.0
        self.max_abs_weight = 25.0

        # experts --------------------------------------------------------
        self.max_experts = _env("MAX_EXPERTS", 6 if tiny else 24, int)
        self.min_experts = 1
        self.expert_replay_size = _env("EXPERT_REPLAY", 48 if tiny else 512, int)
        self.retrain_cooldown = _env("RETRAIN_COOLDOWN", 25 if tiny else 10, int)
        self.retrain_min_samples = _env("RETRAIN_MIN_SAMPLES", 12, int)
        self.retrain_epochs = _env("RETRAIN_EPOCHS", 3 if tiny else 8, int)
        self.error_threshold = _env("ERROR_THRESHOLD", 0.35, float)
        self.reward_threshold = _env("REWARD_THRESHOLD", 0.15, float)
        self.confidence_threshold = _env("CONFIDENCE_THRESHOLD", 0.55, float)
        self.spawn_confidence = _env("SPAWN_CONFIDENCE", 0.45, float)
        self.spawn_failures = _env("SPAWN_FAILURES", 20, int)
        self.merge_similarity = _env("MERGE_SIMILARITY", 0.985, float)
        self.prune_min_usage = _env("PRUNE_MIN_USAGE", 15, int)
        self.prune_reward = _env("PRUNE_REWARD", -0.25, float)
        self.validation_fraction = 0.25
        self.validation_tolerance = _env("VALIDATION_TOLERANCE", 0.02, float)
        self.freeze_stability_steps = _env("FREEZE_STEPS", 400, int)

        # router ---------------------------------------------------------
        self.router_top_k = _env("ROUTER_TOP_K", 1 if tiny else 2, int)
        self.router_lr = _env("ROUTER_LR", 0.08, float)
        self.router_explore = _env("ROUTER_EXPLORE", 0.35, float)

        # rewards --------------------------------------------------------
        self.reward_clip = _env("REWARD_CLIP", 1.0, float)
        self.reward_ema_alpha = _env("REWARD_ALPHA", 0.9, float)
        self.error_ema_alpha = _env("ERROR_ALPHA", 0.9, float)
        self.w_teacher = _env("W_TEACHER", 1.0, float)
        self.w_class = _env("W_CLASS", 0.5, float)
        self.w_user = _env("W_USER", 1.0, float)
        self.w_relevance = _env("W_RELEVANCE", 0.4, float)
        self.w_repetition = _env("W_REPETITION", 0.6, float)
        self.w_invalid = _env("W_INVALID", 1.0, float)
        self.w_length = _env("W_LENGTH", 0.3, float)
        self.w_uncertainty = _env("W_UNCERTAINTY", 0.2, float)

        # generation -----------------------------------------------------
        self.max_generated_tokens = _env("MAX_GEN", 40 if tiny else 96, int)
        self.max_candidates = _env("MAX_CANDIDATES", 16 if tiny else 48, int)
        self.temperature = _env("TEMPERATURE", 0.1, float)
        self.repetition_window = _env("REPETITION_WINDOW", 8, int)

        # teacher --------------------------------------------------------
        self.ollama_host = _env("OLLAMA_HOST", "192.168.1.29", str)
        self.ollama_port = _env("OLLAMA_PORT", 11434, int)
        self.ollama_model = _env("OLLAMA_MODEL", "aya-expanse:8b", str)
        self.teacher_timeout = _env("TEACHER_TIMEOUT", 60.0, float)
        self.teacher_probe_timeout = _env("TEACHER_PROBE_TIMEOUT", 1.5, float)
        self.teacher_recheck_seconds = _env("TEACHER_RECHECK", 30.0, float)
        self.teacher_enabled = _env("TEACHER_ENABLED", True, bool)

        # misc -----------------------------------------------------------
        self.seed = _env("SEED", 20200440, int)
        self.debug = _env("DEBUG", False, bool)
        self.autosave_every = _env("AUTOSAVE_EVERY", 5, int)

        self.validate()

    # ------------------------------------------------------------------
    def validate(self):
        if self.id_bits < 4 or self.id_bits > 32:
            raise ConfigError("id_bits must be within 4..32")
        if self.max_tokens > self.id_capacity():
            raise ConfigError(
                "max_tokens ({0}) exceeds id capacity for {1} bits ({2})".format(
                    self.max_tokens, self.id_bits, self.id_capacity()))
        if self.max_experts < self.min_experts:
            raise ConfigError("max_experts < min_experts")
        if self.router_top_k < 1:
            raise ConfigError("router_top_k must be >= 1")
        if not (0.0 < self.reward_ema_alpha < 1.0):
            raise ConfigError("reward_ema_alpha must be in (0,1)")
        if not (0.0 < self.error_ema_alpha < 1.0):
            raise ConfigError("error_ema_alpha must be in (0,1)")
        if self.lambda_q < 0.0:
            raise ConfigError("lambda_q must be >= 0")
        if not (0.0 < self.backoff_alpha < 1.0):
            raise ConfigError("backoff_alpha must be in (0,1)")
        if self.markov_order < 1 or self.markov_order > 3:
            raise ConfigError("markov_order must be 1, 2 or 3")
        return self

    def id_capacity(self):
        return 1 << self.id_bits

    def ollama_url(self, path="/api/generate"):
        return "http://{0}:{1}{2}".format(self.ollama_host, self.ollama_port, path)

    def reward_weights(self):
        return {
            "teacher_agreement": self.w_teacher,
            "class_correctness": self.w_class,
            "user_feedback": self.w_user,
            "relevance": self.w_relevance,
            "repetition": -self.w_repetition,
            "invalid_output": -self.w_invalid,
            "excessive_length": -self.w_length,
            "uncertainty": -self.w_uncertainty,
        }

    def to_dict(self):
        out = {}
        for name in self.__slots__:
            out[name] = getattr(self, name)
        return out

    def __repr__(self):
        return "<Config profile={0} max_experts={1} max_tokens={2}>".format(
            self.profile, self.max_experts, self.max_tokens)


_ACTIVE = None


def get_config(profile=None, refresh=False):
    """Return the process-wide configuration singleton."""
    global _ACTIVE
    if _ACTIVE is None or refresh or (profile is not None and profile != _ACTIVE.profile):
        _ACTIVE = Config(profile or os.environ.get("SPARK_PROFILE") or PROFILE_TINY)
    return _ACTIVE


def set_config(cfg):
    global _ACTIVE
    _ACTIVE = cfg
    return cfg
