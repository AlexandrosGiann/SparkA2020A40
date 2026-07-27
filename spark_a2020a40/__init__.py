# -*- coding: utf-8 -*-
"""SparkA2020A40 -- experimental ultra-light teacher/student chatbot.

The *student* runtime is pure standard library and is designed to run on old
Android hardware (Lenovo A2020a40 class: 1 GB RAM, Android 5.1.1).

Nothing in this package (outside ``desktop_trainer.py``) imports NumPy,
PyTorch, TensorFlow, JAX or scikit-learn.  The source deliberately avoids
f-strings and dataclasses so that it stays parsable by the older Python 3
interpreters shipped with QPython 3H / Pydroid.
"""

__version__ = "2.1.0"
__author__ = "Alexandros Giannakis"

SCHEMA_VERSION = 3

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "Config",
    "get_config",
    "Tokenizer",
    "FeatureExtractor",
    "TokenMemory",
    "AdaptiveQuadraticNeuron",
    "Expert",
    "ExpertPool",
    "MarkovScorer",
    "Router",
    "RewardEngine",
    "ReplayBuffer",
    "TeacherClient",
    "StudentModel",
    "TrainingCoordinator",
]


def __getattr__(name):
    # Lazy re-export keeps ``import spark_a2020a40`` cheap on the phone.
    if name in ("Config", "get_config"):
        from . import config as _m
        return getattr(_m, name)
    if name == "Tokenizer":
        from .tokenizer import Tokenizer
        return Tokenizer
    if name == "FeatureExtractor":
        from .features import FeatureExtractor
        return FeatureExtractor
    if name == "TokenMemory":
        from .memory import TokenMemory
        return TokenMemory
    if name == "AdaptiveQuadraticNeuron":
        from .adaptive_neuron import AdaptiveQuadraticNeuron
        return AdaptiveQuadraticNeuron
    if name in ("Expert", "ExpertPool"):
        from . import experts as _m
        return getattr(_m, name)
    if name == "MarkovScorer":
        from .markov import MarkovScorer
        return MarkovScorer
    if name == "Router":
        from .router import Router
        return Router
    if name == "RewardEngine":
        from .rewards import RewardEngine
        return RewardEngine
    if name == "ReplayBuffer":
        from .replay import ReplayBuffer
        return ReplayBuffer
    if name == "TeacherClient":
        from .teacher import TeacherClient
        return TeacherClient
    if name == "StudentModel":
        from .student import StudentModel
        return StudentModel
    if name == "TrainingCoordinator":
        from .trainer import TrainingCoordinator
        return TrainingCoordinator
    if name == "simple_tokenize":
        # Backwards compatibility with the pre-2.0 flat module.
        from .tokenizer import tokenize
        return tokenize
    raise AttributeError(name)
