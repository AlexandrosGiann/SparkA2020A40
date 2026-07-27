# -*- coding: utf-8 -*-
"""Legacy entry point -- kept so ``python spark_a2020a40.py`` keeps working.

The implementation moved into the ``spark_a2020a40/`` package in v2.0.0.  This
shim only forwards to the new CLI; the old memory file is migrated
automatically on first load (see ``spark_a2020a40/persistence.py``).

The historical helper names are re-exported so that any script that did
``from spark_a2020a40 import simple_tokenize`` still runs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spark_a2020a40.cli import run  # noqa: E402
from spark_a2020a40.config import Config  # noqa: E402
from spark_a2020a40.persistence import Persistence, migrate_v1  # noqa: E402
from spark_a2020a40.student import StudentModel  # noqa: E402
from spark_a2020a40.tokenizer import Tokenizer  # noqa: E402

MODEL_FILE = "bittreelm_memory.json"
LOCKED_TOKEN = "<unk>"


def simple_tokenize(text):
    """Deprecated: the old ASCII-only tokenizer deleted every Greek letter.

    Forwards to the Unicode-aware tokenizer instead.
    """
    return Tokenizer().tokenize(text)


def load_model(path=None):
    """Deprecated: returns the migrated v2 state dict."""
    cfg = Config()
    return Persistence(cfg, path).load()


def generate_student_text(state, prompt, max_words=20):
    """Deprecated: builds a student from ``state`` and answers ``prompt``."""
    cfg = Config()
    student = StudentModel(cfg)
    if state:
        student.load_dict(state)
    return student.generate(prompt, max_tokens=max_words)["text"]


def interactive():
    return run([])


if __name__ == "__main__":
    sys.exit(run())
