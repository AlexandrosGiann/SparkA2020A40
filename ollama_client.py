# -*- coding: utf-8 -*-
"""Deprecated standalone Ollama helper -- use ``spark_a2020a40.teacher``.

The original version of this file had an empty ``IP_ADDRESS``, ran ``input()``
twice at import time and printed the answer to stdout as a side effect of being
imported.  It is kept as a thin, importable wrapper so existing scripts do not
break; it no longer does anything on import.
"""

import sys

from spark_a2020a40.config import Config
from spark_a2020a40.teacher import TeacherClient


def generate_text(prompt, model=None, host=None, port=None, timeout=None):
    """Return the teacher's answer, or ``None`` when Ollama is unreachable."""
    cfg = Config()
    if host:
        cfg.ollama_host = host
    if port:
        cfg.ollama_port = int(port)
    if model:
        cfg.ollama_model = model
    return TeacherClient(cfg).generate(prompt, timeout=timeout)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    prompt = " ".join(argv) if argv else input("Write something: ")
    answer = generate_text(prompt)
    print(answer if answer is not None else "(teacher unavailable)")
    return 0 if answer is not None else 1


if __name__ == "__main__":
    sys.exit(main())
