# -*- coding: utf-8 -*-
"""Example: migrate the original bittreelm_memory.json to the v2 schema.

    python3 examples/migrate_legacy_memory.py bittreelm_memory.json spark_memory.json

The old file is never modified.  Running the CLI without arguments performs the
same migration automatically the first time it starts.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_a2020a40.config import Config  # noqa: E402
from spark_a2020a40.persistence import Persistence, migrate_legacy_file  # noqa: E402
from spark_a2020a40.student import StudentModel  # noqa: E402


def main(argv):
    source = argv[0] if argv else "bittreelm_memory.json"
    destination = argv[1] if len(argv) > 1 else "spark_memory.json"

    if not os.path.exists(source):
        print("not found: {0}".format(source))
        return 1

    with open(source, encoding="utf-8") as handle:
        legacy = json.load(handle)
    print("legacy file : {0}".format(source))
    print("  tokens    : {0}".format(len(legacy.get("tokens") or {})))
    print("  tags      : {0}".format(len(legacy.get("tags") or {})))
    print("  next_id   : {0}".format(legacy.get("next_id")))

    payload = migrate_legacy_file(source)
    state = payload["state"]

    cfg = Config()
    student = StudentModel(cfg)
    student.load_dict(state)

    print("\nmigrated to schema v{0}".format(payload["schema_version"]))
    print("  tokens    : {0}".format(len(student.memory)))
    print("  relations : {0}".format(student.memory.total_relations()))
    print("  legacy tags preserved under state['legacy']['tags']: {0}".format(
        sorted(state.get("legacy", {}).get("tags", {}))))

    for text in sorted(student.memory.tokens)[:5]:
        record = student.memory.get(text)
        print("  {0:<10} id={1} count={2} successors={3}".format(
            text, student.memory.binary_id(text), record.count,
            dict(list(record.pos[0].items())[:3])))

    persistence = Persistence(cfg, destination)
    if persistence.save(student.to_dict(compact=cfg.compact_mode)):
        print("\nwrote {0} ({1} bytes)".format(destination, persistence.size_bytes()))
    else:
        print("\nsave failed: {0}".format(persistence.last_error))
        return 1

    print("student answer to 'hello': {0}".format(student.answer("hello")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
