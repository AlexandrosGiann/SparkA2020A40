# -*- coding: utf-8 -*-
"""Migration from bittreelm_memory.json plus the bounded-memory guarantees."""
import json
import os
import shutil
import tempfile
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config
from spark_a2020a40.memory import TokenMemory
from spark_a2020a40.persistence import (SCHEMA_VERSION, CorruptStateError,
                                        Persistence, PersistenceError,
                                        is_legacy_payload, migrate_v1,
                                        validate_state)
from spark_a2020a40.student import StudentModel

LEGACY = {
    "tokens": {
        "<UNK>": {"id": "0000111", "tags": [], "commonality": 0, "relations": {}},
        "hello": {"id": "0001000", "tags": ["general", "greeting"],
                  "commonality": 3,
                  "relations": {"hi": 2, "how": 2, "you": 2}},
        "hi": {"id": "0001001", "tags": ["greeting"], "commonality": 1,
               "relations": {"how": 1}},
        "how": {"id": "0001010", "tags": ["question"], "commonality": 1,
                "relations": {"you": 1}},
        "you": {"id": "0001011", "tags": [], "commonality": 1, "relations": {}},
    },
    "tags": {"general": "0000000", "question": "0000001", "greeting": "0000100"},
    "next_id": 16,
    "meta": {"name": "BitTreeLM", "max_total_items": 110},
}


class TestLegacyDetection(unittest.TestCase):
    def test_legacy_payload_is_recognised(self):
        self.assertTrue(is_legacy_payload(LEGACY))

    def test_v2_payload_is_not_legacy(self):
        self.assertFalse(is_legacy_payload(
            {"schema_version": 2, "state": {"memory": {"tokens": {}}}}))

    def test_garbage_is_not_legacy(self):
        self.assertFalse(is_legacy_payload([1, 2, 3]))
        self.assertFalse(is_legacy_payload({"nonsense": True}))

    def test_migrating_a_non_legacy_payload_raises(self):
        with self.assertRaises(PersistenceError):
            migrate_v1({"schema_version": 2})


class TestMigrationContent(unittest.TestCase):
    def setUp(self):
        self.payload = migrate_v1(LEGACY)
        self.memory = self.payload["state"]["memory"]

    def test_schema_version_is_current(self):
        self.assertEqual(self.payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(self.payload["migrated_from"], 1)

    def test_every_token_survives(self):
        expected = set(LEGACY["tokens"])
        # The legacy "<UNK>" is folded into the canonical lowercase special.
        expected.discard("<UNK>")
        expected.add("<unk>")
        self.assertEqual(set(self.memory["tokens"]), expected)

    def test_legacy_unk_is_canonicalised_not_duplicated(self):
        self.assertIn("<unk>", self.memory["tokens"])
        self.assertNotIn("<UNK>", self.memory["tokens"])

    def test_binary_ids_are_decoded(self):
        self.assertEqual(self.memory["tokens"]["hello"]["id"], 0b0001000)
        self.assertEqual(self.memory["tokens"]["hi"]["id"], 0b0001001)

    def test_commonality_becomes_the_count(self):
        self.assertEqual(self.memory["tokens"]["hello"]["c"], 3)
        self.assertEqual(self.memory["max_count"], 3)

    def test_relations_become_order_one_statistics(self):
        self.assertEqual(self.memory["tokens"]["hello"]["p1"],
                         {"hi": 2, "how": 2, "you": 2})
        self.assertEqual(self.memory["tokens"]["hello"]["p2"], {})

    def test_legacy_tags_are_preserved(self):
        self.assertEqual(self.payload["state"]["legacy"]["tags"], LEGACY["tags"])
        self.assertEqual(self.payload["state"]["legacy"]["meta"]["name"], "BitTreeLM")

    def test_next_id_does_not_collide(self):
        ids = [entry["id"] for entry in self.memory["tokens"].values()]
        self.assertGreater(self.memory["next_id"], max(ids))

    def test_the_migrated_state_loads_into_a_student(self):
        cfg = Config()
        student = StudentModel(cfg)
        student.load_dict(self.payload["state"])
        self.assertIn("hello", student.memory.tokens)
        self.assertEqual(student.memory.successors("hello", 1)["hi"], 2)
        self.assertGreater(student.memory.commonality("hello"), 0.0)

    def test_the_migrated_student_can_answer(self):
        cfg = Config()
        student = StudentModel(cfg)
        student.load_dict(self.payload["state"])
        self.assertIsInstance(student.answer("hello"), str)


class TestMigrationOnDisk(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.cfg = Config()
        self.legacy_path = os.path.join(self.directory, "bittreelm_memory.json")
        with open(self.legacy_path, "w", encoding="utf-8") as handle:
            json.dump(LEGACY, handle)
        self.path = os.path.join(self.directory, "spark_memory.json")

    def test_legacy_file_is_picked_up_automatically(self):
        state = Persistence(self.cfg, self.path).load()
        self.assertIsNotNone(state)
        self.assertIn("hello", state["memory"]["tokens"])

    def test_the_legacy_file_is_not_modified(self):
        with open(self.legacy_path, encoding="utf-8") as handle:
            original = handle.read()
        Persistence(self.cfg, self.path).load()
        with open(self.legacy_path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), original)

    def test_the_real_repository_memory_migrates(self):
        repository_file = os.path.join(_bootstrap.ROOT, "bittreelm_memory.json")
        if not os.path.exists(repository_file):
            self.skipTest("bittreelm_memory.json is not in the repository")
        with open(repository_file, encoding="utf-8") as handle:
            payload = migrate_v1(json.load(handle))
        student = StudentModel(self.cfg)
        student.load_dict(payload["state"])
        self.assertGreater(len(student.memory), 3)


class TestBoundedMemory(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.max_tokens = 40
        self.cfg.max_relations_per_token = 5
        self.cfg.eviction_batch = 4
        self.memory = TokenMemory(self.cfg)

    def test_token_count_never_exceeds_the_cap(self):
        for index in range(500):
            self.memory.observe("token{0}".format(index))
        self.assertLessEqual(len(self.memory), self.cfg.max_tokens)

    def test_relations_are_bounded_per_token(self):
        for index in range(200):
            self.memory.observe_sequence(["source", "target{0}".format(index)])
        record = self.memory.get("source")
        self.assertIsNotNone(record)
        self.assertLessEqual(len(record.pos[0]), self.cfg.max_relations_per_token)

    def test_learning_continues_after_the_cap(self):
        """The legacy model stopped learning forever at 110 items."""
        for index in range(200):
            self.memory.observe("early{0}".format(index))
        self.memory.observe_sequence(["νέα", "λέξη"])
        self.assertIn("νέα", self.memory.tokens)
        self.assertEqual(self.memory.successors("νέα", 1), {"λέξη": 1})

    def test_special_tokens_are_never_evicted(self):
        for index in range(500):
            self.memory.observe("t{0}".format(index))
        for special in ("<unk>", "<bos>", "<eos>"):
            self.assertIn(special, self.memory.tokens)

    def test_ids_are_reused_after_eviction(self):
        for index in range(200):
            self.memory.observe("x{0}".format(index))
        ids = [record.tid for record in self.memory.tokens.values()]
        self.assertEqual(len(ids), len(set(ids)), "duplicate token ids")
        self.assertLess(max(ids), self.cfg.id_capacity())

    def test_frequent_tokens_survive_eviction(self):
        for _ in range(50):
            self.memory.observe("σημαντικό")
        for index in range(300):
            self.memory.observe("noise{0}".format(index))
        self.assertIn("σημαντικό", self.memory.tokens)

    def test_compact_removes_dangling_relations(self):
        self.memory.observe_sequence(["a", "b"])
        del self.memory.tokens["b"]
        self.assertEqual(self.memory.compact(), 1)
        self.assertEqual(self.memory.get("a").pos[0], {})

    def test_binary_id_export_is_still_available(self):
        self.memory.observe("λέξη")
        binary = self.memory.binary_id("λέξη")
        self.assertEqual(len(binary), self.cfg.id_bits)
        self.assertEqual(int(binary, 2), self.memory.get("λέξη").tid)

    def test_roundtrip(self):
        self.memory.observe_sequence(["γεια", "σου", "κόσμε"], class_bit=1)
        restored = TokenMemory.from_dict(self.cfg, self.memory.to_dict())
        self.assertEqual(set(restored.tokens), set(self.memory.tokens))
        self.assertEqual(restored.successors("γεια", 1),
                         self.memory.successors("γεια", 1))
        self.assertEqual(restored.commonality("γεια"),
                         self.memory.commonality("γεια"))


class TestStateValidation(unittest.TestCase):
    def test_valid_payload_passes(self):
        validate_state({"schema_version": 2, "state": {"memory": {"tokens": {}}}})

    def test_missing_version_is_rejected(self):
        with self.assertRaises(CorruptStateError):
            validate_state({"state": {}})

    def test_future_version_is_rejected(self):
        with self.assertRaises(CorruptStateError):
            validate_state({"schema_version": 99, "state": {}})

    def test_non_object_is_rejected(self):
        with self.assertRaises(CorruptStateError):
            validate_state("not a dict")

    def test_bad_memory_shape_is_rejected(self):
        with self.assertRaises(CorruptStateError):
            validate_state({"schema_version": 2,
                            "state": {"memory": {"tokens": []}}})


if __name__ == "__main__":
    unittest.main()
