# -*- coding: utf-8 -*-
"""Atomic, versioned, recoverable saving."""
import json
import os
import shutil
import tempfile
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config
from spark_a2020a40.persistence import SCHEMA_VERSION, Persistence
from spark_a2020a40.student import StudentModel
from spark_a2020a40.teacher import OfflineTeacher
from spark_a2020a40.trainer import TrainingCoordinator


class PersistenceTestBase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.cfg = Config()
        self.path = os.path.join(self.directory, "spark_memory.json")
        self.persistence = Persistence(self.cfg, self.path)

    def make_coordinator(self):
        student = StudentModel(self.cfg)
        coordinator = TrainingCoordinator(
            self.cfg, student, OfflineTeacher(self.cfg),
            persistence=self.persistence)
        coordinator.process_turn("γεια σου", teacher_text="γεια σου κόσμε")
        return coordinator


class TestSaving(PersistenceTestBase):
    def test_save_writes_a_versioned_payload(self):
        coordinator = self.make_coordinator()
        self.assertTrue(coordinator.save())
        with open(self.path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["profile"], self.cfg.profile)
        self.assertIn("state", payload)

    def test_no_temporary_files_are_left_behind(self):
        self.make_coordinator().save()
        leftovers = [name for name in os.listdir(self.directory)
                     if name.endswith(".tmp") or name.endswith(".bakm")]
        self.assertEqual(leftovers, [])

    def test_a_backup_is_kept(self):
        coordinator = self.make_coordinator()
        coordinator.save()
        coordinator.save()
        self.assertTrue(os.path.exists(self.persistence.backup_path))

    def test_the_file_is_valid_utf8_with_readable_greek(self):
        self.make_coordinator().save()
        with open(self.path, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("γεια", content)

    def test_saving_is_idempotent(self):
        coordinator = self.make_coordinator()
        coordinator.save()
        first = self.persistence.size_bytes()
        coordinator.save()
        self.assertEqual(self.persistence.size_bytes(), first)


class TestLoading(PersistenceTestBase):
    def test_roundtrip_preserves_the_student(self):
        coordinator = self.make_coordinator()
        coordinator.save()
        before = coordinator.stats()

        restored = StudentModel(self.cfg)
        state = Persistence(self.cfg, self.path).load()
        restored.load_dict(state)
        self.assertEqual(len(restored.memory), before["tokens"])
        self.assertEqual(len(restored.pool), before["experts"])
        self.assertIn("κόσμε", restored.memory.tokens)

    def test_expert_weights_survive_the_roundtrip(self):
        coordinator = self.make_coordinator()
        expert = coordinator.student.pool.get(coordinator.student.pool.ids()[0])
        for _ in range(30):
            expert.train_online([0.5] * expert.neuron.n, 1)
        coordinator.save()
        restored = StudentModel(self.cfg)
        restored.load_dict(Persistence(self.cfg, self.path).load())
        self.assertAlmostEqual(
            restored.pool.get(expert.unique_id).predict_proba([0.5] * expert.neuron.n),
            expert.predict_proba([0.5] * expert.neuron.n), places=3)

    def test_router_state_survives(self):
        coordinator = self.make_coordinator()
        for _ in range(20):
            coordinator.student.router.update(0, [0.5] * 10, 1.0)
        coordinator.save()
        restored = StudentModel(self.cfg)
        restored.load_dict(Persistence(self.cfg, self.path).load())
        self.assertAlmostEqual(restored.router.arms[0].value([0.5] * 10),
                               coordinator.student.router.arms[0].value([0.5] * 10),
                               places=3)

    def test_missing_file_returns_none(self):
        self.assertIsNone(Persistence(self.cfg, self.path).load())


class TestRecovery(PersistenceTestBase):
    def test_a_truncated_file_falls_back_to_the_backup(self):
        coordinator = self.make_coordinator()
        coordinator.save()
        coordinator.save()  # now a .bak exists
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write('{"schema_version": 2, "sta')
        state = Persistence(self.cfg, self.path).load()
        self.assertIsNotNone(state)
        self.assertIn("κόσμε", state["memory"]["tokens"])

    def test_a_wrong_shape_file_is_rejected(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 2, "state": {"memory": {"tokens": 5}}}, handle)
        persistence = Persistence(self.cfg, self.path)
        self.assertIsNone(persistence.load())
        self.assertIsNotNone(persistence.last_error)

    def test_a_newer_schema_is_refused_rather_than_misread(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": SCHEMA_VERSION + 5,
                       "state": {"memory": {"tokens": {}}}}, handle)
        self.assertIsNone(Persistence(self.cfg, self.path).load())

    def test_binary_garbage_does_not_raise(self):
        with open(self.path, "wb") as handle:
            handle.write(b"\x00\x01\x02\xff")
        self.assertIsNone(Persistence(self.cfg, self.path).load())

    def test_an_unwritable_path_reports_failure_without_raising(self):
        persistence = Persistence(self.cfg, os.path.join(self.directory, "x", "y", "z"))
        # The directory is created on demand, so this must succeed ...
        self.assertTrue(persistence.save({"memory": {"tokens": {}}}))
        # ... whereas an invalid payload must be reported, not raised.
        self.assertFalse(persistence.save("not a state"))
        self.assertIsNotNone(persistence.last_error)


class TestCompactMode(PersistenceTestBase):
    def test_compact_mode_is_smaller(self):
        coordinator = self.make_coordinator()
        for index in range(20):
            coordinator.process_turn("λέξη{0}".format(index),
                                     teacher_text="απάντηση {0} εδώ".format(index))
        self.cfg.compact_mode = True
        coordinator.save()
        compact_size = self.persistence.size_bytes()

        self.cfg.compact_mode = False
        coordinator.save()
        full_size = self.persistence.size_bytes()
        self.assertLess(compact_size, full_size)

    def test_both_modes_reload(self):
        coordinator = self.make_coordinator()
        for compact in (True, False):
            self.cfg.compact_mode = compact
            self.assertTrue(coordinator.save())
            self.assertIsNotNone(Persistence(self.cfg, self.path).load())


if __name__ == "__main__":
    unittest.main()
