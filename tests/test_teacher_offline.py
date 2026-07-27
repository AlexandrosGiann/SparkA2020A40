# -*- coding: utf-8 -*-
"""The student must keep working when Ollama is unreachable."""
import json
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config
from spark_a2020a40.rewards import RewardEngine
from spark_a2020a40.student import StudentModel
from spark_a2020a40.teacher import OfflineTeacher, TeacherClient, TeacherUnavailable
from spark_a2020a40.trainer import TrainingCoordinator


class UnreachableTeacher(TeacherClient):
    """A TeacherClient whose socket layer always fails."""

    def _fail(self, *args, **kwargs):
        raise OSError("Network is unreachable")

    def __init__(self, cfg):
        TeacherClient.__init__(self, cfg)

        class _Opener(object):
            def open(self_inner, *args, **kwargs):
                raise OSError("Network is unreachable")

        self._opener = _Opener()


class FakeResponse(object):
    def __init__(self, payload):
        self._payload = payload.encode("utf-8")

    def read(self, size=None):
        return self._payload if size is None else self._payload[:size]

    def close(self):
        pass


class ScriptedTeacher(TeacherClient):
    def __init__(self, cfg, answer):
        TeacherClient.__init__(self, cfg)
        response = json.dumps({"response": answer})

        class _Opener(object):
            def open(self_inner, request, timeout=None):
                if request.full_url.endswith("/"):
                    return FakeResponse("Ollama is running")
                return FakeResponse(response)

        self._opener = _Opener()


class TestOfflineTeacher(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_offline_teacher_is_never_available(self):
        teacher = OfflineTeacher(self.cfg)
        self.assertFalse(teacher.is_available())
        self.assertIsNone(teacher.generate("γεια"))

    def test_offline_teacher_can_raise_when_asked(self):
        with self.assertRaises(TeacherUnavailable):
            OfflineTeacher(self.cfg).generate("γεια", raise_on_error=True)

    def test_network_failure_returns_none_instead_of_raising(self):
        teacher = UnreachableTeacher(self.cfg)
        self.assertIsNone(teacher.generate("γεια"))
        self.assertIn("unreachable", teacher.last_error)

    def test_availability_is_cached(self):
        teacher = UnreachableTeacher(self.cfg)
        self.assertFalse(teacher.is_available())
        checked = teacher._checked_at
        self.assertFalse(teacher.is_available())
        self.assertEqual(teacher._checked_at, checked)

    def test_disabled_teacher_short_circuits(self):
        cfg = Config()
        cfg.teacher_enabled = False
        teacher = TeacherClient(cfg)
        self.assertFalse(teacher.is_available())
        self.assertIsNone(teacher.generate("γεια"))

    def test_status_is_reportable(self):
        status = OfflineTeacher(self.cfg).status()
        for key in ("enabled", "url", "model", "available", "calls"):
            self.assertIn(key, status)


class TestStudentWorksOffline(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.student = StudentModel(self.cfg)
        self.student.memory.observe_sequence(
            self.student.tokenizer.tokenize_typed(
                "γεια σου κόσμε πώς είσαι σήμερα φίλε"), class_bit=1)
        self.coordinator = TrainingCoordinator(
            self.cfg, self.student, OfflineTeacher(self.cfg), RewardEngine(self.cfg))

    def test_a_turn_completes_without_a_teacher(self):
        report = self.coordinator.process_turn("γεια σου")
        self.assertFalse(report["teacher_available"])
        self.assertIsNone(report["teacher_text"])
        self.assertIsInstance(report["student_text"], str)

    def test_reward_is_still_computed_offline(self):
        report = self.coordinator.process_turn("γεια σου")
        self.assertIsInstance(report["reward"], float)
        self.assertLessEqual(abs(report["reward"]), self.cfg.reward_clip)

    def test_class_prediction_still_runs_offline(self):
        report = self.coordinator.process_turn("Τι κάνεις;")
        self.assertIn(report["class"]["predicted"], (0, 1))

    def test_the_student_keeps_answering_over_many_offline_turns(self):
        for index in range(15):
            report = self.coordinator.process_turn("γεια σου {0}".format(index))
            self.assertIsInstance(report["student_text"], str)
        self.assertGreaterEqual(self.coordinator.turns, 15)

    def test_answer_never_raises_on_an_empty_memory(self):
        student = StudentModel(Config())
        self.assertIsInstance(student.answer("κάτι εντελώς άγνωστο"), str)

    def test_user_feedback_works_offline(self):
        self.coordinator.process_turn("γεια σου")
        self.assertIsNotNone(self.coordinator.apply_feedback(1))
        self.assertIsNotNone(self.coordinator.apply_feedback(-1))


class TestTeacherWhenAvailable(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.teacher = ScriptedTeacher(self.cfg, "γεια σου κόσμε")

    def test_probe_detects_availability(self):
        self.assertTrue(self.teacher.is_available())

    def test_generate_returns_the_response_field(self):
        self.assertEqual(self.teacher.generate("γεια"), "γεια σου κόσμε")
        self.assertEqual(self.teacher.calls, 1)

    def test_the_turn_uses_the_teacher_and_distils(self):
        student = StudentModel(self.cfg)
        coordinator = TrainingCoordinator(self.cfg, student, self.teacher)
        report = coordinator.process_turn("γεια")
        self.assertTrue(report["teacher_available"])
        self.assertEqual(report["teacher_text"], "γεια σου κόσμε")
        self.assertGreater(report["distill"]["pairs"], 0)
        self.assertIn("κόσμε", student.memory.tokens)

    def test_teacher_tokens_reach_the_memory_relations(self):
        student = StudentModel(self.cfg)
        coordinator = TrainingCoordinator(self.cfg, student, self.teacher)
        coordinator.process_turn("γεια")
        self.assertIn("σου", student.memory.successors("γεια", 1))

    def test_falling_back_mid_session_does_not_break_anything(self):
        student = StudentModel(self.cfg)
        coordinator = TrainingCoordinator(self.cfg, student, self.teacher)
        coordinator.process_turn("γεια")
        coordinator.teacher = OfflineTeacher(self.cfg)
        report = coordinator.process_turn("γεια ξανά")
        self.assertFalse(report["teacher_available"])
        self.assertIsInstance(report["student_text"], str)


if __name__ == "__main__":
    unittest.main()
