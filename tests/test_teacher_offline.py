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


class WrongModelTeacher(TeacherClient):
    """Server is up and answers /, but the requested model is not installed."""

    def __init__(self, cfg, installed=("llama3.2:latest", "qwen2.5:0.5b")):
        TeacherClient.__init__(self, cfg)
        import io
        import urllib.error
        tags = json.dumps({"models": [{"name": n} for n in installed]})

        class _Opener(object):
            def open(self_inner, request, timeout=None):
                url = request.full_url
                if url.endswith("/"):
                    return FakeResponse("Ollama is running")
                if url.endswith("/api/tags"):
                    return FakeResponse(tags)
                raise urllib.error.HTTPError(
                    url, 404, "Not Found", {},
                    io.BytesIO(json.dumps(
                        {"error": 'model "%s" not found, try pulling it first'
                                  % cfg.ollama_model}).encode()))

        self._opener = _Opener()


class TestDiagnosableFailures(unittest.TestCase):
    """A silent None is useless: the user must be told what went wrong."""

    def setUp(self):
        self.cfg = Config()
        self.cfg.ollama_model = "tinyllama"
        self.teacher = WrongModelTeacher(self.cfg)

    def test_the_server_looks_available(self):
        self.assertTrue(self.teacher.is_available())

    def test_generation_still_fails(self):
        self.assertIsNone(self.teacher.generate("γεια"))

    def test_the_error_names_the_actual_problem(self):
        self.teacher.generate("γεια")
        self.assertIn("not found", self.teacher.last_error)
        self.assertIn("404", self.teacher.last_error)

    def test_models_can_be_listed(self):
        self.assertEqual(sorted(self.teacher.list_models()),
                         ["llama3.2:latest", "qwen2.5:0.5b"])

    def test_check_model_reports_the_mismatch_and_the_alternatives(self):
        ok, message = self.teacher.check_model()
        self.assertFalse(ok)
        self.assertIn("tinyllama", message)
        self.assertIn("llama3.2:latest", message)
        self.assertIn("ollama pull", message)

    def test_check_model_accepts_a_tag_variant(self):
        cfg = Config()
        cfg.ollama_model = "llama3.2"
        teacher = WrongModelTeacher(cfg)
        ok, _ = teacher.check_model()
        self.assertTrue(ok)

    def test_an_http_error_does_not_mark_the_server_offline(self):
        """A 4xx means the server is talking to us; do not stop retrying for
        30 seconds over a bad model name."""
        self.teacher.generate("γεια")
        self.assertTrue(self.teacher.is_available())

    def test_a_network_error_does_mark_the_server_offline(self):
        teacher = UnreachableTeacher(Config())
        teacher._available = True
        teacher.generate("γεια")
        self.assertFalse(teacher._available)

    def test_empty_server_is_reported(self):
        teacher = WrongModelTeacher(self.cfg, installed=())
        ok, message = teacher.check_model()
        self.assertFalse(ok)
        self.assertIn("no models installed", message)


class InstalledModelTeacher(TeacherClient):
    """A server that behaves like Ollama: only *exact* model names work."""

    def __init__(self, cfg, installed):
        TeacherClient.__init__(self, cfg)
        import io
        import urllib.error
        tags = json.dumps({"models": [{"name": n} for n in installed]})

        class _Opener(object):
            def open(self_inner, request, timeout=None):
                url = request.full_url
                if url.endswith("/"):
                    return FakeResponse("Ollama is running")
                if url.endswith("/api/tags"):
                    return FakeResponse(tags)
                name = json.loads(request.data.decode("utf-8"))["model"]
                if name in installed or (name + ":latest") in installed:
                    return FakeResponse(json.dumps({"response": "γεια σου φίλε"}))
                raise urllib.error.HTTPError(
                    url, 404, "Not Found", {},
                    io.BytesIO(json.dumps(
                        {"error": 'model "%s" not found' % name}).encode()))

        self._opener = _Opener()


class TestModelNameResolution(unittest.TestCase):
    """Ollama rejects "aya-expanse" when "aya-expanse:8b" is what is installed."""

    def client(self, installed, wanted):
        cfg = Config()
        cfg.ollama_model = wanted
        return InstalledModelTeacher(cfg, installed)

    def test_bare_name_resolves_to_the_installed_tag(self):
        teacher = self.client(["aya-expanse:8b"], "aya-expanse")
        self.assertEqual(teacher.resolve_model(), "aya-expanse:8b")

    def test_generation_succeeds_with_a_bare_name(self):
        teacher = self.client(["aya-expanse:8b"], "aya-expanse")
        self.assertIsNotNone(teacher.generate("γεια"))

    def test_the_resolved_name_is_what_gets_sent(self):
        teacher = self.client(["aya-expanse:8b"], "aya-expanse")
        teacher.generate("γεια")
        self.assertEqual(teacher._resolved, "aya-expanse:8b")

    def test_latest_is_preferred_when_several_tags_exist(self):
        teacher = self.client(
            ["aya-expanse:8b", "aya-expanse:latest"], "aya-expanse")
        self.assertEqual(teacher.resolve_model(), "aya-expanse:latest")

    def test_exact_name_is_left_alone(self):
        teacher = self.client(["aya-expanse:8b"], "aya-expanse:8b")
        self.assertEqual(teacher.resolve_model(), "aya-expanse:8b")

    def test_unknown_model_does_not_resolve(self):
        teacher = self.client(["aya-expanse:8b"], "tinyllama")
        self.assertIsNone(teacher.resolve_model())
        self.assertIsNone(teacher.generate("γεια"))

    def test_check_model_agrees_with_generation(self):
        """The diagnostic must never claim success where generation fails."""
        cases = [
            (["aya-expanse:8b"], "aya-expanse"),
            (["aya-expanse:latest"], "aya-expanse"),
            (["aya-expanse:8b"], "aya-expanse:8b"),
            (["aya-expanse:8b"], "tinyllama"),
            (["llama3.2:latest", "aya-expanse:8b"], "aya-expanse"),
        ]
        for installed, wanted in cases:
            teacher = self.client(installed, wanted)
            ok, message = teacher.check_model()
            produced = teacher.generate("γεια") is not None
            self.assertEqual(ok, produced,
                             "{0} / {1}: {2}".format(installed, wanted, message))

    def test_resolution_survives_an_unlistable_server(self):
        cfg = Config()
        cfg.ollama_model = "aya-expanse"
        teacher = UnreachableTeacher(cfg)
        self.assertIsNone(teacher.resolve_model())
        self.assertIsNone(teacher.generate("γεια"))


if __name__ == "__main__":
    unittest.main()
