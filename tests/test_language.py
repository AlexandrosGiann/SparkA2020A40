# -*- coding: utf-8 -*-
"""Answering in the language of the question."""
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config
from spark_a2020a40.student import StudentModel
from spark_a2020a40.teacher import OfflineTeacher
from spark_a2020a40.tokenizer import (LANG_GREEK, LANG_LATIN, LANG_NEUTRAL,
                                      Tokenizer, keep_language, split_segments,
                                      text_language, token_language)
from spark_a2020a40.trainer import TrainingCoordinator

BILINGUAL = [
    ("Γεια σου", "Γεια σου! Πώς μπορώ να σε βοηθήσω σήμερα;"),
    ("Hello", "Hello! How can I help you today?"),
    ("Πώς είσαι;", "Είμαι καλά, ευχαριστώ που ρωτάς."),
    ("How are you?", "I am fine, thanks for asking."),
]

POLLUTED = [
    ("Γεια σου", "Γεια σου! Πώς μπορώ να σε βοηθήσω; (Hello! How can I help?)"),
    ("Πώς είσαι;", "Είμαι καλά, ευχαριστώ. (I am fine, thank you.)"),
]


class TestTokenLanguage(unittest.TestCase):
    def test_greek_tokens(self):
        for text in ("γεια", "συναισθήματα", "πώς"):
            self.assertEqual(token_language(text), LANG_GREEK)

    def test_latin_tokens(self):
        for text in ("hello", "python", "today"):
            self.assertEqual(token_language(text), LANG_LATIN)

    def test_neutral_tokens(self):
        for text in ("2026", "!", ",", "", "<eos>", "3.14"):
            self.assertEqual(token_language(text), LANG_NEUTRAL)

    def test_text_language_is_a_majority_vote(self):
        tokenizer = Tokenizer()
        self.assertEqual(text_language(tokenizer.tokenize("Πώς είσαι σήμερα;")),
                         LANG_GREEK)
        self.assertEqual(text_language(tokenizer.tokenize("How are you today?")),
                         LANG_LATIN)

    def test_a_loanword_does_not_flip_the_language(self):
        tokenizer = Tokenizer()
        self.assertEqual(
            text_language(tokenizer.tokenize("Τι είναι η Python;")), LANG_GREEK)

    def test_neutral_input_uses_the_default(self):
        self.assertEqual(text_language(["123", "!"], default=LANG_LATIN),
                         LANG_LATIN)


class TestSegmentation(unittest.TestCase):
    def test_parenthetical_is_its_own_segment(self):
        segments = split_segments("Γεια σου! (Hello!)")
        self.assertIn("(Hello!)", segments)

    def test_sentences_split(self):
        self.assertEqual(len(split_segments("Ένα. Δύο. Τρία.")), 3)

    def test_greek_question_mark_ends_a_sentence(self):
        self.assertEqual(len(split_segments("Τι κάνεις; Καλά.")), 2)

    def test_empty_text(self):
        self.assertEqual(split_segments(""), [])


class TestKeepLanguage(unittest.TestCase):
    def test_english_aside_is_stripped_from_greek(self):
        cleaned = keep_language(
            "Γεια σου! Πώς μπορώ να βοηθήσω; (Hello! How can I help?)", LANG_GREEK)
        self.assertNotIn("Hello", cleaned)
        self.assertIn("Γεια σου", cleaned)

    def test_greek_aside_is_stripped_from_english(self):
        cleaned = keep_language("Hello there. (Γεια σου)", LANG_LATIN)
        self.assertNotIn("Γεια", cleaned)
        self.assertIn("Hello", cleaned)

    def test_loanwords_survive(self):
        text = "Χρησιμοποίησε την Python 3."
        self.assertEqual(keep_language(text, LANG_GREEK), text)

    def test_nothing_surviving_returns_the_original(self):
        """Learning the wrong language beats learning nothing at all."""
        text = "Hello there"
        self.assertEqual(keep_language(text, LANG_GREEK), text)

    def test_no_language_is_a_no_op(self):
        text = "whatever"
        self.assertEqual(keep_language(text, LANG_NEUTRAL), text)


class TestSystemPrompt(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.coordinator = TrainingCoordinator(
            self.cfg, StudentModel(self.cfg), OfflineTeacher(self.cfg))

    def test_greek_prompt_names_greek(self):
        self.assertIn("Greek", self.coordinator.system_prompt(LANG_GREEK))

    def test_english_prompt_names_english(self):
        self.assertIn("English", self.coordinator.system_prompt(LANG_LATIN))

    def test_it_forbids_parenthetical_translations(self):
        self.assertIn("parenthetical", self.coordinator.system_prompt(LANG_GREEK))

    def test_an_empty_template_disables_it(self):
        self.cfg.teacher_system = ""
        self.assertIsNone(self.coordinator.system_prompt(LANG_GREEK))

    def test_a_template_without_placeholders_is_passed_through(self):
        self.cfg.teacher_system = "Be brief."
        self.assertEqual(self.coordinator.system_prompt(LANG_GREEK), "Be brief.")


class TestEndToEndLanguage(unittest.TestCase):
    def train(self, corpus, turns=120, clean=True):
        cfg = Config()
        cfg.match_language = clean
        student = StudentModel(cfg)
        coordinator = TrainingCoordinator(cfg, student, OfflineTeacher(cfg))
        for index in range(turns):
            prompt, answer = corpus[index % len(corpus)]
            coordinator.process_turn(prompt, teacher_text=answer)
        return student

    def purity(self, student, corpus):
        correct = total = 0
        for prompt, _ in corpus:
            wanted = text_language(student.tokenizer.tokenize(prompt))
            for token in student.generate(prompt, greedy=True)["tokens"]:
                language = token_language(token)
                if not language:
                    continue
                total += 1
                correct += (language == wanted)
        return correct / float(max(1, total))

    def test_a_bilingual_corpus_keeps_each_language_separate(self):
        student = self.train(BILINGUAL)
        self.assertEqual(self.purity(student, BILINGUAL), 1.0)

    def test_greek_question_gets_a_greek_answer(self):
        student = self.train(BILINGUAL)
        answer = student.generate("Γεια σου", greedy=True)["tokens"]
        self.assertNotIn(LANG_LATIN, [token_language(t) for t in answer])

    def test_english_question_gets_an_english_answer(self):
        student = self.train(BILINGUAL)
        answer = student.generate("Hello", greedy=True)["tokens"]
        self.assertNotIn(LANG_GREEK, [token_language(t) for t in answer])

    def test_cleaning_removes_teacher_asides(self):
        student = self.train(POLLUTED)
        self.assertEqual(self.purity(student, POLLUTED), 1.0)

    def test_without_cleaning_the_asides_leak_through(self):
        """Documents why match_language defaults to True."""
        student = self.train(POLLUTED, clean=False)
        self.assertLess(self.purity(student, POLLUTED), 1.0)

    def test_the_turn_reports_the_cleaning(self):
        cfg = Config()
        coordinator = TrainingCoordinator(
            cfg, StudentModel(cfg), OfflineTeacher(cfg))
        report = coordinator.process_turn(
            "Γεια σου", teacher_text="Γεια σου! (Hello!)")
        self.assertEqual(report["language"], LANG_GREEK)
        self.assertIn("teacher_text_cleaned", report)
        self.assertNotIn("Hello", report["teacher_text"])

    def test_answers_still_stop_by_themselves(self):
        student = self.train(POLLUTED)
        for prompt, _ in POLLUTED:
            self.assertTrue(student.generate(prompt, greedy=True)["finished"])


if __name__ == "__main__":
    unittest.main()
