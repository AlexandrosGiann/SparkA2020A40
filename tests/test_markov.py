# -*- coding: utf-8 -*-
"""The backoff Markov backbone: ordering, stopping and topic conditioning."""
import math
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.config import Config
from spark_a2020a40.markov import MarkovScorer
from spark_a2020a40.memory import TokenMemory, context_key
from spark_a2020a40.student import StudentModel
from spark_a2020a40.teacher import OfflineTeacher
from spark_a2020a40.tokenizer import BOS, EOS, UNK, Tokenizer
from spark_a2020a40.trainer import TrainingCoordinator

CORPUS = [
    "γεια σου χαίρομαι που σε βλέπω",
    "γεια σου φίλε",
    "είμαι καλά ευχαριστώ",
]


def memory_with(corpus=CORPUS, cfg=None, anchor=True):
    cfg = cfg or Config()
    memory = TokenMemory(cfg)
    tokenizer = Tokenizer()
    for sentence in corpus:
        memory.observe_sequence(tokenizer.tokenize_typed(sentence), anchor=anchor)
    return cfg, memory


class TestAnchoring(unittest.TestCase):
    def test_anchoring_adds_bos_and_eos(self):
        cfg, memory = memory_with()
        self.assertGreater(memory.get(BOS).count, 0)
        self.assertGreater(memory.get(EOS).count, 0)

    def test_bos_learns_how_answers_start(self):
        cfg, memory = memory_with()
        successors = memory.successors(BOS, 1)
        self.assertIn("γεια", successors)
        self.assertIn("είμαι", successors)

    def test_eos_learns_where_answers_stop(self):
        cfg, memory = memory_with()
        self.assertIn(EOS, memory.successors("βλέπω", 1))
        self.assertEqual(memory.context_score("σε", "βλέπω", EOS), 1.0)

    def test_without_anchoring_there_is_no_stop_signal(self):
        """This was the v2.0 bug: answers that could never end."""
        cfg, memory = memory_with(anchor=False)
        self.assertEqual(memory.get(EOS).count, 0)
        self.assertEqual(memory.successors("βλέπω", 1), {})


class TestContexts(unittest.TestCase):
    def setUp(self):
        self.cfg, self.memory = memory_with()

    def test_trigram_context_is_recorded(self):
        self.assertEqual(self.memory.context_score("γεια", "σου", "χαίρομαι"), 0.5)
        self.assertEqual(self.memory.context_score("γεια", "σου", "φίλε"), 0.5)

    def test_context_distinguishes_pairs_not_just_distance(self):
        """pos[1] is a skip-gram; contexts are real trigrams."""
        self.assertGreater(self.memory.context_score(BOS, "γεια", "σου"), 0.0)
        self.assertEqual(self.memory.context_score("είμαι", "καλά", "σου"), 0.0)

    def test_unknown_context_is_zero(self):
        self.assertEqual(self.memory.context_score("α", "β", "γ"), 0.0)

    def test_context_key_is_json_safe(self):
        key = context_key(("γεια", "σου"))
        self.assertIsInstance(key, str)
        self.assertEqual(key.split(""), ["γεια", "σου"])

    def test_contexts_are_bounded(self):
        cfg = Config()
        cfg.max_contexts = 20
        cfg.eviction_batch = 4
        memory = TokenMemory(cfg)
        for index in range(400):
            memory.observe_sequence(["a{0}".format(index), "b{0}".format(index),
                                     "c{0}".format(index)])
        self.assertLessEqual(memory.total_contexts(), cfg.max_contexts)

    def test_successors_per_context_are_bounded(self):
        cfg = Config()
        cfg.max_successors_per_context = 3
        memory = TokenMemory(cfg)
        for index in range(50):
            memory.observe_context("x", "y", "z{0}".format(index))
        self.assertLessEqual(len(memory.context_successors("x", "y")), 3)

    def test_contexts_survive_a_roundtrip(self):
        restored = TokenMemory.from_dict(self.cfg, self.memory.to_dict())
        self.assertEqual(restored.total_contexts(), self.memory.total_contexts())
        self.assertEqual(restored.context_score("γεια", "σου", "χαίρομαι"), 0.5)

    def test_compact_purges_dead_contexts(self):
        del self.memory.tokens["χαίρομαι"]
        self.memory.compact()
        self.assertEqual(self.memory.context_score("γεια", "σου", "χαίρομαι"), 0.0)


class TestBackoff(unittest.TestCase):
    def setUp(self):
        self.cfg, self.memory = memory_with()
        self.scorer = MarkovScorer(self.cfg, self.memory)

    def test_trigram_wins_when_available(self):
        self.assertEqual(self.scorer.score("χαίρομαι", ["γεια", "σου"]), 0.5)

    def test_backs_off_to_bigram(self):
        """"καλά" never follows the pair (γεια, σου), but it does follow "είμαι"."""
        score = self.scorer.score("καλά", ["ευχαριστώ", "είμαι"])
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, self.cfg.backoff_alpha)

    def test_backs_off_to_unigram_for_unseen_continuations(self):
        score = self.scorer.score("ευχαριστώ", ["γεια", "σου"])
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.1)

    def test_score_is_never_zero(self):
        self.assertGreaterEqual(self.scorer.score("τελείως-άγνωστο", ["γεια"]),
                                self.cfg.markov_floor)

    def test_log_score_is_finite(self):
        for candidate in ("χαίρομαι", "άγνωστο", EOS):
            self.assertTrue(math.isfinite(self.scorer.log_score(candidate, ["γεια"])))

    def test_history_context_pads_with_bos(self):
        self.assertEqual(self.scorer.history_context([]), (BOS, BOS))
        self.assertEqual(self.scorer.history_context(["α"]), (BOS, "α"))
        self.assertEqual(self.scorer.history_context(["α", "β", "γ"]), ("β", "γ"))

    def test_order_one_ignores_context(self):
        cfg = Config()
        cfg.markov_order = 1
        scorer = MarkovScorer(cfg, self.memory)
        self.assertEqual(scorer.score("χαίρομαι", ["γεια", "σου"]),
                         scorer.score("χαίρομαι", ["είμαι", "καλά"]))

    def test_perplexity_is_lower_on_seen_text(self):
        tokenizer = Tokenizer()
        seen = self.scorer.perplexity([tokenizer.tokenize(CORPUS[0])])
        unseen = self.scorer.perplexity([tokenizer.tokenize("εντελώς άλλο κείμενο")])
        self.assertLess(seen, unseen)


class TestAssociations(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.memory = TokenMemory(self.cfg)
        self.memory.observe_sequence(["γεια", "σου", "φίλε"], anchor=True)
        self.memory.observe_association(["γεια"], ["γεια", "σου", "φίλε"])
        self.scorer = MarkovScorer(self.cfg, self.memory)

    def test_association_is_recorded(self):
        self.assertGreater(self.memory.association_score(["γεια"], "φίλε"), 0.0)

    def test_unrelated_token_scores_zero(self):
        self.assertEqual(self.memory.association_score(["γεια"], "τίποτα"), 0.0)

    def test_unknown_question_scores_zero(self):
        self.assertEqual(self.memory.association_score(["άγνωστο"], "φίλε"), 0.0)

    def test_bonus_is_zero_without_evidence(self):
        """Absent evidence must not be a penalty.

        As a log-probability this term silently punished <eos>, which never
        appears in an association table -- so answers could not end.
        """
        self.assertEqual(self.scorer.association_bonus(EOS, ["γεια"]), 0.0)
        self.assertEqual(self.scorer.association_bonus("τίποτα", ["γεια"]), 0.0)

    def test_bonus_is_positive_with_evidence(self):
        self.assertGreater(self.scorer.association_bonus("φίλε", ["γεια"]), 0.0)

    def test_associations_are_bounded(self):
        cfg = Config()
        cfg.max_associations_per_token = 4
        memory = TokenMemory(cfg)
        memory.observe_association(["q"], ["a{0}".format(i) for i in range(50)])
        self.assertLessEqual(len(memory.associations["q"]), 4)

    def test_associations_survive_a_roundtrip(self):
        restored = TokenMemory.from_dict(self.cfg, self.memory.to_dict())
        self.assertEqual(restored.association_score(["γεια"], "φίλε"),
                         self.memory.association_score(["γεια"], "φίλε"))


class TestCandidateSet(unittest.TestCase):
    def setUp(self):
        self.cfg, self.memory = memory_with()
        self.scorer = MarkovScorer(self.cfg, self.memory)

    def test_banned_tokens_are_never_offered(self):
        candidates = self.scorer.candidates(["γεια"], ["γεια"], 32)
        self.assertNotIn(BOS, candidates)
        self.assertNotIn(UNK, candidates)

    def test_trigram_continuations_come_first(self):
        candidates = self.scorer.candidates([], ["γεια", "σου"], 32)
        self.assertIn(candidates[0], ("χαίρομαι", "φίλε"))

    def test_candidate_set_is_bounded(self):
        self.assertLessEqual(len(self.scorer.candidates(["γεια"], [], 5)), 5)

    def test_only_known_tokens_are_offered(self):
        for token in self.scorer.candidates(["γεια"], ["γεια"], 32):
            self.assertIn(token, self.memory.tokens)


class TestGenerationQuality(unittest.TestCase):
    """End-to-end: the model must reproduce what it was taught, and stop."""

    def setUp(self):
        self.lessons = [
            ("γεια σου", "γεια σου, χαίρομαι που σε βλέπω"),
            ("τι κάνεις;", "είμαι καλά ευχαριστώ, εσύ τι κάνεις;"),
            ("ποιος είσαι;", "είμαι ένα μικρό μοντέλο που τρέχει στο κινητό"),
        ]
        self.cfg = Config()
        self.student = StudentModel(self.cfg)
        coordinator = TrainingCoordinator(
            self.cfg, self.student, OfflineTeacher(self.cfg))
        for index in range(60):
            prompt, answer = self.lessons[index % len(self.lessons)]
            coordinator.process_turn(prompt, teacher_text=answer)

    def normalise(self, text):
        return " ".join(self.student.tokenizer.tokenize(text))

    def test_greedy_reproduces_the_taught_answer(self):
        for prompt, answer in self.lessons:
            result = self.student.generate(prompt, greedy=True)
            self.assertEqual(self.normalise(result["text"]),
                             self.normalise(answer), prompt)

    def test_generation_stops_by_itself(self):
        for prompt, _ in self.lessons:
            result = self.student.generate(prompt, greedy=True)
            self.assertTrue(result["finished"], prompt)

    def test_answers_are_not_truncated_at_the_token_cap(self):
        for prompt, _ in self.lessons:
            result = self.student.generate(prompt, greedy=True)
            self.assertLess(len(result["tokens"]), self.cfg.max_generated_tokens)

    def test_different_questions_get_different_answers(self):
        answers = set(self.normalise(self.student.generate(p, greedy=True)["text"])
                      for p, _ in self.lessons)
        self.assertEqual(len(answers), len(self.lessons))

    def test_an_unseen_question_still_produces_a_clean_sentence(self):
        result = self.student.generate("κάτι εντελώς άγνωστο", greedy=True)
        self.assertTrue(result["text"].strip())
        self.assertNotIn(BOS, result["tokens"])
        self.assertNotIn(UNK, result["tokens"])

    def test_no_immediate_token_repetition(self):
        for prompt, _ in self.lessons:
            tokens = self.student.generate(prompt, greedy=True)["tokens"]
            for index in range(1, len(tokens)):
                self.assertNotEqual(tokens[index], tokens[index - 1], tokens)

    def test_the_markov_backbone_works_without_any_expert(self):
        """With no experts routed, generation must degrade to a plain trigram
        model -- not to noise."""
        prompt, answer = self.lessons[0]
        input_texts = self.student.tokenizer.tokenize(prompt)
        candidates = self.student.candidate_tokens(input_texts, [])
        scores, _, _ = self.student.score_candidates(
            candidates, [], input_texts, [], 0, learn_norm=False)
        best = candidates[scores.index(max(scores))]
        self.assertEqual(best, "γεια")


if __name__ == "__main__":
    unittest.main()
