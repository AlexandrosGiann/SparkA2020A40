# -*- coding: utf-8 -*-
"""The tokenizer must never silently delete Greek text again."""
import unicodedata
import unittest

from . import _bootstrap  # noqa: F401
from spark_a2020a40.tokenizer import (KIND_NUMBER, KIND_OP, KIND_PUNCT,
                                      KIND_URL, KIND_WORD, Tokenizer,
                                      normalize_text, restore_final_sigma)


class TestGreek(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_greek_accents_are_preserved(self):
        tokens = self.tokenizer.tokenize("καλημέρα κόσμε")
        self.assertEqual(tokens, ["καλημέρα", "κόσμε"])
        self.assertIn("έ", tokens[0])

    def test_greek_is_not_stripped(self):
        # The legacy regex mapped every Greek character to whitespace.
        text = "Η γλώσσα δεν χάνεται"
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(len(tokens), 4)
        self.assertTrue(all(any(ord(c) > 0x370 for c in t) for t in tokens))

    def test_nfd_and_nfc_agree(self):
        nfc = "καλημέρα"
        nfd = unicodedata.normalize("NFD", nfc)
        self.assertNotEqual(nfc, nfd)
        self.assertEqual(self.tokenizer.tokenize(nfc),
                         self.tokenizer.tokenize(nfd))

    def test_case_folding(self):
        self.assertEqual(self.tokenizer.tokenize("ΓΕΙΑ"),
                         self.tokenizer.tokenize("γεια"))
        self.assertEqual(self.tokenizer.tokenize("Πόλη"),
                         self.tokenizer.tokenize("ΠΌΛΗ".lower()))

    def test_final_sigma_folds(self):
        # "λόγος" and "λόγοσ" must land on the same memory key.
        self.assertEqual(self.tokenizer.tokenize("λόγος"),
                         self.tokenizer.tokenize("λόγοσ"))

    def test_greek_question_mark(self):
        tokens = self.tokenizer.tokenize_typed("Τι κάνεις;")
        # casefold() maps final sigma to sigma on purpose (see
        # test_final_sigma_folds), so "κάνεις" is keyed as "κάνεισ".
        self.assertEqual([t.text for t in tokens], ["τι", "κάνεισ", ";"])
        self.assertEqual(tokens[-1].kind, KIND_PUNCT)


class TestLatinNumbersPunctuation(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_latin_case(self):
        self.assertEqual(self.tokenizer.tokenize("Hello World"),
                         ["hello", "world"])

    def test_numbers_are_kept(self):
        tokens = self.tokenizer.tokenize_typed("έχω 3 μήλα και 2.5 λίτρα")
        texts = [t.text for t in tokens]
        self.assertIn("3", texts)
        self.assertIn("2.5", texts)
        kinds = dict((t.text, t.kind) for t in tokens)
        self.assertEqual(kinds["3"], KIND_NUMBER)
        self.assertEqual(kinds["2.5"], KIND_NUMBER)

    def test_alphanumeric_stays_together(self):
        self.assertEqual(self.tokenizer.tokenize("0x1f 10px 3rd"),
                         ["0x1f", "10px", "3rd"])

    def test_punctuation_is_tokenised(self):
        tokens = self.tokenizer.tokenize("Hello, world! Really?")
        for mark in (",", "!", "?"):
            self.assertIn(mark, tokens)

    def test_mixed_scripts(self):
        tokens = self.tokenizer.tokenize("Python και Ελληνικά 2026")
        self.assertEqual(tokens, ["python", "και", "ελληνικά", "2026"])


class TestUrlsAndCode(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(max_token_chars=64)

    def test_url_is_one_token(self):
        tokens = self.tokenizer.tokenize_typed("δες https://example.com/a?b=1 τώρα")
        urls = [t for t in tokens if t.kind == KIND_URL]
        self.assertEqual(len(urls), 1)
        self.assertEqual(urls[0].text, "https://example.com/a?b=1")

    def test_trailing_period_is_not_part_of_url(self):
        tokens = self.tokenizer.tokenize_typed("see http://a.example.org.")
        self.assertEqual(tokens[-2].text, "http://a.example.org")
        self.assertEqual(tokens[-1].text, ".")

    def test_www_url(self):
        tokens = self.tokenizer.tokenize_typed("www.example.com works")
        self.assertEqual(tokens[0].kind, KIND_URL)

    def test_simple_python_code(self):
        tokens = self.tokenizer.tokenize("def add(x, y): return x + y")
        self.assertEqual(
            tokens, ["def", "add", "(", "x", ",", "y", ")", ":", "return",
                     "x", "+", "y"])

    def test_multi_character_operators(self):
        tokens = self.tokenizer.tokenize_typed("if x != y and a ** 2 >= b:")
        ops = [t.text for t in tokens if t.kind == KIND_OP]
        self.assertIn("!=", ops)
        self.assertIn("**", ops)
        self.assertIn(">=", ops)

    def test_underscore_identifiers(self):
        self.assertEqual(self.tokenizer.tokenize("my_var_2 = 3"),
                         ["my_var_2", "=", "3"])


class TestBoundsAndHelpers(unittest.TestCase):
    def test_long_tokens_are_split_not_dropped(self):
        tokenizer = Tokenizer(max_token_chars=8)
        tokens = tokenizer.tokenize("α" * 20)
        self.assertEqual("".join(tokens), "α" * 20)
        self.assertTrue(all(len(t) <= 8 for t in tokens))

    def test_empty_and_whitespace(self):
        tokenizer = Tokenizer()
        self.assertEqual(tokenizer.tokenize(""), [])
        self.assertEqual(tokenizer.tokenize("   \n\t "), [])

    def test_normalize_text_is_idempotent(self):
        once = normalize_text("ΚΑΛΗΜΈΡΑ")
        self.assertEqual(once, normalize_text(once))

    def test_detokenize_roundtrip(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("γεια σου, κόσμε!")
        self.assertEqual(tokenizer.detokenize(tokens), "γεια σου, κόσμε!")

    def test_detokenize_skips_special_tokens(self):
        from spark_a2020a40.tokenizer import EOS
        self.assertEqual(Tokenizer().detokenize(["γεια", EOS]), "γεια")

    def test_word_kind(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize_typed("λέξη")
        self.assertEqual(tokens[0].kind, KIND_WORD)


class TestFinalSigmaRendering(unittest.TestCase):
    """casefold() flattens ς to σ for matching; display must undo it."""

    def test_word_final_sigma_is_restored(self):
        for folded, expected in (("ωσ", "ως"), ("πώσ", "πώς"),
                                 ("λόγοσ", "λόγος"), ("κάνεισ", "κάνεις"),
                                 ("ερωτήσεισ", "ερωτήσεις"), ("μασ", "μας")):
            self.assertEqual(restore_final_sigma(folded), expected)

    def test_internal_sigma_is_left_alone(self):
        for text in ("σήμερα", "συναισθήματα", "κόσμος".casefold(), "σοσ"):
            restored = restore_final_sigma(text)
            self.assertEqual(restored.count("ς"), 1 if text.endswith("σ") else 0,
                             text + " -> " + restored)

    def test_non_greek_is_untouched(self):
        for text in ("hello", "class", "os", "https://a.com"):
            self.assertEqual(restore_final_sigma(text), text)

    def test_lone_sigma_is_untouched(self):
        self.assertEqual(restore_final_sigma("σ"), "σ")

    def test_detokenize_renders_correct_greek(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Πώς είσαι; Ως τεχνητή νοημοσύνη.")
        rendered = tokenizer.detokenize(tokens)
        self.assertIn("πώς", rendered)
        self.assertIn("ως", rendered)
        self.assertNotIn("πώσ", rendered)
        self.assertNotIn("ωσ ", rendered)

    def test_detokenize_can_be_asked_not_to_restore(self):
        tokenizer = Tokenizer()
        raw = tokenizer.detokenize(["πώσ"], restore_sigma=False)
        self.assertEqual(raw, "πώσ")

    def test_matching_still_collapses_both_spellings(self):
        """The memory key must stay folded -- only rendering changes."""
        tokenizer = Tokenizer()
        self.assertEqual(tokenizer.tokenize("λόγος"), tokenizer.tokenize("λόγοσ"))


if __name__ == "__main__":
    unittest.main()
