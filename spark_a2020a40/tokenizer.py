# -*- coding: utf-8 -*-
"""Unicode-aware tokenizer for Greek + Latin, numbers, punctuation and code.

The legacy implementation used ``re.sub(r"[^a-z0-9_+#./:-]+", " ", text)`` after
``str.lower()``, which silently deleted **every** Greek character.  This module
replaces that blacklist with a whitelist-free scanner driven by
``unicodedata.category``: anything the Unicode database calls a letter is a
letter, whether it is ``a``, ``ά`` or ``Ω``.

No external tokenizer package is required; ``unicodedata`` ships with CPython.
"""

import unicodedata

KIND_WORD = "word"
KIND_NUMBER = "number"
KIND_PUNCT = "punct"
KIND_URL = "url"
KIND_OP = "op"
KIND_OTHER = "other"

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = (BOS, EOS, UNK)

# Longest first: the scanner takes the first match.
MULTI_OPS = (
    "**=", "//=", ">>=", "<<=", "...", "!==", "===",
    "==", "!=", "<=", ">=", "->", "=>", ":=", "**", "//", "::",
    "+=", "-=", "*=", "/=", "%=", "&&", "||", "<<", ">>",
)

URL_PREFIXES = ("https://", "http://", "ftp://", "www.")
URL_TRAILING = ".,;:!?)]}\"'»’”"

# Characters that may appear *inside* a word without breaking it.
_WORD_INNER = "_'’-"


def normalize_text(text, casefold=True):
    """NFC normalisation (+ optional casefold).

    NFC is used rather than NFD so that ``ά`` stays a single codepoint and
    ``ord_sum`` is stable across input encodings.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    if casefold:
        # casefold() maps Greek final sigma to sigma, so "λόγος"
        # and "λόγοσ" collapse to the same key.
        text = unicodedata.normalize("NFC", text.casefold())
    return text


def _is_letter(ch):
    return unicodedata.category(ch)[0] in ("L", "M")


def _is_digit(ch):
    return unicodedata.category(ch) == "Nd"


class Token(object):
    __slots__ = ("text", "kind", "start")

    def __init__(self, text, kind, start):
        self.text = text
        self.kind = kind
        self.start = start

    def __repr__(self):
        return "Token({0!r}, {1})".format(self.text, self.kind)

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.text == other.text and self.kind == other.kind
        return NotImplemented

    def __hash__(self):
        return hash((self.text, self.kind))


class Tokenizer(object):
    """Deterministic, dependency-free, Greek-safe tokenizer."""

    __slots__ = ("casefold", "max_token_chars")

    def __init__(self, casefold=True, max_token_chars=32):
        self.casefold = bool(casefold)
        self.max_token_chars = max(4, int(max_token_chars))

    # -- public API ----------------------------------------------------
    def normalize(self, text):
        return normalize_text(text, self.casefold)

    def tokenize(self, text):
        """Return a flat list of token strings."""
        return [t.text for t in self.tokenize_typed(text)]

    def tokenize_typed(self, text):
        """Return a list of :class:`Token` (text + kind + offset)."""
        text = normalize_text(text, self.casefold)
        out = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue

            consumed = self._scan_url(text, i, out)
            if consumed:
                i += consumed
                continue

            if _is_letter(ch) or ch == "_":
                i = self._scan_word(text, i, out)
                continue

            if _is_digit(ch):
                i = self._scan_number(text, i, out)
                continue

            op = self._match_operator(text, i)
            if op is not None:
                out.append(Token(op, KIND_OP, i))
                i += len(op)
                continue

            cat = unicodedata.category(ch)
            kind = KIND_PUNCT if cat[0] in ("P", "S") else KIND_OTHER
            out.append(Token(ch, kind, i))
            i += 1
        return out

    def detokenize(self, tokens):
        """Best-effort inverse -- used when printing student output."""
        parts = []
        no_space_before = set(".,;:!?)]}·…%")
        no_space_after = set("([{")
        for tok in tokens:
            text = tok.text if isinstance(tok, Token) else tok
            if text in SPECIAL_TOKENS:
                continue
            if not parts:
                parts.append(text)
            elif text and text[0] in no_space_before:
                parts.append(text)
            elif parts[-1] and parts[-1][-1] in no_space_after:
                parts.append(text)
            else:
                parts.append(" " + text)
        return "".join(parts).strip()

    # -- scanners ------------------------------------------------------
    def _scan_url(self, text, i, out):
        lowered = text[i:i + 8]
        matched = None
        for prefix in URL_PREFIXES:
            if lowered.startswith(prefix):
                matched = prefix
                break
        if matched is None:
            return 0
        j = i
        n = len(text)
        while j < n and not text[j].isspace():
            j += 1
        raw = text[i:j]
        consumed = len(raw)
        # A sentence-final period belongs to the sentence, not to the URL.
        while len(raw) > len(matched) and raw[-1] in URL_TRAILING:
            raw = raw[:-1]
        self._emit(out, raw, KIND_URL, i)
        for offset, ch in enumerate(text[i + len(raw):j]):
            out.append(Token(ch, KIND_PUNCT, i + len(raw) + offset))
        return consumed

    def _scan_word(self, text, i, out):
        n = len(text)
        j = i
        while j < n:
            ch = text[j]
            if _is_letter(ch) or _is_digit(ch) or ch == "_":
                j += 1
                continue
            if ch in _WORD_INNER and j + 1 < n and (_is_letter(text[j + 1]) or _is_digit(text[j + 1])):
                j += 1
                continue
            break
        self._emit(out, text[i:j], KIND_WORD, i)
        return j

    def _scan_number(self, text, i, out):
        n = len(text)
        j = i
        seen_letter = False
        while j < n:
            ch = text[j]
            if _is_digit(ch):
                j += 1
                continue
            if ch in ".," and j + 1 < n and _is_digit(text[j + 1]):
                j += 1
                continue
            if _is_letter(ch) or ch == "_":
                # "3rd", "0x1f", "10px" stay a single token.
                seen_letter = True
                j += 1
                continue
            break
        self._emit(out, text[i:j], KIND_WORD if seen_letter else KIND_NUMBER, i)
        return j

    def _match_operator(self, text, i):
        for op in MULTI_OPS:
            if text.startswith(op, i):
                return op
        return None

    def _emit(self, out, raw, kind, start):
        """Append ``raw``, splitting instead of dropping over-long tokens.

        The legacy code threw away anything longer than 32 characters, which
        quietly discarded most URLs and long Greek compounds.
        """
        if not raw:
            return
        limit = self.max_token_chars
        if len(raw) <= limit:
            out.append(Token(raw, kind, start))
            return
        for offset in range(0, len(raw), limit):
            out.append(Token(raw[offset:offset + limit], kind, start + offset))


_DEFAULT = Tokenizer()


def tokenize(text):
    """Module-level convenience wrapper using the default tokenizer."""
    return _DEFAULT.tokenize(text)
