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


#: Greek letters, used to decide where a final sigma is required.
_GREEK_RANGES = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))


def _is_greek(ch):
    code = ord(ch)
    for low, high in _GREEK_RANGES:
        if low <= code <= high:
            return True
    return False


def restore_final_sigma(text):
    """Render a casefolded Greek token with correct orthography.

    ``casefold()`` maps ``ς`` to ``σ`` so that "λόγος" and "λόγοσ" share one
    memory key -- correct for matching, wrong for display: the user sees "πώσ"
    and "ωσ".  In Greek a word-final sigma is *always* written ``ς``, so the
    surface form can be restored deterministically at render time without
    storing a second copy of every token.
    """
    if not text or "σ" not in text:
        return text
    characters = list(text)
    last = len(characters) - 1
    for index, ch in enumerate(characters):
        if ch != "σ":
            continue
        following = characters[index + 1] if index < last else ""
        # Final position, or followed by anything that is not a Greek letter.
        if not following or not _is_greek(following):
            # ... but only inside a Greek word.
            previous = characters[index - 1] if index > 0 else ""
            if previous and _is_greek(previous):
                characters[index] = "ς"
    return "".join(characters)


LANG_GREEK = "el"
LANG_LATIN = "en"
LANG_NEUTRAL = ""

#: Latin ranges we care about (ASCII + Latin-1/Extended letters).
_LATIN_RANGES = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))


def _is_latin(ch):
    code = ord(ch)
    for low, high in _LATIN_RANGES:
        if low <= code <= high:
            return True
    return False


def token_language(text):
    """``LANG_GREEK``, ``LANG_LATIN`` or ``LANG_NEUTRAL`` for one token.

    Numbers, punctuation, operators and URLs are neutral: they belong to every
    language and must never be penalised by the language filter.
    """
    if not text or text in SPECIAL_TOKENS:
        return LANG_NEUTRAL
    greek = latin = 0
    for ch in text:
        if _is_greek(ch):
            greek += 1
        elif _is_latin(ch):
            latin += 1
    if greek > latin:
        return LANG_GREEK
    if latin > greek:
        return LANG_LATIN
    return LANG_NEUTRAL


def text_language(tokens, default=LANG_NEUTRAL):
    """Majority language of a token sequence, ignoring neutral tokens."""
    greek = latin = 0
    for token in tokens or ():
        text = token.text if hasattr(token, "text") else token
        language = token_language(text)
        if language == LANG_GREEK:
            greek += 1
        elif language == LANG_LATIN:
            latin += 1
    if greek > latin:
        return LANG_GREEK
    if latin > greek:
        return LANG_LATIN
    return default


LANGUAGE_NAMES = {LANG_GREEK: "Greek", LANG_LATIN: "English"}

#: Segment boundaries used when stripping foreign-language asides.
_SEGMENT_OPEN = "([{"
_SEGMENT_CLOSE = ")]}"
_SEGMENT_END = ".!?;·\n"


def split_segments(text):
    """Split text into parenthetical and sentence-level segments.

    Multilingual teachers love to append "(Hello! How can I assist you?)" to a
    Greek answer.  Splitting on brackets and sentence ends is enough to isolate
    those asides so they can be dropped before they poison the n-gram tables.
    """
    segments = []
    current = []
    depth = 0
    for ch in text or "":
        if ch in _SEGMENT_OPEN:
            if current:
                segments.append("".join(current))
                current = []
            depth += 1
            current.append(ch)
            continue
        current.append(ch)
        if ch in _SEGMENT_CLOSE and depth > 0:
            depth -= 1
            segments.append("".join(current))
            current = []
            continue
        if depth == 0 and ch in _SEGMENT_END:
            segments.append("".join(current))
            current = []
    if current:
        segments.append("".join(current))
    return [seg for seg in segments if seg.strip()]


def keep_language(text, language, tokenizer=None):
    """Drop the segments of ``text`` written in a different language.

    Returns the surviving text.  If nothing survives (for example the teacher
    ignored the instruction entirely) the original text is returned unchanged
    -- silently learning nothing would be worse than learning the wrong
    language.
    """
    if not language or not text:
        return text
    tok = tokenizer or _DEFAULT
    kept = []
    for segment in split_segments(text):
        segment_language = text_language(tok.tokenize(segment))
        if segment_language and segment_language != language:
            continue
        kept.append(segment)
    result = "".join(kept).strip()
    return result if result else text


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

    def detokenize(self, tokens, restore_sigma=True):
        """Best-effort inverse -- used when printing student output.

        ``restore_sigma`` fixes the word-final sigma that ``casefold()``
        flattened, so the output reads "πώς" rather than "πώσ".
        """
        parts = []
        no_space_before = set(".,;:!?)]}·…%")
        no_space_after = set("([{")
        for tok in tokens:
            text = tok.text if isinstance(tok, Token) else tok
            if text in SPECIAL_TOKENS:
                continue
            if restore_sigma:
                text = restore_final_sigma(text)
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
