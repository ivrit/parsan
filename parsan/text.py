# -*- coding: utf-8 -*-
"""Raw-text front-end: sentence splitting + word tokenization, and CoNLL-U surface
readers. Produces the (sent_id, text, words, nospace) stream the pipeline consumes,
from either plain text (real use) or a gold CoNLL-U (eval, using its surface words).

`nospace[i]` is True iff the next char was glued (no space) -> drives SpaceAfter=No.
Moved verbatim from predict.py (regexes and the R1 line-preserving contract unchanged).
"""
import re

# A number keeps internal decimal points / thousands separators together (3.14, 60,000,
# 1,000.50 -> one token, matching IAHLT gold). The separator must be FOLLOWED by a digit,
# so a sentence-final period ("...היה 3.") stays a separate punctuation token.
_NUM = r"\d+(?:[.,]\d+)+"
# A word = Hebrew/Latin/digit run, keeping internal geresh/gershayim together
# (so acronyms and abbreviations stay one token); any other non-space char is its own token.
# Inclusive-writing gender-slash (כותב/ת, חבר/ה, הצטרף/י) is kept as ONE word: a base +
# "/" + a short gendered ending at a boundary. Digit/letter slashes (12/2020, א/ב) are
# unaffected, since the ending must be a Hebrew gendered suffix.
_GENDER = "יות|ות|ית|ים|ת|ה|ן|י"
_WORD = (r"[A-Za-z֐-׿0-9]+(?:[\"'׳״][A-Za-z֐-׿0-9]+)*"
         r"(?:/(?:" + _GENDER + r")(?![֐-׿0-9]))?")
# numbers first (greedy) so 3.14 isn't pre-empted by the bare-digit branch of _WORD.
_TOKEN_RE = re.compile(_NUM + r"|" + _WORD + r"|[^\s]", re.UNICODE)
# split after sentence-final punctuation followed by space (3.14 is safe: no space after dot).
_SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=\S)")


def split_sentences(raw, mode="both"):
    blocks = raw.split("\n") if mode in ("newline", "both") else [raw]
    out = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if mode == "newline":
            out.append(b)
        else:
            out.extend(p.strip() for p in _SENT_RE.split(b) if p.strip())
    return out


def tokenize_words(sent):
    """-> list of (word, nospace); nospace = True iff the next char is non-space."""
    out = []
    for m in _TOKEN_RE.finditer(sent):
        end = m.end()
        nospace = end < len(sent) and not sent[end].isspace()
        out.append((m.group(0), nospace))
    return out


def read_text(path, mode):
    """Raw .txt -> (sent_id, text, words, nospace).

    R1: in `newline` mode this is STRICTLY line-preserving -- N input lines yield exactly
    N blocks, in order, including blank lines (the caller emits a placeholder block for
    those). We iterate the file directly (NOT raw.split) to avoid a phantom trailing-
    newline block that would make N+1 blocks."""
    if mode == "newline":
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                toks = tokenize_words(line.rstrip("\n"))
                yield (f"s{i}", line.rstrip("\n"),
                       [w for w, _ in toks], [ns for _, ns in toks])
        return
    raw = open(path, encoding="utf-8").read()
    n = 0
    for sent in split_sentences(raw, mode):
        toks = tokenize_words(sent)
        if toks:                                # punct/both: count is not contractual
            n += 1
            yield f"s{n}", sent, [w for w, _ in toks], [ns for _, ns in toks]


def read_surface(path):
    """Gold CoNLL-U -> (sent_id, text, words, nospace) using MWT ranges as surface words."""
    sid, text, rows = None, None, []

    def flush():
        words, idx = [], 1
        ranges = {}
        for ids, form in rows:
            if "-" in ids:
                a, b = ids.split("-"); ranges[int(a)] = (int(b), form)
        ids_present = [int(x) for x, _ in rows if "-" not in x and "." not in x]
        maxid = max(ids_present) if ids_present else 0
        single = {int(x): f for x, f in rows if "-" not in x and "." not in x}
        while idx <= maxid:
            if idx in ranges:
                b, form = ranges[idx]; words.append(form); idx = b + 1
            else:
                words.append(single.get(idx, "_")); idx += 1
        return words

    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                sid = line.split("=", 1)[1].strip()
            elif line.startswith("# text ="):
                text = line.split("=", 1)[1].strip()
            continue
        if not line:
            if rows:
                w = flush(); yield sid, text, w, [False] * len(w)
            sid, text, rows = None, None, []
            continue
        c = line.split("\t")
        if len(c) == 10 and "." not in c[0]:
            rows.append((c[0], c[1]))
    if rows:
        w = flush(); yield sid, text, w, [False] * len(w)


def normalize_source(src):
    """Map every surface word + the text into the training quote convention (ASCII).
    Word COUNT is preserved, so nospace stays aligned. Applied to both raw-text and
    surface paths."""
    from .normalize import normalize_text
    for sid, text, words, nospace in src:
        yield (sid, normalize_text(text) if text else text,
               [normalize_text(w) for w in words], nospace)
