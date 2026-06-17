# -*- coding: utf-8 -*-
"""Lemma lexicon ("bank"): (form,UPOS)->lemma and form->lemma, most-frequent.

Built from IAHLT gold train (CoNLL-U) plus, optionally, Amir Zeldes' heb.lemma lexicon
(word<TAB>tag<TAB>lemma); in-domain train wins ties (weight 2 vs 1). Used at inference as
a lookup with the neural edit-script lemmatizer as backoff. The bank touches ONLY the
lemma column -- it does not change tags or attachments.

Moved from build_lemma_bank.py; the lookup logic is factored out of predict.py.
"""
import json
from collections import defaultdict, Counter


def build_bank(paths):
    """paths: CoNLL-U gold (10 cols) and/or heb.lemma lexicon (3 cols: word,tag,lemma)."""
    fu = defaultdict(Counter)   # (form,upos) -> Counter(lemma)
    f = defaultdict(Counter)    # form -> Counter(lemma)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) == 10 and "-" not in c[0] and "." not in c[0]:
                form, lemma, upos = c[1], c[2], c[3]          # conllu gold train
                w = 2                                         # in-domain train wins ties
            elif len(c) == 3:
                form, upos, lemma = c[0], c[1], c[2]          # heb.lemma lexicon
                w = 1
            else:
                continue
            if lemma == "_" or not lemma:
                continue
            fu[(form, upos)][lemma] += w
            f[form][lemma] += w
    return {
        "by_form_upos": {f"{k[0]}\t{k[1]}": v.most_common(1)[0][0] for k, v in fu.items()},
        "by_form": {k: v.most_common(1)[0][0] for k, v in f.items()},
    }


def load_bank(path):
    return json.load(open(path, encoding="utf-8")) if path else None


def lookup(bank, form, upos, fallback):
    """Bank lemma for (form,upos) -> (form) -> the model's edit-script `fallback`."""
    if bank is None:
        return fallback
    key = form + "\t" + upos
    if key in bank["by_form_upos"]:
        return bank["by_form_upos"][key]
    if form in bank["by_form"]:
        return bank["by_form"][form]
    return fallback
