# -*- coding: utf-8 -*-
"""Scoring against gold with the official conll18_ud_eval (character-aligned, so
segmentation/tokenization differences are handled). Returns/prints F1x100 per metric for
one or more (label, gold, system) triples. Factored out of compare_eval.py.
"""
from . import udeval as E

METRICS = ["Tokens", "Words", "UPOS", "XPOS", "UFeats", "Lemmas", "UAS", "LAS", "MLAS", "BLEX"]


def score_pair(gold_path, sys_path):
    """-> {metric: F1x100}."""
    g = E.load_conllu_file(gold_path)
    s = E.load_conllu_file(sys_path)
    ev = E.evaluate(g, s)
    return {m: 100 * ev[m].f1 for m in METRICS}


def table(pairs):
    """pairs: list of (label, gold_path, sys_path). -> formatted string table."""
    rows = []
    for label, gold, sysf in pairs:
        try:
            rows.append((label, score_pair(gold, sysf)))
        except Exception as ex:
            rows.append((label, {"ERR": str(ex)[:60]}))
    w = max((len(r[0]) for r in rows), default=12) + 2
    lines = [f"{'system/domain':<{w}}" + "".join(f"{m:>8}" for m in METRICS)]
    lines.append("-" * len(lines[0]))
    for label, vals in rows:
        if "ERR" in vals:
            lines.append(f"{label:<{w}}  ERROR: {vals['ERR']}")
        else:
            lines.append(f"{label:<{w}}" + "".join(f"{vals[m]:>8.2f}" for m in METRICS))
    return "\n".join(lines)
