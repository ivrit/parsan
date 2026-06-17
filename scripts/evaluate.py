#!/usr/bin/env python3
"""Score system CoNLL-U vs gold with the official conll18_ud_eval (F1x100 table).

  python scripts/evaluate.py --pair wiki gold_wiki.conllu preds_wiki.conllu \
                             --pair knesset gold_kn.conllu preds_kn.conllu
"""
import _bootstrap  # noqa: F401
import argparse
from parsan.score import table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=3, action="append", metavar=("LABEL", "GOLD", "SYS"),
                    required=True)
    a = ap.parse_args()
    print(table([(lbl, g, s) for lbl, g, s in a.pair]))


if __name__ == "__main__":
    main()
