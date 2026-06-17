#!/usr/bin/env python3
"""Build the lemma bank from gold train CoNLL-U and/or the heb.lemma lexicon.

  python scripts/build_lemma_bank.py data/ud/he_iahltwiki-ud-train.conllu \
      data/ud/he_iahltknesset-ud-train.conllu data/heb.lemma  data/lemma_bank_amir.json
"""
import _bootstrap  # noqa: F401
import sys, json
from parsan.lemma_bank import build_bank


def main():
    *paths, out = sys.argv[1:]
    bank = build_bank(paths)
    json.dump(bank, open(out, "w"), ensure_ascii=False)
    print(f"bank: {len(bank['by_form_upos'])} (form,upos) + {len(bank['by_form'])} form entries -> {out}")


if __name__ == "__main__":
    main()
