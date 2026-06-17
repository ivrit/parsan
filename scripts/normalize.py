#!/usr/bin/env python3
"""Apply surface char normalization (quotes -> ASCII) to a .txt or .conllu.

  python scripts/normalize.py txt    in.txt    out.txt
  python scripts/normalize.py conllu in.conllu out.conllu
"""
import _bootstrap  # noqa: F401
import sys
from parsan.normalize import normalize_conllu_file, normalize_txt_file


def main():
    kind, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    (normalize_conllu_file if kind == "conllu" else normalize_txt_file)(inp, outp)
    print(f"normalized {kind}: {inp} -> {outp}")


if __name__ == "__main__":
    main()
