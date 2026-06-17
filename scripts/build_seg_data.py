#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build SENTENCE-GROUPED segmentation tables from the IAHLT UD treebanks.

Identical gold to build_seg_tabs.py (same vendored get_segs conversion, same
wiki+knesset merge, same sentence rejection) -- the ONLY difference is we keep the
sentence boundaries instead of flattening them, so a char segmenter can be trained
WITH sentence context. Sentences are separated by a blank line; words within a
sentence are one `form<TAB>pipe|seg` row each.

Guarantee: stripping the blank lines reproduces iahlt_{split}.tab byte-for-byte
(verified by --verify), so the context gold == the existing flat gold.

Usage:
  python build_seg_tabs_ctx.py --src DIR --out DIR [--verify DIR]
"""

import io, os
from argparse import ArgumentParser


def get_sent_rows(conllu):
    """Vendored from RFTokenizer/conllu2segs.py (Amir Zeldes), refactored to RETURN
    a list of sentences, each a list of (surface, pipe-label) rows. Logic unchanged."""
    super_length = 0
    limit = 10
    sents, words, labels, word = [], [], [], []
    max_len = 0
    lines = conllu.split("\n")
    for l, line in enumerate(lines):
        if "\t" in line:
            fields = line.split("\t")
            if "-" in fields[0]:
                start, end = fields[0].split("-")
                super_length = int(end) - int(start) + 1
            else:
                if super_length > 0:
                    word.append(fields[1])
                    super_length -= 1
                    if super_length == 0:
                        words.append("".join(word))
                        labels.append("|".join(word))
                        if len(word) > max_len:
                            max_len = len(word)
                        word = []
                else:
                    if "SpaceAfter=No" in line and ("ADP\t" in line or "DET\t" in line):
                        done = False
                        word.append(fields[1])
                        counter = 1
                        while not done:
                            if "SpaceAfter" in lines[l + counter] and not ("\t,\t" in lines[l + counter + 1] or "\t.\t" in lines[l + counter + 1]):
                                super_length += 1
                                counter += 1
                            else:
                                super_length += 1
                                done = True
                                if super_length > 10:
                                    print(l); quit()
                    else:
                        words.append(fields[1])
                        labels.append(fields[1])
        elif len(line) == 0 and len(words) > 0:
            if max_len > limit or " " in "".join(words):  # reject sentence (identical rule)
                max_len = 0
            else:
                sents.append([(w, lab) for w, lab in zip(words, labels)])
            words, labels = [], []
    return sents


def main():
    p = ArgumentParser()
    p.add_argument("--src", default="data/ud")
    p.add_argument("--out", default="data/seg")
    p.add_argument("--verify", default="", help="dir with existing flat iahlt_*.tab to diff against")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    splits = {
        "train": ["he_iahltwiki-ud-train.conllu", "he_iahltknesset-ud-train.conllu"],
        "dev":   ["he_iahltwiki-ud-dev.conllu",   "he_iahltknesset-ud-dev.conllu"],
        "test":  ["he_iahltwiki-ud-test.conllu",  "he_iahltknesset-ud-test.conllu"],
    }
    for split, files in splits.items():
        sents = []
        for f in files:
            sents.extend(get_sent_rows(io.open(os.path.join(args.src, f), encoding="utf8").read()))
        # ctx file: sentences separated by a blank line
        blocks = ["\n".join(f"{w}\t{lab}" for w, lab in s) for s in sents]
        ctx = "\n\n".join(blocks) + "\n"
        outp = os.path.join(args.out, f"iahlt_{split}_ctx.tab")
        io.open(outp, "w", encoding="utf8", newline="\n").write(ctx)
        nrows = sum(len(s) for s in sents)
        print(f"== {split}: {len(sents):,} sents, {nrows:,} word-rows -> {outp}")
        if args.verify:
            # flatten (drop blank lines) and diff against the existing flat .tab
            flat = "\n".join(ln for ln in ctx.split("\n") if ln) + "\n"
            ref = io.open(os.path.join(args.verify, f"iahlt_{split}.tab"), encoding="utf8").read()
            ok = flat == ref
            print(f"   verify vs {split}.tab: {'IDENTICAL ✓' if ok else 'MISMATCH ✗'}"
                  + ("" if ok else f"  (flat={len(flat)}B ref={len(ref)}B)"))


if __name__ == "__main__":
    main()
