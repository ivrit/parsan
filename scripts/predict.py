#!/usr/bin/env python3
"""End-to-end: raw text (or a gold CoNLL-U's surface words) -> UD CoNLL-U.

  python scripts/predict.py --text input.txt --sent newline --profile base --out out.conllu
  python scripts/predict.py --gold gold.conllu --profile base --segmenter char --out preds.conllu

--profile {tiny,base}  --segmenter {rftok,char}  --no-normalize to disable quote normalization.
Thin wrapper over parsan.pipeline.run.
"""
import _bootstrap  # noqa: F401
import argparse
from parsan import config
from parsan import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="")
    ap.add_argument("--gold", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", choices=list(config.PROFILES), default="base")
    ap.add_argument("--segmenter", choices=list(config.SEGMENTERS), default="rftok")
    ap.add_argument("--sent", choices=["both", "newline", "punct"], default="both")
    ap.add_argument("--lemma-bank", default=config.LEMMA_BANK)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    import torch
    device = torch.device("cpu") if a.cpu else None
    pipeline.run(out=a.out, text=a.text, gold=a.gold, profile=a.profile,
                 segmenter=a.segmenter, lemma_bank=a.lemma_bank, sent=a.sent,
                 normalize=not a.no_normalize, device=device)


if __name__ == "__main__":
    main()
