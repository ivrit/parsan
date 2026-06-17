# Parsan

Parsan takes raw Hebrew text and returns Universal Dependencies CoNLL-U: morphological
segmentation, POS (UPOS/XPOS), morphological features, lemmas, and a dependency parse.
One shared DictaBERT encoder does all of it, with a character-level segmenter in front.

The name is the Hebrew word for "analyst" (*parshan*), and it doubles as a pun on *parse*.

It is a rewrite of the morphosyntactic core of [HebPipe](https://github.com/amir-zeldes/HebPipe)
as a single jointly-trained model rather than a pipeline of separate tools. From raw text it
does a little better than HebPipe and quite a bit better than Stanza on the IAHLT treebanks,
in and out of domain. A writeup is in preparation; for now this repository is just the code.

## How it compares

End-to-end from raw text, scored against IAHLT gold with the official CoNLL-18 evaluation
(LAS, F1×100). "OOD" is the micro-average over the 500 out-of-domain sentences in
`data/ood/`. Stanza is its combined Hebrew model with the AlephBERTGimmel (BERT) backbone.

| system  | wiki | knesset | OOD |
|---------|:----:|:-------:|:---:|
| **Parsan**  | **92.2** | **88.6** | **89.2** |
| HebPipe | 89.7 | 86.2 | 86.1 |
| Stanza  | 83.1 | 80.7 | 80.5 |

Parsan also leads on the morphology-aware MLAS by a wider margin (e.g. +5.6 over HebPipe
on wiki), and its character segmenter beats RFTokenizer's segmentation (0.9931 vs 0.9870
perfect-word) while running about 1.8× faster.

## Install

```
pip install -e .
```

It needs `torch`, `transformers`, and `networkx`. Segmentation uses the bundled
character segmenter by default (it is both more accurate and faster); the legacy
RFTokenizer front-end is also supported, in which case you need
[RFTokenizer](https://github.com/amir-zeldes/RFTokenizer) checked out and pointed to with
the `PARSAN_RF` environment variable.

## Use

```
python scripts/predict.py --text input.txt --sent newline --profile base --out out.conllu
```

`--profile` is `base` (DictaBERT, best) or `tiny` (DictaBERT-tiny, about 3x faster).
`--segmenter` is `char` (default) or `rftok`. Trained checkpoints live under `runs/`; set
`PARSAN_RUNS` to point somewhere else.

## Layout

```
parsan/        the library: model, data, segmenters, the end-to-end pipeline, scoring
scripts/       command-line entry points
experiments/   the Slurm jobs we ran, with a short README
data/ood/      the out-of-domain evaluation set (released with IAHLT's permission)
```

## Training

The recipe is in `experiments/README.md`. In brief:

```
python scripts/train_joint.py     --profile base --train ... --dev ... --test ...
python scripts/train_segmenter.py --train ... --dev ... --test ...
```

## Lemma bank

The lemma head is a neural edit-script classifier, but most of the lemma accuracy comes
from a lookup bank that overrides it for known forms. The strong bank we use is built from
Amir Zeldes' `heb.lemma` lexicon (shipped with HebPipe) plus the IAHLT training data. We do
not redistribute the bank itself, because the lexicon has its own licensing; build it
yourself with the included script:

```
python scripts/build_lemma_bank.py he_iahltwiki-ud-train.conllu \
       he_iahltknesset-ud-train.conllu heb.lemma  data/lemma_bank.json
```

It also works with the IAHLT training data alone (no external lexicon), which is weaker but
fully open. Point `predict.py` at the result with `--lemma-bank`.

## License and credits

MIT, except `parsan/udeval.py`, which is the official CoNLL-18 UD evaluation script from
UFAL and stays under its own MPL-2.0 license (header preserved).

Parsan stands on other people's work: DictaBERT (Dicta), RFTokenizer and the UD Hebrew-IAHLT
treebanks (Amir Zeldes and colleagues), and the Universal Dependencies ecosystem. Thanks to
Amir Zeldes for the encouragement and inspiration, and to Avner Algom at IAHLT for permission
to use the out-of-domain evaluation data.
