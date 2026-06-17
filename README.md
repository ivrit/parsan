# Parsan

Parsan takes raw Hebrew text and returns Universal Dependencies CoNLL-U: morphological
segmentation, POS (UPOS/XPOS), morphological features, lemmas, and a dependency parse.
One shared DictaBERT encoder does all of it, with a character-level segmenter in front.

The name is the Hebrew word for "analyst" (*parshan*), and it doubles as a pun on *parse*.

It is a rewrite of the morphosyntactic core of [HebPipe](https://github.com/amir-zeldes/HebPipe)
as a single jointly-trained model rather than a pipeline of separate tools. From raw text it
does a little better than HebPipe and quite a bit better than Stanza on the IAHLT treebanks,
in and out of domain. A writeup is in preparation; for now this repository is just the code.

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
```

## Training

The recipe is in `experiments/README.md`. In brief:

```
python scripts/train_joint.py     --profile base --train ... --dev ... --test ...
python scripts/train_segmenter.py --train ... --dev ... --test ...
```

## License and credits

MIT, except `parsan/udeval.py`, which is the official CoNLL-18 UD evaluation script from
UFAL and stays under its own MPL-2.0 license (header preserved).

Parsan stands on other people's work: DictaBERT (Dicta), RFTokenizer and the UD Hebrew-IAHLT
treebanks (Amir Zeldes and colleagues), and the Universal Dependencies ecosystem. Thanks to
Amir Zeldes for the encouragement and inspiration, and to Avner Algom at IAHLT for permission
to use the out-of-domain evaluation data.
