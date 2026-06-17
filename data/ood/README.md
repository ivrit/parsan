# Out-of-domain evaluation set

500 Hebrew sentences with full Universal Dependencies gold annotation (segmentation, POS,
features, lemmas, heads, relations), 100 from each of five genres that are **not** in the
public UD Hebrew-IAHLT wiki/knesset splits used for training:

| file | genre |
|------|-------|
| `ood_bagatz`      | legal (court decisions) |
| `ood_geektime`    | tech / startup news |
| `ood_davar`       | news |
| `ood_israelhayom` | news |
| `ood_allrights`   | social rights / NGO |

`ood_all.*` is the five concatenated. Each genre has a `.conllu` (gold) and a `.txt`
(one sentence per line, the raw input). The sentences were sampled from the broader IAHLT
treebank with a fixed seed, deduplicated, and checked disjoint from the training data.

These sentences are released with the kind permission of **IAHLT** (thanks to Avner Algom);
they come from the IAHLT Hebrew treebank and remain the property of IAHLT
(<https://www.iahlt.org>). Please credit IAHLT if you use them.

To reproduce the scores in the paper:

```
python scripts/predict.py --text data/ood/ood_bagatz.txt --sent newline \
       --profile base --out preds_bagatz.conllu
python scripts/evaluate.py --pair bagatz data/ood/ood_bagatz.conllu preds_bagatz.conllu
```
