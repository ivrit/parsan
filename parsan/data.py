"""Data layer: UD CoNLL-U -> tensorized batches of gold morphemes.

The model operates on the gold-segmented morpheme stream (segmentation is a separate
upstream stage), so we read only integer-id token rows (skip MWT ranges 'n-m' and empty
nodes 'n.m'). Lemmatization is cast as an edit-script classification: each (form, lemma)
becomes a short prefix/suffix rule, so the lemma head predicts a rule index that
reconstructs the lemma and generalizes to unseen forms of the same pattern.

Moved verbatim from the original single-file joint_train.py (behavior unchanged).
"""
import torch
from torch.utils.data import Dataset


def read_conllu(paths):
    """Yield sentences; each = list of dicts for integer-id tokens only
    (skip multiword ranges 'n-m' and empty nodes 'n.m')."""
    sents = []
    for path in paths:
        cur = []
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if not line:
                if cur:
                    sents.append(cur); cur = []
                continue
            c = line.split("\t")
            if len(c) != 10:
                continue
            tid = c[0]
            if "-" in tid or "." in tid:
                continue
            cur.append({
                "id": int(tid), "form": c[1], "lemma": c[2],
                "upos": c[3], "xpos": c[4],
                "feats": c[5],                       # raw string; "_" == no feats
                "head": int(c[6]) if c[6] != "_" else 0,
                "deprel": c[7].split(":")[0],        # basic relation (strip subtype)
            })
        if cur:
            sents.append(cur)
    return sents


def gen_lemma_rule(form, lemma):
    """Edit-script (prefix/suffix) rule transforming form -> lemma.
    rule = 'p{p}s{s}+{ins}': keep p prefix chars + ins + s suffix chars.
    Reconstructs the training lemma exactly; generalizes to unseen forms of
    the same morphological pattern (short Hebrew stems post-segmentation)."""
    if lemma == "_" or lemma == "":
        return "IDENT"
    p = 0
    while p < len(form) and p < len(lemma) and form[p] == lemma[p]:
        p += 1
    s = 0
    while s < (len(form) - p) and s < (len(lemma) - p) and form[-1 - s] == lemma[-1 - s]:
        s += 1
    ins = lemma[p: len(lemma) - s] if s > 0 else lemma[p:]
    return f"p{p}s{s}+{ins}"


def apply_lemma_rule(form, rule):
    if rule == "IDENT":
        return form
    try:
        body = rule[1:]
        p_str, rest = body.split("s", 1)
        s_str, ins = rest.split("+", 1)
        p, s = int(p_str), int(s_str)
        if len(form) < p + s:
            return form
        return form[:p] + ins + (form[len(form) - s:] if s > 0 else "")
    except Exception:
        return form


class Vocab:
    def __init__(self):
        self.s2i = {}; self.i2s = []
    def add(self, s):
        if s not in self.s2i:
            self.s2i[s] = len(self.i2s); self.i2s.append(s)
        return self.s2i[s]
    def get(self, s, default=0):
        return self.s2i.get(s, default)
    def __len__(self):
        return len(self.i2s)
    def state(self):
        return self.i2s


def build_vocabs(sents):
    v = {k: Vocab() for k in ["upos", "xpos", "feats", "deprel", "lemma"]}
    for k in v:                       # index 0 reserved as <unk>/fallback
        v[k].add("<unk>")
    for s in sents:
        for t in s:
            for k in ["upos", "xpos", "feats", "deprel"]:
                v[k].add(t[k])
            v["lemma"].add(gen_lemma_rule(t["form"], t["lemma"]))
    return v


def load_vocabs(run_dir):
    """Rebuild the Vocab objects from a run's vocabs.json (inference side)."""
    import json, os
    vstate = json.load(open(os.path.join(run_dir, "vocabs.json")))
    v = {}
    for k, lst in vstate.items():
        vv = Vocab()
        for s in lst:
            vv.add(s)
        v[k] = vv
    return v


class ConlluDataset(Dataset):
    def __init__(self, sents, tokenizer, vocabs, max_len=160):
        self.tok = tokenizer; self.v = vocabs; self.max_len = max_len
        self.ex = [s for s in sents if 1 <= len(s) <= max_len]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        s = self.ex[i]
        forms = [t["form"] for t in s]
        enc = self.tok(forms, is_split_into_words=True,
                       truncation=True, max_length=512,
                       return_tensors=None, add_special_tokens=True)
        word_ids = enc.word_ids()
        # first subword position for each word index
        first_sub = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in first_sub:
                first_sub[wid] = pos
        n = len(forms)
        # if truncation dropped some words, clip the sentence accordingly
        kept = sorted(first_sub.keys())
        if len(kept) != n:
            n = len(kept)
            s = s[:n]
        upos = [self.v["upos"].get(t["upos"]) for t in s]
        xpos = [self.v["xpos"].get(t["xpos"]) for t in s]
        feats = [self.v["feats"].get(t["feats"]) for t in s]
        deprel = [self.v["deprel"].get(t["deprel"]) for t in s]
        lemma = [self.v["lemma"].get(gen_lemma_rule(t["form"], t["lemma"])) for t in s]
        heads = [t["head"] for t in s]               # 0..n, 0 = root
        # clamp heads that point beyond truncated length to 0 (root) defensively
        heads = [h if 0 <= h <= n else 0 for h in heads]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "first_sub": [first_sub[k] for k in kept],   # len n
            "upos": upos, "xpos": xpos, "feats": feats, "lemma": lemma,
            "deprel": deprel, "heads": heads, "n": n,
        }


def collate(batch, pad_id):
    B = len(batch)
    max_sub = max(len(b["input_ids"]) for b in batch)
    max_n = max(b["n"] for b in batch)
    L = max_n + 1                                    # +1 for ROOT at position 0
    input_ids = torch.full((B, max_sub), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_sub), dtype=torch.long)
    first_sub = torch.zeros((B, L), dtype=torch.long)    # 0 unused (ROOT=CLS)
    tok_mask = torch.zeros((B, L), dtype=torch.bool)     # True at real tokens 1..n
    root_mask = torch.zeros((B, L), dtype=torch.bool)    # True at 0..n (valid heads)
    IGN = -100
    upos = torch.full((B, L), IGN, dtype=torch.long)
    xpos = torch.full((B, L), IGN, dtype=torch.long)
    feats = torch.full((B, L), IGN, dtype=torch.long)
    lemma = torch.full((B, L), IGN, dtype=torch.long)
    deprel = torch.full((B, L), IGN, dtype=torch.long)
    heads = torch.full((B, L), IGN, dtype=torch.long)
    for bi, b in enumerate(batch):
        ns = len(b["input_ids"])
        input_ids[bi, :ns] = torch.tensor(b["input_ids"])
        attn[bi, :ns] = torch.tensor(b["attention_mask"])
        n = b["n"]
        root_mask[bi, :n + 1] = True
        for j in range(n):
            first_sub[bi, j + 1] = b["first_sub"][j]     # token j -> position j+1
            tok_mask[bi, j + 1] = True
            upos[bi, j + 1] = b["upos"][j]
            xpos[bi, j + 1] = b["xpos"][j]
            feats[bi, j + 1] = b["feats"][j]
            lemma[bi, j + 1] = b["lemma"][j]
            deprel[bi, j + 1] = b["deprel"][j]
            heads[bi, j + 1] = b["heads"][j]
    return dict(input_ids=input_ids, attn=attn, first_sub=first_sub,
                tok_mask=tok_mask, root_mask=root_mask,
                upos=upos, xpos=xpos, feats=feats, lemma=lemma,
                deprel=deprel, heads=heads)
