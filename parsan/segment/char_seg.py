# -*- coding: utf-8 -*-
"""Sentence-context character segmenter on dicta-il/dictabert-char.

The whole sentence (space-joined surface words) is fed to a char-level BERT; a per-char
B/I head predicts morpheme-start positions. Word boundaries come for free from the
spaces, so the model only learns the within-word splits -- but now WITH left+right
sentence context, which is exactly what the isolated-word head lacked. Result: 0.9931
perfect-word on IAHLT seg test, beating RFTokenizer (0.9870) and the no-context head
(0.9857).

This module holds the model + data/eval layer (used by scripts/train_segmenter.py) AND
the inference wrapper `CharSegmenter` (the integration that was previously missing), so
`segment(words)` returns pipe-joined morphemes with the same interface as RFTokenizer.
Model/data logic moved from the original seg_ctx_train.py.
"""
import json, os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .. import config

MAXCHARS = 460  # per chunk; < dictabert-char's 512 window after [CLS]/[SEP]


# --------------------------------------------------------------------------- #
# Data layer (training + eval)                                                 #
# --------------------------------------------------------------------------- #
def read_ctx(path):
    """Sentence-grouped seg .tab -> list of sentences; each = list of (form, starts_set)
    where starts_set are within-word morpheme-start char positions (incl 0)."""
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "":
            if cur:
                sents.append(cur); cur = []
            continue
        if "\t" not in line:
            continue
        form, seg = line.split("\t")
        if not form:
            continue
        starts = {0}; c = 0
        for p in seg.split("|")[:-1]:
            c += len(p); starts.add(c)
        cur.append((form, starts))
    if cur:
        sents.append(cur)
    return sents


def chunk_sents(sents):
    """Pack each sentence's words into <=MAXCHARS char chunks (never crossing a
    sentence), so every word keeps in-sentence context. -> list of word-lists."""
    chunks = []
    for s in sents:
        cur, clen = [], 0
        for form, starts in s:
            add = len(form) + (1 if cur else 0)
            if cur and clen + add > MAXCHARS:
                chunks.append(cur); cur, clen = [], 0
                add = len(form)
            cur.append((form, starts)); clen += add
        if cur:
            chunks.append(cur)
    return chunks


def chunk_words(words):
    """Inference variant: pack a list of plain surface words into <=MAXCHARS chunks."""
    chunks, cur, clen = [], [], 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and clen + add > MAXCHARS:
            chunks.append(cur); cur, clen = [], 0
            add = len(w)
        cur.append(w); clen += add
    if cur:
        chunks.append(cur)
    return chunks


def build_charvocab(chunks):
    ch = {"<pad>": 0, "<unk>": 1}
    for c in chunks:
        for x in " ".join(f for f, _ in c):
            if x not in ch:
                ch[x] = len(ch)
    return ch


def _encode(text, tok, charvocab):
    """text -> (input_ids, attn, char2sub, char_ids) for one chunk."""
    L = len(text)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True,
              truncation=True, max_length=MAXCHARS + 8)
    char2sub = [0] * L
    for si, (a, b) in enumerate(enc["offset_mapping"]):
        if b > a:
            for c in range(a, min(b, L)):
                char2sub[c] = si
    char_ids = [charvocab.get(c, 1) for c in text]
    return enc["input_ids"], enc["attention_mask"], char2sub, char_ids


# --------------------------------------------------------------------------- #
# Model (param names match the trained checkpoint -- do not rename)            #
# --------------------------------------------------------------------------- #
class SegModel(nn.Module):
    def __init__(self, enc_name, n_chars, char_dim=64):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc_name)
        H = self.enc.config.hidden_size
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.head = nn.Sequential(nn.Linear(H + char_dim, 256), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(256, 2))

    def forward(self, b):
        h = self.enc(input_ids=b["input_ids"], attention_mask=b["attn"]).last_hidden_state
        idx = b["char2sub"].unsqueeze(-1).expand(-1, -1, h.size(-1))
        char_h = torch.gather(h, 1, idx)
        ce = self.char_emb(b["char_ids"])
        return self.head(torch.cat([char_h, ce], -1))


# --------------------------------------------------------------------------- #
# Inference wrapper -- the missing integration: words -> pipe-segmented morphemes
# --------------------------------------------------------------------------- #
class CharSegmenter:
    def __init__(self, model, tok, charvocab, device):
        self.model = model; self.tok = tok; self.charvocab = charvocab
        self.device = device; self.name = "char"

    @classmethod
    def load(cls, run_dir=None, encoder=None, device=None):
        seg = config.segmenter("char")
        run_dir = run_dir or config.ensure_run(seg["run"])
        encoder = encoder or seg["encoder"]
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        charvocab = json.load(open(os.path.join(run_dir, "charvocab.json"), encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(encoder)
        model = SegModel(encoder, len(charvocab)).to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
        model.eval()
        return cls(model, tok, charvocab, device)

    @torch.no_grad()
    def segment(self, words):
        """words -> list of pipe-joined morpheme strings (one per word), with context.
        Output is in input order; chunks are consumed in order so positions line up."""
        result = []
        for chunk in chunk_words(words):
            text = " ".join(chunk)
            ids, attn, c2s, cids = _encode(text, self.tok, self.charvocab)
            b = dict(input_ids=torch.tensor([ids], device=self.device),
                     attn=torch.tensor([attn], device=self.device),
                     char2sub=torch.tensor([c2s], device=self.device),
                     char_ids=torch.tensor([cids], device=self.device))
            pred = self.model(b).argmax(-1)[0]              # per-char B/I over the chunk
            o = 0
            for w in chunk:
                L = len(w)
                starts = [0] + [k for k in range(1, L) if pred[o + k].item() == 1]
                pieces = [w[a:b] for a, b in zip(starts, starts[1:] + [L])]
                result.append("|".join(pieces) if pieces else w)
                o += L + 1                                  # skip the space
        return result
