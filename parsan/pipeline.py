# -*- coding: utf-8 -*-
"""End-to-end pipeline: raw text (or a gold CoNLL-U's surface words) -> valid UD CoNLL-U.

Stages: normalize -> segment (rftok | char) -> joint tag/parse (one DictaBERT pass,
biaffine + single-root MST) -> edit-script lemma with optional bank override -> emit
CoNLL-U with MWT ranges and SpaceAfter. Honest end-to-end on PREDICTED segmentation.

Two model sizes via `profile` ("tiny" | "base") and two segmenters via `segmenter`
("rftok" | "char"); everything else is shared. Consolidates the original predict.py.
"""
import json, os, sys
import torch

from . import config
from .data import load_vocabs, apply_lemma_rule
from .model import JointModel
from .decode import mst_decode
from .lemma_bank import load_bank, lookup
from .text import read_text, read_surface, normalize_source
from .segment import load_segmenter
from transformers import AutoTokenizer

PLACEHOLDER = "1\t_\t_\tX\tX\t_\t0\tdep\t_\t_\n"


class Pipeline:
    def __init__(self, profile="base", segmenter="rftok",
                 lemma_bank=config.LEMMA_BANK, device=None):
        prof = config.profile(profile)
        self.run_dir = prof["run_dir"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = json.load(open(os.path.join(self.run_dir, "args.json")))
        self.vocabs = load_vocabs(self.run_dir)
        vsz = {k: len(v) for k, v in self.vocabs.items()}
        self.tok = AutoTokenizer.from_pretrained(args["model"])
        self.model = JointModel(args["model"], vsz).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(self.run_dir, "best.pt"), map_location=self.device))
        self.model.eval()
        self.seg = load_segmenter(segmenter, device=self.device)
        self.bank = load_bank(lemma_bank) if lemma_bank and os.path.exists(lemma_bank) else None

    @torch.no_grad()
    def tag_parse(self, morphs):
        """morphs: list of morpheme strings -> per-morpheme prediction dicts."""
        enc = self.tok(morphs, is_split_into_words=True, truncation=True, max_length=512)
        wids = enc.word_ids()
        first = {}
        for pos, wid in enumerate(wids):
            if wid is not None and wid not in first:
                first[wid] = pos
        kept = sorted(first.keys()); n = len(kept)
        if n == 0:
            return []
        L = n + 1
        input_ids = torch.tensor([enc["input_ids"]], device=self.device)
        attn = torch.tensor([enc["attention_mask"]], device=self.device)
        fs = torch.zeros(1, L, dtype=torch.long, device=self.device)
        root_mask = torch.zeros(1, L, dtype=torch.bool, device=self.device)
        root_mask[0, :n + 1] = True
        for j, k in enumerate(kept):
            fs[0, j + 1] = first[k]
        batch = dict(input_ids=input_ids, attn=attn, first_sub=fs, root_mask=root_mask)
        tu, tx, tf, tl, arc, rel = self.model(batch)
        pu, px, pf, pl = (t.argmax(-1)[0] for t in (tu, tx, tf, tl))
        arc_np = arc[0].detach().cpu().float().numpy()
        heads = mst_decode(arc_np, n)
        heads_t = torch.tensor(heads, device=rel.device)
        Rn = rel.size(-1)
        prel = rel[0].gather(1, heads_t.view(-1, 1, 1).expand(-1, 1, Rn)).squeeze(1).argmax(-1)
        V = self.vocabs
        return [dict(
            upos=V["upos"].i2s[pu[j].item()], xpos=V["xpos"].i2s[px[j].item()],
            feats=V["feats"].i2s[pf[j].item()], lemma_rule=V["lemma"].i2s[pl[j].item()],
            head=int(heads[j]), deprel=V["deprel"].i2s[prel[j].item()]) for j in range(1, n + 1)]

    def block(self, sid, text, words, nospace):
        """Produce one CoNLL-U block (string) for a sentence's surface words."""
        pairs = [(w, ns) for w, ns in zip(words, nospace) if w and w != "_"]
        words = [w for w, _ in pairs]; nospace = [ns for _, ns in pairs]
        out = [f"# sent_id = {sid}", f"# text = {text}"]
        if not words:
            out.append(PLACEHOLDER.rstrip("\n")); out.append("")
            return "\n".join(out) + "\n"
        segd = self.seg.segment(words)
        morphs, wcounts = [], []
        for seg in segd:
            parts = seg.split("|") if seg else [""]
            parts = [p for p in parts if p != ""] or [seg]
            wcounts.append(len(parts)); morphs.extend(parts)
        preds = self.tag_parse(morphs)
        while len(preds) < len(morphs):                  # truncation safety
            preds.append(dict(upos="X", xpos="X", feats="_", lemma_rule="IDENT",
                              head=0, deprel="dep"))
        m = 1
        for wi, (w, k) in enumerate(zip(words, wcounts)):
            misc_w = "SpaceAfter=No" if (wi < len(nospace) and nospace[wi]) else "_"
            if k > 1:
                out.append(f"{m}-{m + k - 1}\t{w}\t_\t_\t_\t_\t_\t_\t_\t{misc_w}")
            for _ in range(k):
                p = preds[m - 1]; form = morphs[m - 1]
                lemma = lookup(self.bank, form, p["upos"], apply_lemma_rule(form, p["lemma_rule"]))
                feats = p["feats"] if p["feats"] != "<unk>" else "_"
                xpos = p["xpos"] if p["xpos"] != "<unk>" else "_"
                deprel = p["deprel"] if p["deprel"] != "<unk>" else "dep"
                tok_misc = "_" if k > 1 else misc_w
                out.append(f"{m}\t{form}\t{lemma}\t{p['upos']}\t{xpos}\t{feats}\t"
                           f"{p['head']}\t{deprel}\t_\t{tok_misc}")
                m += 1
        out.append("")
        return "\n".join(out) + "\n"


def run(out, text="", gold="", profile="base", segmenter="rftok",
        lemma_bank=config.LEMMA_BANK, sent="both", normalize=True, device=None):
    """Run the pipeline over a raw .txt (`text`) or a gold CoNLL-U (`gold`) -> `out`."""
    if not (text or gold):
        raise ValueError("provide text= (raw .txt) or gold= (CoNLL-U)")
    pipe = Pipeline(profile=profile, segmenter=segmenter, lemma_bank=lemma_bank, device=device)
    source = read_text(text, sent) if text else read_surface(gold)
    if normalize:
        source = normalize_source(source)
    nsent = nlines = 0
    with open(out, "w", encoding="utf-8") as fout:
        for sid, txt, words, nospace in source:
            nlines += 1
            fout.write(pipe.block(sid, txt, words, nospace))
            nsent += 1
            if nsent % 1000 == 0:
                fout.flush(); print(f"[predict] {nsent} sentences...", flush=True)
    print(f"[predict] wrote {nsent} blocks from {nlines} lines -> {out}", flush=True)
    # R1 hard gate: in the contractual mode, blocks MUST equal input lines.
    if text and sent == "newline" and nsent != nlines:
        sys.stderr.write(f"R1 FAIL: {nsent} blocks != {nlines} lines\n")
        sys.exit(1)
    return nsent
