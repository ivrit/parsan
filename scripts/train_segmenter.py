#!/usr/bin/env python3
"""Train the sentence-context character segmenter (beats RFTokenizer).

  python scripts/train_segmenter.py \
      --train data/seg/iahlt_train_ctx.tab --dev data/seg/iahlt_dev_ctx.tab \
      --test data/seg/iahlt_test_ctx.tab --out runs/seg_char_ctx

Scores per-word perfect segmentation (same metric as the legacy segmenter), so the test
number is directly comparable to RFTokenizer 0.9870. The model + data readers live in
parsan.segment.char_seg; the training-only Dataset/collate/eval are here.
"""
import _bootstrap  # noqa: F401
import argparse, json, os, time, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from parsan.segment import char_seg as C


class SegData(Dataset):
    def __init__(self, chunks, tok, charvocab):
        self.chunks = chunks; self.tok = tok; self.cv = charvocab

    def __len__(self): return len(self.chunks)

    def __getitem__(self, i):
        ch = self.chunks[i]
        text = " ".join(f for f, _ in ch)
        L = len(text)
        labels = [-100] * L
        words, o = [], 0
        for form, starts in ch:
            for k in range(len(form)):
                labels[o + k] = 1 if k in starts else 0
            words.append((o, len(form), {k for k in starts if k != 0}))
            o += len(form) + 1
        ids, attn, c2s, cids = C._encode(text, self.tok, self.cv)
        return dict(input_ids=ids, attn=attn, char2sub=c2s, char_ids=cids,
                    labels=labels, n=L, words=words)


def collate(batch, pad_id):
    B = len(batch); maxs = max(len(b["input_ids"]) for b in batch); maxc = max(b["n"] for b in batch)
    input_ids = torch.full((B, maxs), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxs), dtype=torch.long)
    char2sub = torch.zeros((B, maxc), dtype=torch.long)
    char_ids = torch.zeros((B, maxc), dtype=torch.long)
    labels = torch.full((B, maxc), -100, dtype=torch.long)
    for bi, b in enumerate(batch):
        s = len(b["input_ids"]); input_ids[bi, :s] = torch.tensor(b["input_ids"]); attn[bi, :s] = torch.tensor(b["attn"])
        n = b["n"]; char2sub[bi, :n] = torch.tensor(b["char2sub"]); char_ids[bi, :n] = torch.tensor(b["char_ids"])
        labels[bi, :n] = torch.tensor(b["labels"])
    return dict(input_ids=input_ids, attn=attn, char2sub=char2sub, char_ids=char_ids,
                labels=labels, words=[b["words"] for b in batch])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); perfect = n = tp = fp = fn = 0
    for b in loader:
        words = b["words"]
        bb = {k: v.to(device) for k, v in b.items() if k != "words"}
        pred = model(bb).argmax(-1)
        for bi, wlist in enumerate(words):
            pl = pred[bi]
            for (o, wl, gold) in wlist:
                pb = {k for k in range(1, wl) if pl[o + k].item() == 1}
                if pb == gold:
                    perfect += 1
                tp += len(pb & gold); fp += len(pb - gold); fn += len(gold - pb)
                n += 1
    P = tp / (tp + fp) if tp + fp else 1.0; R = tp / (tp + fn) if tp + fn else 1.0
    return dict(perfect=perfect / n, F=2 * P * R / (P + R) if P + R else 0.0, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test", default="")
    ap.add_argument("--model", default="dicta-il/dictabert-char"); ap.add_argument("--out", default="runs/seg_char_ctx")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--enc-lr", type=float, default=2e-5)
    a = ap.parse_args()
    random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    tr = C.chunk_sents(C.read_ctx(a.train)); dv = C.chunk_sents(C.read_ctx(a.dev))
    cv = C.build_charvocab(tr)
    json.dump(cv, open(f"{a.out}/charvocab.json", "w"), ensure_ascii=False)
    print(f"[seg-ctx] train_chunks={len(tr)} dev_chunks={len(dv)} chars={len(cv)} MAXCHARS={C.MAXCHARS}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); pad = tok.pad_token_id or 0
    mk = lambda d, sh: DataLoader(SegData(d, tok, cv), batch_size=a.batch, shuffle=sh, collate_fn=lambda x: collate(x, pad))
    tl, dl = mk(tr, True), mk(dv, False)
    model = C.SegModel(a.model, len(cv)).to(device)
    enc_ids = {id(p) for p in model.enc.parameters()}
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.enc_lr},
                             {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": a.lr}], weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best = -1
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0
        for b in tl:
            bb = {k: v.to(device) for k, v in b.items() if k != "words"}; opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(bb).reshape(-1, 2), bb["labels"].reshape(-1), ignore_index=-100)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        d = evaluate(model, dl, device)
        print(f"[ep {ep:02d}] {time.time()-t0:.0f}s loss={tot/len(tl):.3f}  DEV perfect={d['perfect']:.4f} F={d['F']:.4f}", flush=True)
        if d["perfect"] > best:
            best = d["perfect"]; torch.save(model.state_dict(), f"{a.out}/best.pt"); json.dump(vars(a), open(f"{a.out}/args.json", "w"))
    if a.test:
        te = C.chunk_sents(C.read_ctx(a.test)); tel = mk(te, False)
        model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=device))
        t = evaluate(model, tel, device)
        print(f"[TEST] perfect={t['perfect']:.4f} F={t['F']:.4f} (n={t['n']})  vs RFTokenizer=0.9870", flush=True)
        json.dump(t, open(f"{a.out}/test.json", "w"))


if __name__ == "__main__":
    main()
