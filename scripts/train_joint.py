#!/usr/bin/env python3
"""Train the joint tagger+parser. Profile picks the encoder + output dir:

  python scripts/train_joint.py --profile base \
      --train data/ud/he_iahltwiki-ud-train.conllu,data/ud/he_iahltknesset-ud-train.conllu \
      --dev   data/ud/he_iahltwiki-ud-dev.conllu,data/ud/he_iahltknesset-ud-dev.conllu \
      --test  data/ud/he_iahltwiki-ud-test.conllu,data/ud/he_iahltknesset-ud-test.conllu

--profile {tiny,base} sets --encoder/--out from config; override either explicitly.
Training loop moved from the original joint_train.py (hyperparameters unchanged).
"""
import _bootstrap  # noqa: F401
import argparse, json, os, time, random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from parsan import config
from parsan.data import read_conllu, build_vocabs, ConlluDataset, collate
from parsan.model import JointModel, compute_loss, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(config.PROFILES), default="base")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--encoder", default="")           # override profile encoder
    ap.add_argument("--out", default="")               # override profile run dir
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)        # heads
    ap.add_argument("--enc-lr", type=float, default=2e-5)    # encoder
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    prof = config.profile(a.profile)
    model_name = a.encoder or prof["encoder"]
    out = a.out or prof["run_dir"]
    random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    os.makedirs(out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    log(f"[setup] device={device} profile={a.profile} model={model_name} out={out}")

    train_sents = read_conllu(a.train.split(","))
    dev_sents = read_conllu(a.dev.split(","))
    test_sents = read_conllu(a.test.split(",")) if a.test else []
    if a.limit:
        train_sents = train_sents[:a.limit]
        dev_sents = dev_sents[:max(10, a.limit // 5)]
        test_sents = test_sents[:max(10, a.limit // 5)] if test_sents else []
    log(f"[data] train={len(train_sents)} dev={len(dev_sents)} test={len(test_sents)}")

    vocabs = build_vocabs(train_sents)
    vsz = {k: len(v) for k, v in vocabs.items()}
    log(f"[vocab] {vsz}")

    tok = AutoTokenizer.from_pretrained(model_name)
    pad_id = tok.pad_token_id or 0
    mk = lambda ss, sh: DataLoader(
        ConlluDataset(ss, tok, vocabs, a.max_len), batch_size=a.batch,
        shuffle=sh, collate_fn=lambda b: collate(b, pad_id))
    tr = mk(train_sents, True); dv = mk(dev_sents, False)
    te = mk(test_sents, False) if test_sents else None

    model = JointModel(model_name, vsz).to(device)
    enc_ids = {id(p) for p in model.enc.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.enc.parameters()), "lr": a.enc_lr},
        {"params": head_params, "lr": a.lr},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.enc_lr, a.lr], total_steps=len(tr) * a.epochs, pct_start=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # persist the config the checkpoint was trained with (model name is authoritative)
    args_out = dict(vars(a)); args_out["model"] = model_name; args_out["out"] = out
    json.dump({k: v.state() for k, v in vocabs.items()},
              open(f"{out}/vocabs.json", "w"), ensure_ascii=False)
    json.dump(args_out, open(f"{out}/args.json", "w"))

    best_las = -1.0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); agg = {}
        for batch in tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, parts = compute_loss(model(batch), batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v
        nb = len(tr)
        dev, _ = evaluate(model, dv, device)
        msg = (f"[ep {ep:02d}] {time.time()-t0:.0f}s "
               f"loss(upos={agg['upos']/nb:.3f} feats={agg['feats']/nb:.3f} "
               f"lemma={agg['lemma']/nb:.3f} arc={agg['arc']/nb:.3f} rel={agg['rel']/nb:.3f}) || "
               f"DEV upos={dev['upos']:.4f} feats={dev['feats']:.4f} "
               f"lemma={dev['lemma']:.4f} UAS={dev['uas']:.4f} LAS={dev['las']:.4f}")
        log(msg); open(f"{out}/train.log", "a").write(msg + "\n")
        if dev["las"] > best_las:
            best_las = dev["las"]
            torch.save(model.state_dict(), f"{out}/best.pt")
            json.dump(dev, open(f"{out}/best_dev.json", "w"))
            log(f"   * saved best (LAS={best_las:.4f})")

    if te is not None:
        model.load_state_dict(torch.load(f"{out}/best.pt", map_location=device))
        test, nt = evaluate(model, te, device)
        json.dump(test, open(f"{out}/test.json", "w"))
        log(f"[TEST] {json.dumps(test)}  (tokens={nt})")
    log("[done]")


if __name__ == "__main__":
    main()
