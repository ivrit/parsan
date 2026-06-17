"""The joint model: one shared DictaBERT encoder + light heads.

A single encoder pass feeds four classification heads (UPOS, XPOS, FEATS-as-composite,
lemma-rule) and a biaffine dependency parser (arc + labelled-relation), all pooled at
each token's first subword. Parsing is decoded as a single-root maximum spanning
arborescence (see decode.mst_decode).

IMPORTANT: the submodule attribute names here (enc, upos, xpos, feats, lemma, arc_h,
arc_d, rel_h, rel_d, arc_bi, rel_bi) are the parameter names stored in the trained
checkpoints -- DO NOT rename them, or runs/*/best.pt will fail to load.

Moved verbatim from the original single-file joint_train.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim=1, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x, self.bias_y = bias_x, bias_y
        self.W = nn.Parameter(torch.zeros(out_dim, in_dim + bias_x, in_dim + bias_y))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.shape[:-1] + (1,))], -1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.shape[:-1] + (1,))], -1)
        s = torch.einsum("bxi,oij,byj->boxy", x, self.W, y)
        if s.size(1) == 1:
            s = s.squeeze(1)           # [b, x(dep), y(head)]
        else:
            s = s.permute(0, 2, 3, 1)  # [b, dep, head, out]
        return s


class JointModel(nn.Module):
    def __init__(self, encoder_name, vsz, arc_dim=400, rel_dim=100, dropout=0.33):
        super().__init__()
        self.enc = AutoModel.from_pretrained(encoder_name)
        H = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.upos = nn.Linear(H, vsz["upos"])
        self.xpos = nn.Linear(H, vsz["xpos"])
        self.feats = nn.Linear(H, vsz["feats"])
        self.lemma = nn.Linear(H, vsz["lemma"])
        self.arc_h = nn.Sequential(nn.Linear(H, arc_dim), nn.ReLU(), nn.Dropout(dropout))
        self.arc_d = nn.Sequential(nn.Linear(H, arc_dim), nn.ReLU(), nn.Dropout(dropout))
        self.rel_h = nn.Sequential(nn.Linear(H, rel_dim), nn.ReLU(), nn.Dropout(dropout))
        self.rel_d = nn.Sequential(nn.Linear(H, rel_dim), nn.ReLU(), nn.Dropout(dropout))
        self.arc_bi = Biaffine(arc_dim, 1, bias_x=True, bias_y=False)
        self.rel_bi = Biaffine(rel_dim, vsz["deprel"], bias_x=True, bias_y=True)

    def forward(self, batch):
        out = self.enc(input_ids=batch["input_ids"],
                       attention_mask=batch["attn"]).last_hidden_state  # [B,S,H]
        B, L = batch["first_sub"].shape
        idx = batch["first_sub"].unsqueeze(-1).expand(-1, -1, out.size(-1))
        R = torch.gather(out, 1, idx)              # [B,L,H]; row 0 = CLS = ROOT
        R = self.drop(R)
        tag_upos = self.upos(R); tag_xpos = self.xpos(R); tag_feats = self.feats(R)
        tag_lemma = self.lemma(R)
        ah, ad = self.arc_h(R), self.arc_d(R)
        rh, rd = self.rel_h(R), self.rel_d(R)
        arc = self.arc_bi(ad, ah)                  # [B,dep,head]
        rel = self.rel_bi(rd, rh)                  # [B,dep,head,rel]
        # mask invalid head candidates (padding positions)
        head_mask = batch["root_mask"].unsqueeze(1)        # [B,1,head]
        arc = arc.masked_fill(~head_mask, -1e4)             # fp16-safe (AMP); -1e9 overflows Half
        return tag_upos, tag_xpos, tag_feats, tag_lemma, arc, rel


def compute_loss(out, batch):
    tu, tx, tf, tl, arc, rel = out
    ce = lambda logit, gold: F.cross_entropy(
        logit.reshape(-1, logit.size(-1)), gold.reshape(-1), ignore_index=-100)
    l_upos = ce(tu, batch["upos"])
    l_xpos = ce(tx, batch["xpos"])
    l_feats = ce(tf, batch["feats"])
    l_lemma = ce(tl, batch["lemma"])
    l_arc = ce(arc, batch["heads"])
    # rel loss at GOLD head positions
    B, L, _, Rn = rel.shape
    gh = batch["heads"].clamp(min=0)                       # [B,L]
    gh_idx = gh.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, Rn)
    rel_at = torch.gather(rel, 2, gh_idx).squeeze(2)       # [B,L,Rn]
    l_rel = ce(rel_at, batch["deprel"])
    loss = l_upos + l_xpos + l_feats + l_lemma + l_arc + l_rel
    return loss, dict(upos=l_upos.item(), xpos=l_xpos.item(), feats=l_feats.item(),
                      lemma=l_lemma.item(), arc=l_arc.item(), rel=l_rel.item())


@torch.no_grad()
def evaluate(model, loader, device):
    """Token-level accuracy / attachment on gold segmentation (training-time metric)."""
    model.eval()
    n = 0; c = dict(upos=0, xpos=0, feats=0, lemma=0, uas=0, las=0)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        tu, tx, tf, tl, arc, rel = model(batch)
        m = batch["tok_mask"]                              # [B,L]
        pu = tu.argmax(-1); px = tx.argmax(-1); pf = tf.argmax(-1); pl = tl.argmax(-1)
        phead = arc.argmax(-1)                             # [B,L]
        Rn = rel.size(-1)
        ph_idx = phead.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, Rn)
        prel = torch.gather(rel, 2, ph_idx).squeeze(2).argmax(-1)   # [B,L]
        c["upos"] += ((pu == batch["upos"]) & m).sum().item()
        c["xpos"] += ((px == batch["xpos"]) & m).sum().item()
        c["feats"] += ((pf == batch["feats"]) & m).sum().item()
        c["lemma"] += ((pl == batch["lemma"]) & m).sum().item()   # rule-accuracy proxy
        head_ok = (phead == batch["heads"]) & m
        rel_ok = (prel == batch["deprel"]) & m
        c["uas"] += head_ok.sum().item()
        c["las"] += (head_ok & rel_ok).sum().item()
        n += m.sum().item()
    return {k: v / max(n, 1) for k, v in c.items()}, n
