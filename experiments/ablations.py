
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold, train_test_split
from Bio import SeqIO
from torch_geometric.nn import GATConv, LayerNorm
from tqdm import tqdm

# ---------------- helpers ----------------
def shift_prot_edges(edge, offset):
    return edge if edge.numel() == 0 else edge + offset

def load_sequences_from_fasta(p):
    seqs = {}
    if not os.path.exists(p): return seqs
    with open(p) as f:
        for r in SeqIO.parse(f, "fasta"):
            seqs[r.id] = str(r.seq).upper()
    return seqs

def load_encodings_from_file(p, default_dim=128):
    if not os.path.exists(p): return None
    enc = {}
    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            enc[parts[0]] = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
    return enc

def load_interactions(p):
    triples = []
    if not os.path.exists(p): return triples
    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try: lab = int(parts[2])
                except: continue
                triples.append((parts[1], parts[0], lab))
    return triples

def safe_train_test_split(triples, test_size, random_state, stratify=True):
    from collections import Counter
    labels = [t[2] for t in triples]
    if stratify and len(set(labels)) > 1:
        cnt = Counter(labels)
        if min(cnt.values()) >= 2:
            try:
                return train_test_split(triples, test_size=test_size,
                                        stratify=labels, random_state=random_state, shuffle=True)
            except ValueError: pass
    return train_test_split(triples, test_size=test_size,
                            random_state=random_state, shuffle=True)

# ---------------- graph builders ----------------
def build_weighted_meta_edges(inter, device="cpu"):
    A = inter.to(device)
    rna_rna = torch.mm(A, A.T)
    deg = A.sum(1)
    union = deg.unsqueeze(1) + deg.unsqueeze(0) - rna_rna
    jac = rna_rna / (union + 1e-8); jac.fill_diagonal_(0)
    m = jac > 0.1
    r_e = m.nonzero(as_tuple=False).t().contiguous(); r_w = jac[m]
    pp = torch.mm(A.T, A); dp = A.sum(0)
    up = dp.unsqueeze(1) + dp.unsqueeze(0) - pp
    jp = pp / (up + 1e-8); jp.fill_diagonal_(0)
    mp = jp > 0
    p_e = mp.nonzero(as_tuple=False).t().contiguous(); p_w = jp[mp]
    return r_e, r_w, p_e, p_w

def build_similarity_edges(emb, train_mask, k=10, device="cpu", drop_ratio=0.0):
    n = emb.size(0); idx = torch.where(train_mask)[0]; nt = idx.numel()
    empty = (torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device))
    if n < 2 or nt < 2: return empty
    en = F.normalize(emb, p=2, dim=1); te = en[idx]
    sim = torch.mm(te, te.t()); sim.fill_diagonal_(-2.0)
    k_eff = min(k, nt - 1)
    if k_eff <= 0: return empty
    tv, tp = torch.topk(sim, k_eff, dim=1)
    edges, ws = [], []
    for i in range(nt):
        gi = idx[i].item()
        for j, w in zip(tp[i], tv[i]):
            edges.append((gi, idx[j].item())); ws.append(w.item())
    if not edges: return empty
    ei = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    ew = torch.tensor(ws, dtype=torch.float, device=device)
    if drop_ratio > 0:
        keep = torch.rand(ei.size(1), device=device) >= drop_ratio
        ei, ew = ei[:, keep], ew[keep]
    if ei.size(1) == 0: return empty
    ei = torch.cat([ei, ei.flip(0)], dim=1); ew = torch.cat([ew, ew], dim=0)
    return ei, ew

def build_graph_for_fold(train_triples, rna_emb, prot_emb, rna_ids, prot_ids, device, sim_k=10):
    Nr, Np = len(rna_ids), len(prot_ids)
    inter = torch.zeros(Nr, Np, dtype=torch.float, device=device)
    for r, p, l in train_triples:
        if l == 1: inter[r, p] = 1.0
    rmi, rmw, pmi, pmw = build_weighted_meta_edges(inter, device)
    pmi_s = shift_prot_edges(pmi, Nr)
    rp = []
    for r, p, l in train_triples:
        if l == 1:
            rp.append([r, Nr + p]); rp.append([Nr + p, r])
    rp_e = torch.tensor(rp, dtype=torch.long, device=device).t().contiguous() if rp else torch.empty((2,0), dtype=torch.long, device=device)
    rm = inter.sum(1) > 0; pm = inter.sum(0) > 0
    rse, rsw = build_similarity_edges(rna_emb, rm, k=sim_k, device=device, drop_ratio=0.0)
    pse, psw = build_similarity_edges(prot_emb, pm, k=sim_k, device=device, drop_ratio=0.0)
    return (inter, rmi, rmw, pmi_s, pmw, rse, rsw, shift_prot_edges(pse, Nr), psw, rp_e, rm, pm)

# ---------------- modules ----------------
class ImplicitMetaPathLearner(nn.Module):
    def __init__(self, h, nr, dropout=0.2):
        super().__init__()
        self.h, self.nr = h, nr
        self.rt = nn.ModuleList([nn.Linear(h, h) for _ in range(nr)])
        self.attn = nn.Sequential(nn.Linear(h*nr, h), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(h, nr), nn.Softmax(-1))
        self.drop = nn.Dropout(dropout)
    def forward(self, x, rel_edges, rel_masks):
        dev = x.device; feats = []
        for k, (e, m) in enumerate(zip(rel_edges, rel_masks)):
            if e.size(1) == 0 or m.sum() == 0:
                feats.append(torch.zeros(self.h, device=dev)); continue
            inv = m.nonzero(as_tuple=True)[0]
            feats.append(self.rt[k](x[inv]).mean(0))
        a = self.attn(torch.cat(feats, 0).view(1, -1)).squeeze(0)
        hmeta = (a.unsqueeze(-1) * torch.stack(feats, 0)).sum(0)
        return x + self.drop(hmeta.unsqueeze(0).expand(x.size(0), -1)), a

class MultiRelationalGATLayer(nn.Module):
    def __init__(self, h, nr, heads=4, dropout=0.2):
        super().__init__()
        self.nr = nr
        self.gats = nn.ModuleList([GATConv(h, h, heads=heads, dropout=dropout,
                                           concat=False, edge_dim=1) for _ in range(nr)])
        self.logits = nn.Parameter(torch.ones(nr))
        self.drop = nn.Dropout(dropout); self.norm = LayerNorm(h)
    def forward(self, x, edges, ws=None):
        outs = []
        for k, (e, g) in enumerate(zip(edges, self.gats)):
            if e.size(1) == 0: outs.append(torch.zeros_like(x)); continue
            ea = ws[k].unsqueeze(-1) if (ws is not None and ws[k] is not None) else None
            outs.append(g(x, e, edge_attr=ea))
        st = torch.stack(outs, 0)
        w = (self.logits.view(-1,1,1) * st).sum(0) / self.nr
        out = self.norm(x + self.drop(w))
        return out, F.softmax(self.logits.detach(), 0)

class SimpleGATLayer(nn.Module):
    def __init__(self, h, heads=4, dropout=0.2):
        super().__init__()
        self.gat = GATConv(h, h, heads=heads, dropout=dropout, concat=False)
        self.drop = nn.Dropout(dropout); self.norm = LayerNorm(h)
    def forward(self, x, edge_index, edge_weight=None):
        out = torch.zeros_like(x) if edge_index.size(1) == 0 else self.gat(x, edge_index)
        return self.norm(x + self.drop(out))

class GraphGenerator(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.f1 = nn.Linear(d*2, h); self.f2 = nn.Linear(h, h); self.out = nn.Linear(h, 1)
    def forward(self, xn, xe):
        B, N = xn.size(0), xe.size(0)
        if N == 0: return torch.zeros((B, 0), device=xn.device)
        pair = torch.cat([xn.unsqueeze(1).expand(-1,N,-1), xe.unsqueeze(0).expand(B,-1,-1)], -1)
        h = F.relu(self.f1(pair)); h = F.relu(self.f2(h))
        return torch.sigmoid(self.out(h).squeeze(-1))

def build_soft_edges(cold, targets, probs, device):
    M, N = probs.shape
    if M == 0 or N == 0:
        return torch.empty((2,0), dtype=torch.long, device=device), torch.empty(0, device=device)
    ci = torch.arange(M, device=device).view(-1,1).expand(-1,N).flatten()
    ti = torch.arange(N, device=device).view(1,-1).expand(M,-1).flatten()
    fwd = torch.stack([cold[ci], targets[ti]])
    bwd = torch.stack([targets[ti], cold[ci]])
    w = probs.flatten()
    return torch.cat([fwd, bwd], 1), torch.cat([w, w.clone()])

# ---------------- configurable model ----------------
class EGRLModel(nn.Module):
    def __init__(self, rna_dim, prot_dim, h=128, nr_meta=4,
                 heads=4, layers=2, dropout=0.2,
                 use_meta_path=True, use_multi_rel=True,
                 use_generator=True, use_dynamic_fusion=True):
        super().__init__()
        self.use_meta_path = use_meta_path
        self.use_multi_rel = use_multi_rel
        self.use_generator = use_generator
        self.use_dynamic_fusion = use_dynamic_fusion
        self.rna_proj = nn.Linear(rna_dim, h); self.prot_proj = nn.Linear(prot_dim, h)
        self.meta = ImplicitMetaPathLearner(h, nr_meta, dropout) if use_meta_path else None
        if use_multi_rel:
            self.nr_gat = 5 + (2 if use_generator else 0)
            self.gats = nn.ModuleList([MultiRelationalGATLayer(h, self.nr_gat, heads, dropout) for _ in range(layers)])
        else:
            self.nr_gat = 1
            self.gats = nn.ModuleList([SimpleGATLayer(h, heads, dropout) for _ in range(layers)])
        self.gen = GraphGenerator(h, h) if use_generator else None
        pin = h*4 if use_dynamic_fusion else h*2
        self.predictor = nn.Sequential(nn.Linear(pin, h), nn.LayerNorm(h), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(h, 1))

    def pair_features(self, ro, po, r, p):
        rb, pb = ro[r], po[p]
        if self.use_dynamic_fusion:
            return torch.cat([rb, pb, rb*pb, torch.abs(rb-pb)], -1)
        return torch.cat([rb, pb], -1)

    def forward(self, rna_f, prot_f, rme, rse, pme, pse, rp,
                rmw=None, rsw=None, pmw=None, psw=None, rpw=None,
                cold_r=None, cold_p=None, arm=None, apm=None, trc=None, tpc=None):
        rh = self.rna_proj(rna_f); ph = self.prot_proj(prot_f)
        x = torch.cat([rh, ph], 0); Nr, Np = rh.size(0), ph.size(0); Nt = Nr+Np
        if arm is None:
            arm = torch.zeros(Nt, dtype=torch.bool, device=x.device); arm[:Nr] = True
        if apm is None:
            apm = torch.zeros(Nt, dtype=torch.bool, device=x.device); apm[Nr:] = True
        if self.use_meta_path and self.meta is not None:
            x, _ = self.meta(x, [rme, pme, rse, pse], [arm, apm, arm, apm])
        x_meta = x
        rm_part, pm_part = x_meta[:Nr], x_meta[Nr:]
        sri = srw = spi = spw = None
        if self.use_generator:
            if cold_r is not None and len(cold_r) > 0:
                if tpc is None: tpc = torch.arange(Np, device=x.device)
                pr = self.gen(rm_part[cold_r], pm_part[tpc])
                sri, srw = build_soft_edges(cold_r, tpc + Nr, pr, x.device)
            if cold_p is not None and len(cold_p) > 0:
                if trc is None: trc = torch.arange(Nr, device=x.device)
                pr = self.gen(pm_part[cold_p], rm_part[trc])
                spi, spw = build_soft_edges(cold_p + Nr, trc, pr, x.device)
        if self.use_multi_rel:
            edges = [rme, pme, rse, pse, rp]
            ws = [rmw, pmw, rsw, psw, rpw]
            if self.use_generator:
                edges.append(sri if sri is not None else torch.empty((2,0), dtype=torch.long, device=x.device))
                ws.append(srw)
                edges.append(spi if spi is not None else torch.empty((2,0), dtype=torch.long, device=x.device))
                ws.append(spw)
            for g in self.gats:
                x, _ = g(x, edges, ws)
        else:
            all_e = [e for e in [rme, pme, rse, pse, rp] if e.size(1) > 0]
            if sri is not None: all_e.append(sri)
            if spi is not None: all_e.append(spi)
            comb = torch.cat(all_e, 1) if all_e else torch.empty((2,0), dtype=torch.long, device=x.device)
            for g in self.gats:
                x = g(x, comb, None)
        return x[:Nr], x[Nr:], x_meta

# ---------------- train / eval ----------------
def train_epoch(model, opt, rna_e, prot_e, rme, pme, rp, triples, inter,
                rmw=None, pmw=None, rpw=None,
                bs=256, clip=1.0, lg=0.1, cr=0.05, sim_k=10, sdr=0.1, gamma=1.0, desc="Train"):
    model.train(); random.shuffle(triples)
    dev = rna_e.device; Nr, Np = rna_e.size(0), prot_e.size(0); Nt = Nr+Np
    labs = torch.tensor([t[2] for t in triples], dtype=torch.float, device=dev)
    pos = labs.sum(); neg = len(labs) - pos
    pw = (gamma*neg/pos).clamp(min=1.0) if pos > 0 else torch.tensor(1.0, device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    if model.use_generator and lg > 0 and cr > 0:
        rca = (inter.sum(1) > 0).nonzero(as_tuple=True)[0]
        pca = (inter.sum(0) > 0).nonzero(as_tuple=True)[0]
        cold_r = rca[torch.randperm(rca.numel(), device=dev)[:max(1,int(rca.numel()*cr))]] if rca.numel()>0 else torch.empty(0, dtype=torch.long, device=dev)
        cold_p = pca[torch.randperm(pca.numel(), device=dev)[:max(1,int(pca.numel()*cr))]] if pca.numel()>0 else torch.empty(0, dtype=torch.long, device=dev)
    else:
        cold_r = torch.empty(0, dtype=torch.long, device=dev)
        cold_p = torch.empty(0, dtype=torch.long, device=dev)

    masked = inter.clone()
    if cold_r.numel() > 0: masked[cold_r, :] = 0
    if cold_p.numel() > 0: masked[:, cold_p] = 0
    m_rme, m_rmw, m_pme, m_pmw = build_weighted_meta_edges(masked, dev)
    m_pme = shift_prot_edges(m_pme, Nr)
    cs_r, cs_p = set(cold_r.cpu().tolist()), set(cold_p.cpu().tolist())
    rp_l = []
    for r, p, l in triples:
        if l == 1 and r not in cs_r and p not in cs_p:
            rp_l.append([r, Nr+p]); rp_l.append([Nr+p, r])
    m_rp = torch.tensor(rp_l, dtype=torch.long, device=dev).t().contiguous() if rp_l else torch.empty((2,0), dtype=torch.long, device=dev)
    ret_r = masked.sum(1) > 0; ret_p = masked.sum(0) > 0
    m_rse, m_rsw = build_similarity_edges(rna_e, ret_r, sim_k, dev, sdr)
    m_pse, m_psw = build_similarity_edges(prot_e, ret_p, sim_k, dev, sdr)
    m_pse = shift_prot_edges(m_pse, Nr)
    trc = ret_r.nonzero(as_tuple=True)[0]; tpc = ret_p.nonzero(as_tuple=True)[0]
    arm = torch.zeros(Nt, dtype=torch.bool, device=dev); arm[:Nr] = ret_r
    apm = torch.zeros(Nt, dtype=torch.bool, device=dev); apm[Nr:] = ret_p

    total = 0.0
    for i in tqdm(range(0, len(triples), bs), desc=desc, leave=False, position=2, dynamic_ncols=True):
        batch = triples[i:i+bs]
        ri = [p[0] for p in batch]; pi = [p[1] for p in batch]
        lab = torch.tensor([p[2] for p in batch], dtype=torch.float, device=dev)
        ro, po, xm = model(rna_e, prot_e, m_rme, m_rse, m_pme, m_pse, m_rp,
                           m_rmw, m_rsw, m_pmw, m_psw, None,
                           cold_r, cold_p, arm, apm, trc, tpc)
        logits = model.predictor(model.pair_features(ro, po, ri, pi)).squeeze(-1)
        main = crit(logits, lab)
        gl = torch.tensor(0.0, device=dev)
        if model.use_generator and lg > 0 and (cold_r.numel() > 0 or cold_p.numel() > 0):
            rm_part, pm_part = xm[:Nr], xm[Nr:]; n = 0
            if cold_r.numel() > 0 and tpc.numel() > 0:
                pr = model.gen(rm_part[cold_r], pm_part[tpc])
                gt = inter[cold_r][:, tpc]
                gl = gl + F.binary_cross_entropy(pr, gt, reduction="sum")
                n += cold_r.numel() * tpc.numel()
            if cold_p.numel() > 0 and trc.numel() > 0:
                pr = model.gen(pm_part[cold_p], rm_part[trc])
                gt = inter[:, cold_p][trc, :].T
                gl = gl + F.binary_cross_entropy(pr, gt, reduction="sum")
                n += cold_p.numel() * trc.numel()
            if n > 0: gl = gl / n
        loss = main + lg*gl
        opt.zero_grad(); loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); total += loss.item()*len(batch)
    return total / max(1, len(triples))

@torch.no_grad()
def evaluate(model, rna_e, prot_e, rme, rse, pme, pse, rp, triples, threshold=0.5,
             rmw=None, rsw=None, pmw=None, psw=None, rpw=None,
             cold_r=None, cold_p=None, arm=None, apm=None, trc=None, tpc=None):
    model.eval()
    ro, po, _ = model(rna_e, prot_e, rme, rse, pme, pse, rp,
                      rmw, rsw, pmw, psw, rpw,
                      cold_r, cold_p, arm, apm, trc, tpc)
    scores, labels = [], []
    for r, p, l in triples:
        scores.append(torch.sigmoid(model.predictor(model.pair_features(ro, po, r, p))).item())
        labels.append(l)
    preds = [1 if s >= threshold else 0 for s in scores]
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    return {"auc": auc, "aupr": average_precision_score(labels, scores),
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "mcc": matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0.0}

# ---------------- dataset + cv ----------------
CFG = dict(hidden_dim=128, gat_heads=4, num_gat_layers=2, dropout=0.2,
           lr=1e-4, weight_decay=1e-4, batch_size=256, clip_grad=1.0,
           sim_k=10, sim_drop_ratio=0.1, lambda_gen=0.1, cold_ratio=0.05,
           epochs=300, gamma=1.0, threshold=0.5)

DATASETS = {
    "NPInter2": dict(protein_fasta="../data/sequence/NPInter_protein_seq.fa",
                     rna_fasta="../data/sequence/NPInter_rna_seq.fa",
                     protein_enc="../data/encode_LM/NPInter_protein_encoded.txt",
                     rna_enc="../data/encode_LM/NPInter_RNA_encoded.txt",
                     interactions="../data/NPInter_pairs.txt"),
}

def load_dataset(paths, device):
    rna_seqs = load_sequences_from_fasta(paths["rna_fasta"])
    prot_seqs = load_sequences_from_fasta(paths["protein_fasta"])
    rna_ids, prot_ids = list(rna_seqs.keys()), list(prot_seqs.keys())
    rna_enc = load_encodings_from_file(paths["rna_enc"], 128)
    prot_enc = load_encodings_from_file(paths["protein_enc"], 128)
    enc = {}
    if prot_enc: enc.update(prot_enc)
    if rna_enc: enc.update(rna_enc)
    rdi = next(iter(rna_enc.values())).shape[0] if rna_enc else 128
    pdi = next(iter(prot_enc.values())).shape[0] if prot_enc else 128
    for sid in rna_ids + prot_ids:
        if sid not in enc:
            enc[sid] = torch.randn(rdi if sid in rna_ids else pdi)
    r2l = {r: i for i, r in enumerate(rna_ids)}; p2l = {p: i for i, p in enumerate(prot_ids)}
    triples = load_interactions(paths["interactions"])
    valid = [(r2l[r], p2l[p], l) for r, p, l in triples if r in r2l and p in p2l]
    return dict(rna_ids=rna_ids, prot_ids=prot_ids,
                rna_embs=torch.stack([enc[r] for r in rna_ids]).to(device),
                prot_embs=torch.stack([enc[p] for p in prot_ids]).to(device),
                valid_triples=valid, rna_in_dim=rdi, prot_in_dim=pdi)

def run_cv(name, data, device, model_kwargs, sim_k=10, lg=0.1, cr=0.05, n_folds=5, seed=42):
    triples = data["valid_triples"]
    X = np.arange(len(triples)); y = np.array([l for _,_,l in triples])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    met = {k: [] for k in ["auc","aupr","accuracy","precision","recall","f1","mcc"]}
    for fold, (tv_idx, te_idx) in enumerate(skf.split(X, y)):
        tv = [triples[i] for i in tv_idx]; te = [triples[i] for i in te_idx]
        sub, val = safe_train_test_split(tv, test_size=0.25, random_state=seed+fold, stratify=True)
        (inter, rme, rmw, pme, pmw, rse, rsw, pse, psw, rp, rm, pm) = build_graph_for_fold(
            sub, data["rna_embs"], data["prot_embs"], data["rna_ids"], data["prot_ids"], device, sim_k)
        Nr = len(data["rna_ids"]); Nt = Nr + len(data["prot_ids"])
        arm = torch.zeros(Nt, dtype=torch.bool, device=device); arm[:Nr] = rm
        apm = torch.zeros(Nt, dtype=torch.bool, device=device); apm[Nr:] = pm
        trc = torch.where(rm)[0]; tpc = torch.where(pm)[0]
        model = EGRLModel(data["rna_in_dim"], data["prot_in_dim"],
                          hidden_dim=CFG["hidden_dim"], gat_heads=CFG["gat_heads"],
                          layers=CFG["num_gat_layers"], dropout=CFG["dropout"], **model_kwargs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
        best_auc, best_state = -1.0, None
        for ep in range(CFG["epochs"]):
            train_epoch(model, opt, data["rna_embs"], data["prot_embs"],
                        rme, pme, rp, sub, inter, rmw, pmw, None,
                        CFG["batch_size"], CFG["clip_grad"], lg, cr,
                        sim_k, CFG["sim_drop_ratio"], CFG["gamma"], f"{name} f{fold+1}")
            sch.step()
            v = evaluate(model, data["rna_embs"], data["prot_embs"], rme, rse, pme, pse, rp, val,
                         CFG["threshold"], rmw, rsw, pmw, psw, None, None, None, arm, apm, trc, tpc)
            if v["auc"] > best_auc:
                best_auc = v["auc"]
                best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        if best_state is not None: model.load_state_dict(best_state)
        r = evaluate(model, data["rna_embs"], data["prot_embs"], rme, rse, pme, pse, rp, te,
                     CFG["threshold"], rmw, rsw, pmw, psw, None, None, None, arm, apm, trc, tpc)
        for k in met: met[k].append(r[k])
        print(f"  {name} fold{fold+1}: AUC={r['auc']:.5f} AUPR={r['aupr']:.5f}")
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1 if len(v)>1 else 0))) for k, v in met.items()}

VARIANTS = {
    "Full": dict(use_meta_path=True,  use_multi_rel=True,  use_generator=True,  use_dynamic_fusion=True),
    "-A":   dict(use_meta_path=False, use_multi_rel=True,  use_generator=True,  use_dynamic_fusion=True),
    "-B":   dict(use_meta_path=True,  use_multi_rel=False, use_generator=True,  use_dynamic_fusion=True),
    "-C":   dict(use_meta_path=True,  use_multi_rel=True,  use_generator=False, use_dynamic_fusion=True),
    "-D":   dict(use_meta_path=True,  use_multi_rel=True,  use_generator=True,  use_dynamic_fusion=False),
}

def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    lines = ["Ablation (paper RQ1/Fig.3), 5-fold, threshold=0.5"]
    for ds in DATASETS:
        data = load_dataset(DATASETS[ds], device)
        lines.append(f"\n===== {ds} | triples={len(data['valid_triples'])} =====")
        for var, kw in VARIANTS.items():
            lg = CFG["lambda_gen"] if kw["use_generator"] else 0.0
            cr = CFG["cold_ratio"] if kw["use_generator"] else 0.0
            s = run_cv(f"{ds}-{var}", data, device, kw, CFG["sim_k"], lg, cr)
            line = (f"{var:>4}: ACC={s['accuracy'][0]:.4f}±{s['accuracy'][1]:.4f} "
                    f"F1={s['f1'][0]:.4f}±{s['f1'][1]:.4f} "
                    f"MCC={s['mcc'][0]:.4f}±{s['mcc'][1]:.4f} "
                    f"AUROC={s['auc'][0]:.4f}±{s['auc'][1]:.4f} "
                    f"AUPR={s['aupr'][0]:.4f}±{s['aupr'][1]:.4f}")
            print(line); lines.append(line)
    with open("ablation_results.txt", "w") as f:
        f.write("\n".join(lines))
    print("\nSaved ablation_results.txt")

if __name__ == "__main__":
    main()