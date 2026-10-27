
import os, random, itertools
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

# ==================== helpers ====================
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

# ==================== graph builders ====================
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

# ==================== modules ====================
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

# ==================== model (generator always on) ====================
class EGRLModel(nn.Module):
    def __init__(self, rna_dim, prot_dim, h=128, nr_meta=4,
                 heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.rna_proj = nn.Linear(rna_dim, h); self.prot_proj = nn.Linear(prot_dim, h)
        self.meta = ImplicitMetaPathLearner(h, nr_meta, dropout)
        self.nr_gat = 7
        self.gats = nn.ModuleList([MultiRelationalGATLayer(h, self.nr_gat, heads, dropout) for _ in range(layers)])
        self.gen = GraphGenerator(h, h)
        self.predictor = nn.Sequential(nn.Linear(h*4, h), nn.LayerNorm(h), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(h, 1))

    def pair_features(self, ro, po, r, p):
        rb, pb = ro[r], po[p]
        return torch.cat([rb, pb, rb*pb, torch.abs(rb-pb)], -1)

    def forward(self, rna_f, prot_f, rme, rse, pme, pse, rp,
                rmw=None, rsw=None, pmw=None, psw=None, rpw=None,
                cold_r=None, cold_p=None, arm=None, apm=None, trc=None, tpc=None):
        rh = self.rna_proj(rna_f); ph = self.prot_proj(prot_f)
        x = torch.cat([rh, ph], 0); Nr, Np = rh.size(0), ph.size(0); Nt = Nr+Np
        if arm is None:
            arm = torch.zeros(Nt, dtype=torch.bool, device=x.device); arm[:Nr] = True
        if apm is None:
            apm = torch.zeros(Nt, dtype=torch.bool, device=x.device); apm[Nr:] = True
        x, _ = self.meta(x, [rme, pme, rse, pse], [arm, apm, arm, apm])
        x_meta = x
        rm_part, pm_part = x_meta[:Nr], x_meta[Nr:]
        sri = srw = spi = spw = None
        if cold_r is not None and len(cold_r) > 0:
            if tpc is None: tpc = torch.arange(Np, device=x.device)
            pr = self.gen(rm_part[cold_r], pm_part[tpc])
            sri, srw = build_soft_edges(cold_r, tpc + Nr, pr, x.device)
        if cold_p is not None and len(cold_p) > 0:
            if trc is None: trc = torch.arange(Nr, device=x.device)
            pr = self.gen(pm_part[cold_p], rm_part[trc])
            spi, spw = build_soft_edges(cold_p + Nr, trc, pr, x.device)
        edges = [rme, pme, rse, pse, rp,
                 sri if sri is not None else torch.empty((2,0), dtype=torch.long, device=x.device),
                 spi if spi is not None else torch.empty((2,0), dtype=torch.long, device=x.device)]
        ws = [rmw, pmw, rsw, psw, rpw, srw, spw]
        for g in self.gats:
            x, _ = g(x, edges, ws)
        return x[:Nr], x[Nr:], x_meta

# ==================== epoch-level pseudo cold training ====================
def train_epoch_epochcold(model, opt, rna_e, prot_e, triples, inter,
                          lambda_gen, cold_ratio, sim_k=10, sdr=0.1, gamma=1.0,
                          bs=256, clip=1.0, desc="Train"):
    """One epoch: sample pseudo cold nodes ONCE, rebuild masked graph ONCE,
    then iterate batches on that graph. This matches paper's
    'During each training epoch, a subset ... is randomly selected'."""
    model.train(); random.shuffle(triples)
    dev = rna_e.device; Nr, Np = rna_e.size(0), prot_e.size(0); Nt = Nr + Np

    labs_all = torch.tensor([t[2] for t in triples], dtype=torch.float, device=dev)
    pos = labs_all.sum(); neg = len(labs_all) - pos
    pw = (gamma*neg/pos).clamp(min=1.0) if pos > 0 else torch.tensor(1.0, device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ---- per-epoch pseudo cold sampling ----
    rca = (inter.sum(1) > 0).nonzero(as_tuple=True)[0]
    pca = (inter.sum(0) > 0).nonzero(as_tuple=True)[0]
    if rca.numel() > 0 and cold_ratio > 0:
        n = max(1, int(rca.numel() * cold_ratio))
        cold_r = rca[torch.randperm(rca.numel(), device=dev)[:n]]
    else:
        cold_r = torch.empty(0, dtype=torch.long, device=dev)
    if pca.numel() > 0 and cold_ratio > 0:
        n = max(1, int(pca.numel() * cold_ratio))
        cold_p = pca[torch.randperm(pca.numel(), device=dev)[:n]]
    else:
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

    # ---- batch loop on the same masked graph ----
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
        if lambda_gen > 0 and (cold_r.numel() > 0 or cold_p.numel() > 0):
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

        loss = main + lambda_gen * gl
        opt.zero_grad(); loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); total += loss.item() * len(batch)
    return total / max(1, len(triples))

# ==================== evaluate ====================
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

# ==================== dataset + sweep runner ====================
CFG = dict(hidden_dim=128, gat_heads=4, num_gat_layers=2, dropout=0.2,
           lr=1e-4, weight_decay=1e-4, batch_size=256, clip_grad=1.0,
           sim_k=10, sim_drop_ratio=0.1, epochs=300, gamma=1.0, threshold=0.5)

DATASETS = {
    "RPI1807": dict(protein_fasta="../data/sequence/RPI1807_protein_seq.fa",
                    rna_fasta="../data/sequence/RPI1807_rna_seq.fa",
                    protein_enc="../data/encode_LM/RPI1807_protein_encoded.txt",
                    rna_enc="../data/encode_LM/RPI1807_RNA_encoded.txt",
                    interactions="../data/RPI1807_pairs.txt"),
    "RPI2241": dict(protein_fasta="../data/sequence/RPI2241_protein_seq.fa",
                    rna_fasta="../data/sequence/RPI2241_rna_seq.fa",
                    protein_enc="../data/encode_LM/RPI2241_protein_encoded.txt",
                    rna_enc="../data/encode_LM/RPI2241_RNA_encoded.txt",
                    interactions="../data/RPI2241_pairs.txt"),
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


def run_cv_coldtest(name, data, device, lambda_gen, cold_ratio,
                    sim_k=10, n_folds=5, seed=42):
    """5-fold CV; test-fold nodes are treated as cold at evaluation time.
    Training uses per-epoch pseudo cold sampling with given cold_ratio and lambda_gen."""
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
        trc_all = torch.where(rm)[0]; tpc_all = torch.where(pm)[0]

        # test-fold cold nodes; candidate pools exclude test nodes
        test_r_set = set(r for r, _, _ in te)
        test_p_set = set(p for _, p, _ in te)
        test_r = sorted(test_r_set); test_p = sorted(test_p_set)
        cold_r_test = torch.tensor(test_r, dtype=torch.long, device=device) if test_r else None
        cold_p_test = torch.tensor(test_p, dtype=torch.long, device=device) if test_p else None
        trc_eval = torch.tensor([i for i in trc_all.cpu().tolist() if i not in test_r_set],
                                dtype=torch.long, device=device)
        tpc_eval = torch.tensor([i for i in tpc_all.cpu().tolist() if i not in test_p_set],
                                dtype=torch.long, device=device)

        model = EGRLModel(data["rna_in_dim"], data["prot_in_dim"],
                          hidden_dim=CFG["hidden_dim"], gat_heads=CFG["gat_heads"],
                          layers=CFG["num_gat_layers"], dropout=CFG["dropout"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])

        best_auc, best_state = -1.0, None
        for ep in range(CFG["epochs"]):
            train_epoch_epochcold(
                model, opt, data["rna_embs"], data["prot_embs"],
                sub, inter, lambda_gen, cold_ratio,
                sim_k=sim_k, sdr=CFG["sim_drop_ratio"], gamma=CFG["gamma"],
                bs=CFG["batch_size"], clip=CFG["clip_grad"],
                desc=f"{name} lg={lambda_gen} cr={cold_ratio} f{fold+1}")
            sch.step()
            # validation: no cold, original graph
            v = evaluate(model, data["rna_embs"], data["prot_embs"],
                         rme, rse, pme, pse, rp, val, CFG["threshold"],
                         rmw, rsw, pmw, psw, None)
            if v["auc"] > best_auc:
                best_auc = v["auc"]
                best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # test: test-fold nodes as cold; original training graph + soft edges
        r = evaluate(model, data["rna_embs"], data["prot_embs"],
                     rme, rse, pme, pse, rp, te, CFG["threshold"],
                     rmw, rsw, pmw, psw, None,
                     cold_r=cold_r_test, cold_p=cold_p_test,
                     arm=arm, apm=apm, trc=trc_eval, tpc=tpc_eval)
        for k in met: met[k].append(r[k])
        print(f"  {name} fold{fold+1}: AUC={r['auc']:.5f} AUPR={r['aupr']:.5f}")

    return {k: (float(np.mean(v)), float(np.std(v, ddof=1 if len(v)>1 else 0))) for k, v in met.items()}


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    lambda_grid = [0.02, 0.05, 0.1, 0.2]
    cold_grid = [0.01, 0.05, 0.1, 0.15]
    lines = ["Generator sweep (paper RQ2 Exp2/Fig.6): AUROC mean±std over 5 folds",
             f"lambda_gen={lambda_grid}", f"cold_ratio={cold_grid}"]

    for ds in ["RPI1807", "RPI2241"]:
        data = load_dataset(DATASETS[ds], device)
        lines.append(f"\n===== {ds} | triples={len(data['valid_triples'])} =====")
        header = "lambda\\cold\t" + "\t".join(f"{c:.2f}" for c in cold_grid)
        lines.append(header); print(header)
        for lg in lambda_grid:
            row_auc, row_aupr = [], []
            for cr in cold_grid:
                s = run_cv_coldtest(f"{ds}-lg{lg}-cr{cr}", data, device, lg, cr,
                                    sim_k=CFG["sim_k"], n_folds=5, seed=42)
                row_auc.append(f"{s['auc'][0]:.4f}±{s['auc'][1]:.4f}")
                row_aupr.append(f"{s['aupr'][0]:.4f}±{s['aupr'][1]:.4f}")
            line = f"{lg:.2f}\t" + "\t".join(row_auc)
            print(line); lines.append(line)
            lines.append(f"  AUPR for lg={lg:.2f}: " + " | ".join(row_aupr))
        lines.append("")

    with open("generator_sweep_results.txt", "w") as f:
        f.write("\n".join(lines))
    print("\nSaved generator_sweep_results.txt")


if __name__ == "__main__":
    main()