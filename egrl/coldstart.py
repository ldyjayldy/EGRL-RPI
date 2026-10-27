import random
import numpy as np
import torch
from tqdm import tqdm
from .data import safe_train_test_split
from .graph import build_graph_for_fold
from .modules import EGRLModel
from .train import train_epoch
from .evaluate import evaluate


def build_coldstart_test_triples(full_inter, holdout_rna, holdout_prot,
                                 train_rna_set, train_prot_set,
                                 neg_ratio=1.0, seed=0):
    rng = np.random.RandomState(seed)
    triples = []
    for r in holdout_rna:
        pos_prots = torch.where(full_inter[r] > 0)[0].cpu().numpy()
        pos_prots = [int(p) for p in pos_prots if p in train_prot_set]
        for p in pos_prots:
            triples.append((r, p, 1))
        n_neg = int(len(pos_prots) * neg_ratio)
        if n_neg > 0:
            cands = list(train_prot_set - set(pos_prots))
            if len(cands) >= n_neg:
                negs = rng.choice(cands, size=n_neg, replace=False)
            else:
                negs = rng.choice(cands, size=n_neg, replace=True)
            for p in negs:
                triples.append((r, int(p), 0))
    for p in holdout_prot:
        pos_rnas = torch.where(full_inter[:, p] > 0)[0].cpu().numpy()
        pos_rnas = [int(r) for r in pos_rnas if r in train_rna_set]
        for r in pos_rnas:
            triples.append((r, p, 1))
        n_neg = int(len(pos_rnas) * neg_ratio)
        if n_neg > 0:
            cands = list(train_rna_set - set(pos_rnas))
            if len(cands) >= n_neg:
                negs = rng.choice(cands, size=n_neg, replace=False)
            else:
                negs = rng.choice(cands, size=n_neg, replace=True)
            for r in negs:
                triples.append((int(r), p, 0))
    return triples


def run_one_coldstart(holdout_rna_global, holdout_prot_global,
                      valid_triples, full_inter, rna_embs_all, prot_embs_all,
                      rna_ids, prot_ids, config, device, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    holdout_rna_set = set(holdout_rna_global)
    holdout_prot_set = set(holdout_prot_global)
    remaining = [t for t in valid_triples
                 if t[0] not in holdout_rna_set and t[1] not in holdout_prot_set]
    subtrain, val = safe_train_test_split(
        remaining, test_size=0.2, random_state=seed, stratify=True)

    (train_inter, rna_meta_idx, rna_meta_w,
     prot_meta_idx_shift, prot_meta_w,
     rna_sim_edge, rna_sim_w,
     prot_sim_edge_shift, prot_sim_w,
     rp_edge, train_rna_mask, train_prot_mask) = build_graph_for_fold(
        subtrain, rna_embs_all, prot_embs_all,
        rna_ids, prot_ids, device, sim_k=config["sim_k"])

    N_rna, N_prot = len(rna_ids), len(prot_ids)
    N_total = N_rna + N_prot
    active_rna = torch.zeros(N_total, dtype=torch.bool, device=device)
    active_prot = torch.zeros(N_total, dtype=torch.bool, device=device)
    active_rna[:N_rna] = train_rna_mask
    active_prot[N_rna:] = train_prot_mask
    train_rna_candidates = torch.where(train_rna_mask)[0]
    train_prot_candidates = torch.where(train_prot_mask)[0]

    model = EGRLModel(
        rna_in_dim=config["rna_in_dim"], prot_in_dim=config["prot_in_dim"],
        hidden_dim=config["hidden_dim"], num_meta_relations=4,
        num_gat_relations=5, gat_heads=config["gat_heads"],
        num_gat_layers=config["num_gat_layers"], dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_auc, best_state = -1.0, None
    for _ in tqdm(range(config["epochs"]), desc=f"cold-start seed={seed}",
                  leave=False, position=1, dynamic_ncols=True):
        train_epoch(model, optimizer,
                    rna_embs_all, prot_embs_all,
                    rna_meta_idx, prot_meta_idx_shift,
                    rp_edge, subtrain, train_inter,
                    rna_meta_w=rna_meta_w, prot_meta_w=prot_meta_w, rp_w=None,
                    batch_size=config["batch_size"], clip_grad=config["clip_grad"],
                    lambda_gen=config["lambda_gen"], cold_ratio=config["cold_ratio"],
                    sim_k=config["sim_k"], sim_drop_ratio=config["sim_drop_ratio"],
                    gamma=config["gamma"], desc=f"cold-start train seed={seed}")
        scheduler.step()
        val_res = evaluate(model, rna_embs_all, prot_embs_all,
                           rna_meta_idx, rna_sim_edge,
                           prot_meta_idx_shift, prot_sim_edge_shift,
                           rp_edge, val, threshold=config["threshold"],
                           rna_meta_w=rna_meta_w, rna_sim_w=rna_sim_w,
                           prot_meta_w=prot_meta_w, prot_sim_w=prot_sim_w, rp_w=None,
                           cold_rna_ids=None, cold_prot_ids=None,
                           active_rna_mask=active_rna, active_prot_mask=active_prot,
                           train_rna_candidates=train_rna_candidates,
                           train_prot_candidates=train_prot_candidates)
        if val_res["auc"] > best_val_auc:
            best_val_auc = val_res["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_rna_set = set(torch.where(train_rna_mask)[0].cpu().tolist())
    train_prot_set = set(torch.where(train_prot_mask)[0].cpu().tolist())
    test_triples = build_coldstart_test_triples(
        full_inter, holdout_rna_global, holdout_prot_global,
        train_rna_set, train_prot_set, neg_ratio=1.0, seed=seed)
    if len(test_triples) == 0:
        return None

    cold_rna_t = (torch.tensor(holdout_rna_global, dtype=torch.long, device=device)
                  if len(holdout_rna_global) > 0 else None)
    cold_prot_t = (torch.tensor(holdout_prot_global, dtype=torch.long, device=device)
                   if len(holdout_prot_global) > 0 else None)
    res, scores, labels = evaluate(
        model, rna_embs_all, prot_embs_all,
        rna_meta_idx, rna_sim_edge,
        prot_meta_idx_shift, prot_sim_edge_shift, rp_edge,
        test_triples, threshold=config["threshold"],
        rna_meta_w=rna_meta_w, rna_sim_w=rna_sim_w,
        prot_meta_w=prot_meta_w, prot_sim_w=prot_sim_w, rp_w=None,
        cold_rna_ids=cold_rna_t, cold_prot_ids=cold_prot_t,
        active_rna_mask=active_rna, active_prot_mask=active_prot,
        train_rna_candidates=train_rna_candidates,
        train_prot_candidates=train_prot_candidates,
        return_scores=True)
    return res, scores, labels, test_triples
