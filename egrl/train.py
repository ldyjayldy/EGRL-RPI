import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .data import shift_prot_edges
from .graph import build_weighted_meta_edges, build_similarity_edges_to_train_pool


def train_epoch(model, optimizer,
                rna_embs, prot_embs,
                rna_meta_edge, prot_meta_edge,
                rp_edge, train_triples, train_interaction_matrix,
                rna_meta_w=None, prot_meta_w=None, rp_w=None,
                batch_size=256, clip_grad=1.0,
                lambda_gen=0.1, cold_ratio=0.05,
                sim_k=10, sim_drop_ratio=0.1, gamma=1.0, desc="Train"):
    model.train()
    random.shuffle(train_triples)
    total_loss = 0.0
    device = rna_embs.device
    N_rna, N_prot = rna_embs.size(0), prot_embs.size(0)
    N_total = N_rna + N_prot

    labels_all = torch.tensor([t[2] for t in train_triples], dtype=torch.float, device=device)
    pos = labels_all.sum()
    neg = len(labels_all) - pos
    pos_weight = (gamma * neg / pos).clamp(min=1.0) if pos > 0 else torch.tensor(1.0, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    rna_cand_all = (train_interaction_matrix.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    prot_cand_all = (train_interaction_matrix.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
    if rna_cand_all.numel() > 0 and cold_ratio > 0:
        n = max(1, int(rna_cand_all.numel() * cold_ratio))
        cold_rna_ids = rna_cand_all[torch.randperm(rna_cand_all.numel(), device=device)[:n]]
    else:
        cold_rna_ids = torch.empty(0, dtype=torch.long, device=device)
    if prot_cand_all.numel() > 0 and cold_ratio > 0:
        n = max(1, int(prot_cand_all.numel() * cold_ratio))
        cold_prot_ids = prot_cand_all[torch.randperm(prot_cand_all.numel(), device=device)[:n]]
    else:
        cold_prot_ids = torch.empty(0, dtype=torch.long, device=device)

    masked_inter = train_interaction_matrix.clone()
    if cold_rna_ids.numel() > 0:
        masked_inter[cold_rna_ids, :] = 0.0
    if cold_prot_ids.numel() > 0:
        masked_inter[:, cold_prot_ids] = 0.0

    (masked_rna_meta_idx, masked_rna_meta_w,
     masked_prot_meta_idx, masked_prot_meta_w) = build_weighted_meta_edges(masked_inter, device)
    masked_prot_meta_idx_shift = shift_prot_edges(masked_prot_meta_idx, N_rna)

    cold_rna_set = set(cold_rna_ids.cpu().tolist())
    cold_prot_set = set(cold_prot_ids.cpu().tolist())
    rp_list = []
    for r, p, l in train_triples:
        if l == 1 and r not in cold_rna_set and p not in cold_prot_set:
            rp_list.append([r, N_rna + p])
            rp_list.append([N_rna + p, r])
    masked_rp_edge = (torch.tensor(rp_list, dtype=torch.long, device=device).t().contiguous()
                      if rp_list else torch.empty((2, 0), dtype=torch.long, device=device))

    retained_rna_mask = (masked_inter.sum(dim=1) > 0)
    retained_prot_mask = (masked_inter.sum(dim=0) > 0)

    masked_rna_sim_edge, masked_rna_sim_w = build_similarity_edges_to_train_pool(
        rna_embs, retained_rna_mask, k=sim_k, device=device, drop_ratio=sim_drop_ratio)
    masked_prot_sim_edge, masked_prot_sim_w = build_similarity_edges_to_train_pool(
        prot_embs, retained_prot_mask, k=sim_k, device=device, drop_ratio=sim_drop_ratio)
    masked_prot_sim_edge_shift = shift_prot_edges(masked_prot_sim_edge, N_rna)

    train_rna_candidates = retained_rna_mask.nonzero(as_tuple=True)[0]
    train_prot_candidates = retained_prot_mask.nonzero(as_tuple=True)[0]

    active_rna = torch.zeros(N_total, dtype=torch.bool, device=device)
    active_prot = torch.zeros(N_total, dtype=torch.bool, device=device)
    active_rna[:N_rna] = retained_rna_mask
    active_prot[N_rna:] = retained_prot_mask

    pbar = tqdm(range(0, len(train_triples), batch_size),
                desc=desc, leave=False, position=2, dynamic_ncols=True)
    for i in pbar:
        batch = train_triples[i:i + batch_size]
        r_idx = [p[0] for p in batch]
        p_idx = [p[1] for p in batch]
        labels = torch.tensor([p[2] for p in batch], dtype=torch.float, device=device)

        rna_out, prot_out, _, _, x_meta = model(
            rna_embs, prot_embs,
            masked_rna_meta_idx, masked_rna_sim_edge,
            masked_prot_meta_idx_shift, masked_prot_sim_edge_shift,
            masked_rp_edge,
            rna_meta_w=masked_rna_meta_w, rna_sim_w=masked_rna_sim_w,
            prot_meta_w=masked_prot_meta_w, prot_sim_w=masked_prot_sim_w, rp_w=None,
            cold_rna_ids=cold_rna_ids, cold_prot_ids=cold_prot_ids,
            active_rna_mask=active_rna, active_prot_mask=active_prot,
            train_rna_candidates=train_rna_candidates,
            train_prot_candidates=train_prot_candidates)

        rna_batch = rna_out[r_idx]
        prot_batch = prot_out[p_idx]
        concat = torch.cat([rna_batch, prot_batch,
                            rna_batch * prot_batch,
                            torch.abs(rna_batch - prot_batch)], dim=-1)
        logits = model.predictor(concat).squeeze(-1)
        main_loss = criterion(logits, labels)

        gen_loss = torch.tensor(0.0, device=device)
        if lambda_gen > 0 and (cold_rna_ids.numel() > 0 or cold_prot_ids.numel() > 0):
            rna_meta_part = x_meta[:N_rna]
            prot_meta_part = x_meta[N_rna:]
            n_terms = 0
            if cold_rna_ids.numel() > 0 and train_prot_candidates.numel() > 0:
                probs = model.generator(rna_meta_part[cold_rna_ids],
                                        prot_meta_part[train_prot_candidates])
                gt = train_interaction_matrix[cold_rna_ids][:, train_prot_candidates]
                gen_loss = gen_loss + F.binary_cross_entropy(probs, gt, reduction="sum")
                n_terms += cold_rna_ids.numel() * train_prot_candidates.numel()
            if cold_prot_ids.numel() > 0 and train_rna_candidates.numel() > 0:
                probs = model.generator(prot_meta_part[cold_prot_ids],
                                        rna_meta_part[train_rna_candidates])
                gt = train_interaction_matrix[:, cold_prot_ids][train_rna_candidates, :].T
                gen_loss = gen_loss + F.binary_cross_entropy(probs, gt, reduction="sum")
                n_terms += cold_prot_ids.numel() * train_rna_candidates.numel()
            if n_terms > 0:
                gen_loss = gen_loss / n_terms

        loss = main_loss + lambda_gen * gen_loss
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item() * len(batch)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / max(1, len(train_triples))
