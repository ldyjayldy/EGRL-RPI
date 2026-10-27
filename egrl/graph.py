import torch
import torch.nn.functional as F
from .data import shift_prot_edges


def build_weighted_meta_edges(interaction_matrix, device="cpu"):
    A = interaction_matrix.to(device)
    rna_rna = torch.mm(A, A.T)
    deg_rna = A.sum(dim=1)
    union = deg_rna.unsqueeze(1) + deg_rna.unsqueeze(0) - rna_rna
    jaccard = rna_rna / (union + 1e-8)
    jaccard.fill_diagonal_(0)
    mask = jaccard > 0.1
    rna_edge_idx = mask.nonzero(as_tuple=False).t().contiguous()
    rna_edge_w = jaccard[mask]

    prot_prot = torch.mm(A.T, A)
    deg_prot = A.sum(dim=0)
    union_p = deg_prot.unsqueeze(1) + deg_prot.unsqueeze(0) - prot_prot
    jaccard_p = prot_prot / (union_p + 1e-8)
    jaccard_p.fill_diagonal_(0)
    mask_p = jaccard_p > 0
    prot_edge_idx = mask_p.nonzero(as_tuple=False).t().contiguous()
    prot_edge_w = jaccard_p[mask_p]
    return rna_edge_idx, rna_edge_w, prot_edge_idx, prot_edge_w


def build_similarity_edges_to_train_pool(embeddings, train_mask, k=10,
                                         device="cpu", drop_ratio=0.0):
    n = embeddings.size(0)
    train_indices = torch.where(train_mask)[0]
    num_train = train_indices.numel()
    empty = (torch.empty((2, 0), dtype=torch.long, device=device),
             torch.empty(0, device=device))
    if n < 2 or num_train < 2:
        return empty
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    train_emb = emb_norm[train_indices]
    sim_matrix = torch.mm(train_emb, train_emb.t())
    sim_matrix.fill_diagonal_(-2.0)
    k_eff = min(k, num_train - 1)
    if k_eff <= 0:
        return empty
    topk_vals, topk_pos = torch.topk(sim_matrix, k_eff, dim=1)
    all_edges, all_weights = [], []
    for i_local in range(num_train):
        i_global = train_indices[i_local].item()
        for j_local, w in zip(topk_pos[i_local], topk_vals[i_local]):
            j_global = train_indices[j_local].item()
            all_edges.append((i_global, j_global))
            all_weights.append(w.item())
    if not all_edges:
        return empty
    edge_index = torch.tensor(all_edges, dtype=torch.long, device=device).t().contiguous()
    edge_weight = torch.tensor(all_weights, dtype=torch.float, device=device)
    if drop_ratio > 0 and edge_index.size(1) > 0:
        keep = torch.rand(edge_index.size(1), device=device) >= drop_ratio
        edge_index = edge_index[:, keep]
        edge_weight = edge_weight[keep]
    if edge_index.size(1) == 0:
        return empty
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    return edge_index, edge_weight


def build_graph_for_fold(train_triples, rna_embs_all, prot_embs_all,
                         rna_ids, prot_ids, device, sim_k=10):
    N_rna, N_prot = len(rna_ids), len(prot_ids)
    train_inter = torch.zeros(N_rna, N_prot, dtype=torch.float, device=device)
    for r, p, l in train_triples:
        if l == 1:
            train_inter[r, p] = 1.0

    rna_meta_idx, rna_meta_w, prot_meta_idx, prot_meta_w = build_weighted_meta_edges(train_inter, device)
    prot_meta_idx_shift = shift_prot_edges(prot_meta_idx, N_rna)

    rp_list = []
    for r, p, l in train_triples:
        if l == 1:
            rp_list.append([r, N_rna + p])
            rp_list.append([N_rna + p, r])
    rp_edge = (torch.tensor(rp_list, dtype=torch.long, device=device).t().contiguous()
               if rp_list else torch.empty((2, 0), dtype=torch.long, device=device))

    train_rna_mask = (train_inter.sum(dim=1) > 0)
    train_prot_mask = (train_inter.sum(dim=0) > 0)

    rna_sim_edge, rna_sim_w = build_similarity_edges_to_train_pool(
        rna_embs_all, train_rna_mask, k=sim_k, device=device, drop_ratio=0.0)
    prot_sim_edge, prot_sim_w = build_similarity_edges_to_train_pool(
        prot_embs_all, train_prot_mask, k=sim_k, device=device, drop_ratio=0.0)
    prot_sim_edge_shift = shift_prot_edges(prot_sim_edge, N_rna)

    return (train_inter, rna_meta_idx, rna_meta_w,
            prot_meta_idx_shift, prot_meta_w,
            rna_sim_edge, rna_sim_w,
            prot_sim_edge_shift, prot_sim_w,
            rp_edge, train_rna_mask, train_prot_mask)
