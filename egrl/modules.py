import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm


class ImplicitMetaPathLearner(nn.Module):
    def __init__(self, hidden_dim, num_relations, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.relation_transform = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_relations, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations),
            nn.Softmax(dim=-1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relation_edges, relation_node_masks):
        device = x.device
        relation_feats = []
        for k, (edges, mask) in enumerate(zip(relation_edges, relation_node_masks)):
            if edges.size(1) == 0 or mask.sum() == 0:
                relation_feats.append(torch.zeros(self.hidden_dim, device=device))
                continue
            involved_nodes = mask.nonzero(as_tuple=True)[0]
            transformed = self.relation_transform[k](x[involved_nodes])
            relation_feats.append(transformed.mean(dim=0))
        concat_feats = torch.cat(relation_feats, dim=0).view(1, -1)
        alpha = self.attn_mlp(concat_feats).squeeze(0)
        h_meta = (alpha.unsqueeze(-1) * torch.stack(relation_feats, dim=0)).sum(dim=0)
        x_enhanced = x + self.dropout(h_meta.unsqueeze(0).expand(x.size(0), -1))
        return x_enhanced, alpha


class MultiRelationalGATLayer(nn.Module):
    def __init__(self, hidden_dim, num_relations, heads=4, dropout=0.2):
        super().__init__()
        self.num_relations = num_relations
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout,
                    concat=False, edge_dim=1)
            for _ in range(num_relations)
        ])
        self.relation_logits = nn.Parameter(torch.ones(num_relations))
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x, edge_indices, edge_weights=None):
        out_k_list = []
        for k, (edge_idx, gat) in enumerate(zip(edge_indices, self.gat_layers)):
            if edge_idx.size(1) == 0:
                out_k_list.append(torch.zeros_like(x))
                continue
            edge_attr = (edge_weights[k].unsqueeze(-1)
                         if (edge_weights is not None and edge_weights[k] is not None)
                         else None)
            out_k_list.append(gat(x, edge_idx, edge_attr=edge_attr))
        stacked = torch.stack(out_k_list, dim=0)
        weighted = (self.relation_logits.view(-1, 1, 1) * stacked).sum(dim=0) / self.num_relations
        out = self.norm(x + self.dropout(weighted))
        beta = F.softmax(self.relation_logits.detach(), dim=0)
        return out, beta


class GraphGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x_new, X_existing):
        B, N = x_new.size(0), X_existing.size(0)
        if N == 0:
            return torch.zeros((B, 0), device=x_new.device)
        x_new_exp = x_new.unsqueeze(1).expand(-1, N, -1)
        x_existing_exp = X_existing.unsqueeze(0).expand(B, -1, -1)
        pair_feat = torch.cat([x_new_exp, x_existing_exp], dim=-1)
        h = F.relu(self.fc1(pair_feat))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.out(h).squeeze(-1))


def build_soft_edges(cold_nodes, target_nodes_global, probs, device):
    M, N = probs.shape
    if M == 0 or N == 0:
        return (torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty(0, device=device))
    cold_idx = torch.arange(M, device=device).view(-1, 1).expand(-1, N).flatten()
    target_idx = torch.arange(N, device=device).view(1, -1).expand(M, -1).flatten()
    fwd = torch.stack([cold_nodes[cold_idx], target_nodes_global[target_idx]])
    bwd = torch.stack([target_nodes_global[target_idx], cold_nodes[cold_idx]])
    edge_index = torch.cat([fwd, bwd], dim=1)
    w = probs.flatten()
    return edge_index, torch.cat([w, w.clone()])


class EGRLModel(nn.Module):
    def __init__(self, rna_in_dim, prot_in_dim, hidden_dim=128,
                 num_meta_relations=4, num_gat_relations=5,
                 gat_heads=4, num_gat_layers=2, dropout=0.2):
        super().__init__()
        self.rna_proj = nn.Linear(rna_in_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_in_dim, hidden_dim)
        self.meta_learner = ImplicitMetaPathLearner(hidden_dim, num_meta_relations, dropout)
        self.num_gat_relations = num_gat_relations + 2
        self.gat_layers = nn.ModuleList([
            MultiRelationalGATLayer(hidden_dim, self.num_gat_relations,
                                    heads=gat_heads, dropout=dropout)
            for _ in range(num_gat_layers)
        ])
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.generator = GraphGenerator(hidden_dim, hidden_dim)

    def compute_x_meta(self, rna_feats, prot_feats,
                       rna_meta_edge, rna_sim_edge,
                       prot_meta_edge, prot_sim_edge,
                       active_rna_mask, active_prot_mask):
        rna_h = self.rna_proj(rna_feats)
        prot_h = self.prot_proj(prot_feats)
        x = torch.cat([rna_h, prot_h], dim=0)
        rel_edges = [rna_meta_edge, prot_meta_edge, rna_sim_edge, prot_sim_edge]
        rel_masks = [active_rna_mask, active_prot_mask, active_rna_mask, active_prot_mask]
        x_meta, _ = self.meta_learner(x, rel_edges, rel_masks)
        return x_meta

    def forward(self, rna_feats, prot_feats,
                rna_meta_edge, rna_sim_edge,
                prot_meta_edge, prot_sim_edge,
                rp_edge,
                rna_meta_w=None, rna_sim_w=None,
                prot_meta_w=None, prot_sim_w=None, rp_w=None,
                cold_rna_ids=None, cold_prot_ids=None,
                active_rna_mask=None, active_prot_mask=None,
                train_rna_candidates=None, train_prot_candidates=None):
        rna_h = self.rna_proj(rna_feats)
        prot_h = self.prot_proj(prot_feats)
        x = torch.cat([rna_h, prot_h], dim=0)
        N_rna, N_prot = rna_h.size(0), prot_h.size(0)
        N_total = N_rna + N_prot

        if active_rna_mask is None:
            active_rna_mask = torch.zeros(N_total, dtype=torch.bool, device=x.device)
            active_rna_mask[:N_rna] = True
        if active_prot_mask is None:
            active_prot_mask = torch.zeros(N_total, dtype=torch.bool, device=x.device)
            active_prot_mask[N_rna:] = True

        rel_edges = [rna_meta_edge, prot_meta_edge, rna_sim_edge, prot_sim_edge]
        rel_masks = [active_rna_mask, active_prot_mask, active_rna_mask, active_prot_mask]
        x, meta_alpha = self.meta_learner(x, rel_edges, rel_masks)
        x_meta = x

        gat_edges = [rna_meta_edge, prot_meta_edge, rna_sim_edge, prot_sim_edge, rp_edge]
        edge_weights = [rna_meta_w, prot_meta_w, rna_sim_w, prot_sim_w, rp_w]

        rna_meta_part = x_meta[:N_rna]
        prot_meta_part = x_meta[N_rna:]

        if cold_rna_ids is not None and len(cold_rna_ids) > 0:
            if train_prot_candidates is None:
                train_prot_candidates = torch.arange(N_prot, device=x.device)
            cold_rna_feats = rna_meta_part[cold_rna_ids]
            prot_cand_h = prot_meta_part[train_prot_candidates]
            probs = self.generator(cold_rna_feats, prot_cand_h)
            soft_idx, soft_w = build_soft_edges(
                cold_rna_ids, train_prot_candidates + N_rna, probs, x.device)
            gat_edges.append(soft_idx); edge_weights.append(soft_w)
        else:
            gat_edges.append(torch.empty((2, 0), dtype=torch.long, device=x.device))
            edge_weights.append(None)

        if cold_prot_ids is not None and len(cold_prot_ids) > 0:
            if train_rna_candidates is None:
                train_rna_candidates = torch.arange(N_rna, device=x.device)
            cold_prot_feats = prot_meta_part[cold_prot_ids]
            rna_cand_h = rna_meta_part[train_rna_candidates]
            probs = self.generator(cold_prot_feats, rna_cand_h)
            soft_idx, soft_w = build_soft_edges(
                cold_prot_ids + N_rna, train_rna_candidates, probs, x.device)
            gat_edges.append(soft_idx); edge_weights.append(soft_w)
        else:
            gat_edges.append(torch.empty((2, 0), dtype=torch.long, device=x.device))
            edge_weights.append(None)

        relation_weights = []
        for gat_layer in self.gat_layers:
            x, rel_w = gat_layer(x, gat_edges, edge_weights)
            relation_weights.append(rel_w)

        return x[:N_rna], x[N_rna:], relation_weights, meta_alpha, x_meta

    def predict(self, rna_feats, prot_feats,
                rna_meta_edge, rna_sim_edge,
                prot_meta_edge, prot_sim_edge,
                rp_edge, pairs,
                rna_meta_w=None, rna_sim_w=None,
                prot_meta_w=None, prot_sim_w=None, rp_w=None,
                cold_rna_ids=None, cold_prot_ids=None):
        rna_out, prot_out, _, _, _ = self.forward(
            rna_feats, prot_feats,
            rna_meta_edge, rna_sim_edge, prot_meta_edge, prot_sim_edge, rp_edge,
            rna_meta_w=rna_meta_w, rna_sim_w=rna_sim_w,
            prot_meta_w=prot_meta_w, prot_sim_w=prot_sim_w, rp_w=rp_w,
            cold_rna_ids=cold_rna_ids, cold_prot_ids=cold_prot_ids)
        scores = []
        for r, p in pairs:
            concat = torch.cat([
                rna_out[r], prot_out[p],
                rna_out[r] * prot_out[p],
                torch.abs(rna_out[r] - prot_out[p]),
            ], dim=-1)
            scores.append(torch.sigmoid(self.predictor(concat)).item())
        return scores
