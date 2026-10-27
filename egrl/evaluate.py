import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
)


@torch.no_grad()
def evaluate(model, rna_embs, prot_embs,
             rna_meta_edge, rna_sim_edge,
             prot_meta_edge, prot_sim_edge,
             rp_edge, test_triples, threshold=0.5,
             rna_meta_w=None, rna_sim_w=None,
             prot_meta_w=None, prot_sim_w=None, rp_w=None,
             cold_rna_ids=None, cold_prot_ids=None,
             active_rna_mask=None, active_prot_mask=None,
             train_rna_candidates=None, train_prot_candidates=None,
             return_weights=False, return_scores=False):
    model.eval()
    rna_out, prot_out, rel_weights, meta_alpha, _ = model(
        rna_embs, prot_embs,
        rna_meta_edge, rna_sim_edge, prot_meta_edge, prot_sim_edge, rp_edge,
        rna_meta_w=rna_meta_w, rna_sim_w=rna_sim_w,
        prot_meta_w=prot_meta_w, prot_sim_w=prot_sim_w, rp_w=rp_w,
        cold_rna_ids=cold_rna_ids, cold_prot_ids=cold_prot_ids,
        active_rna_mask=active_rna_mask, active_prot_mask=active_prot_mask,
        train_rna_candidates=train_rna_candidates,
        train_prot_candidates=train_prot_candidates)

    scores, labels = [], []
    for r, p, label in test_triples:
        concat = torch.cat([rna_out[r], prot_out[p],
                            rna_out[r] * prot_out[p],
                            torch.abs(rna_out[r] - prot_out[p])], dim=-1)
        scores.append(torch.sigmoid(model.predictor(concat)).item())
        labels.append(label)

    preds = [1 if s >= threshold else 0 for s in scores]
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    aupr = average_precision_score(labels, scores)
    res = {
        "auc": auc, "aupr": aupr,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0.0,
    }
    if return_weights:
        res["rel_weights"] = torch.stack(rel_weights).mean(dim=0).cpu().numpy()
        res["meta_weights"] = meta_alpha.cpu().numpy()
    if return_scores:
        return res, np.asarray(scores), np.asarray(labels)
    return res
