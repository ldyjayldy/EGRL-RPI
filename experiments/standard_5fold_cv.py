"""Standard 5-fold CV experiment."""
import os
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from egrl.config import DEFAULT_CONFIG
from egrl.data import (load_sequences_from_fasta, load_encodings_from_file,
                       load_interactions, safe_train_test_split)
from egrl.graph import build_graph_for_fold
from egrl.modules import EGRLModel
from egrl.train import train_epoch
from egrl.evaluate import evaluate
from egrl.plotting import (METRIC_LIST, save_results_txt, save_plot_data_txt,
                           plot_cv_boxplot_violin)


def main():
    protein_fasta = "../data/sequence/NPInter_protein_seq.fa"
    rna_fasta = "../data/sequence/NPInter_rna_seq.fa"
    protein_enc = "../data/encode_LM/NPInter_protein_encoded.txt"
    rna_enc = "../data/encode_LM/NPInter_RNA_encoded.txt"
    interactions = "../data/NPInter_pairs.txt"

    cfg = dict(DEFAULT_CONFIG)
    N_FOLDS = 5
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    rna_seqs = load_sequences_from_fasta(rna_fasta)
    prot_seqs = load_sequences_from_fasta(protein_fasta)
    rna_ids, prot_ids = list(rna_seqs.keys()), list(prot_seqs.keys())

    rna_enc_dict = load_encodings_from_file(rna_enc, 128)
    prot_enc_dict = load_encodings_from_file(protein_enc, 128)
    encodings = {}
    if prot_enc_dict: encodings.update(prot_enc_dict)
    if rna_enc_dict: encodings.update(rna_enc_dict)
    rna_in_dim = next(iter(rna_enc_dict.values())).shape[0] if rna_enc_dict else 128
    prot_in_dim = next(iter(prot_enc_dict.values())).shape[0] if prot_enc_dict else 128
    cfg["rna_in_dim"] = rna_in_dim
    cfg["prot_in_dim"] = prot_in_dim
    for sid in rna_ids + prot_ids:
        if sid not in encodings:
            encodings[sid] = torch.randn(rna_in_dim if sid in rna_ids else prot_in_dim)

    rna_id2local = {rid: i for i, rid in enumerate(rna_ids)}
    prot_id2local = {pid: i for i, pid in enumerate(prot_ids)}
    triples = load_interactions(interactions)
    valid_triples = [(rna_id2local[r], prot_id2local[p], l)
                     for r, p, l in triples
                     if r in rna_id2local and p in prot_id2local]
    print("Valid interactions:", len(valid_triples))

    rna_embs_all = torch.stack([encodings[rid] for rid in rna_ids]).to(device)
    prot_embs_all = torch.stack([encodings[pid] for pid in prot_ids]).to(device)

    X = np.arange(len(valid_triples))
    y = np.array([l for _, _, l in valid_triples])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    cv_metrics = {k: [] for k, _ in METRIC_LIST}

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{N_FOLDS}")
        train_val = [valid_triples[i] for i in train_val_idx]
        test = [valid_triples[i] for i in test_idx]
        subtrain, val = safe_train_test_split(
            train_val, test_size=0.25, random_state=42 + fold, stratify=True)

        (train_inter, rna_meta_idx, rna_meta_w,
         prot_meta_idx_shift, prot_meta_w,
         rna_sim_edge, rna_sim_w,
         prot_sim_edge_shift, prot_sim_w,
         rp_edge, train_rna_mask, train_prot_mask) = build_graph_for_fold(
            subtrain, rna_embs_all, prot_embs_all,
            rna_ids, prot_ids, device, sim_k=cfg["sim_k"])

        N_rna, N_prot = len(rna_ids), len(prot_ids)
        N_total = N_rna + N_prot
        active_rna = torch.zeros(N_total, dtype=torch.bool, device=device)
        active_prot = torch.zeros(N_total, dtype=torch.bool, device=device)
        active_rna[:N_rna] = train_rna_mask
        active_prot[N_rna:] = train_prot_mask
        train_rna_cands = torch.where(train_rna_mask)[0]
        train_prot_cands = torch.where(train_prot_mask)[0]

        model = EGRLModel(
            rna_in_dim, prot_in_dim, cfg["hidden_dim"],
            num_meta_relations=4, num_gat_relations=5,
            gat_heads=cfg["gat_heads"], num_gat_layers=cfg["num_gat_layers"],
            dropout=cfg["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

        best_val_auc, best_state = -1.0, None
        for ep in range(cfg["epochs"]):
            train_epoch(model, optimizer,
                        rna_embs_all, prot_embs_all,
                        rna_meta_idx, prot_meta_idx_shift,
                        rp_edge, subtrain, train_inter,
                        rna_meta_w=rna_meta_w, prot_meta_w=prot_meta_w, rp_w=None,
                        batch_size=cfg["batch_size"], clip_grad=cfg["clip_grad"],
                        lambda_gen=cfg["lambda_gen"], cold_ratio=cfg["cold_ratio"],
                        sim_k=cfg["sim_k"], sim_drop_ratio=cfg["sim_drop_ratio"],
                        gamma=cfg["gamma"], desc=f"fold{fold+1}")
            scheduler.step()
            val_res = evaluate(
                model, rna_embs_all, prot_embs_all,
                rna_meta_idx, rna_sim_edge, prot_meta_idx_shift, prot_sim_edge_shift,
                rp_edge, val, threshold=cfg["threshold"],
                rna_meta_w=rna_meta_w, rna_sim_w=rna_sim_w,
                prot_meta_w=prot_meta_w, prot_sim_w=prot_sim_w, rp_w=None,
                active_rna_mask=active_rna, active_prot_mask=active_prot,
                train_rna_candidates=train_rna_cands,
                train_prot_candidates=train_prot_cands)
            if val_res["auc"] > best_val_auc:
                best_val_auc = val_res["auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        res = evaluate(
            model, rna_embs_all, prot_embs_all,
            rna_meta_idx, rna_sim_edge, prot_meta_idx_shift, prot_sim_edge_shift,
            rp_edge, test, threshold=cfg["threshold"],
            rna_meta_w=rna_meta_w, rna_sim_w=rna_sim_w,
            prot_meta_w=prot_meta_w, prot_sim_w=prot_sim_w, rp_w=None,
            active_rna_mask=active_rna, active_prot_mask=active_prot,
            train_rna_candidates=train_rna_cands,
            train_prot_candidates=train_prot_cands)

        for k, _ in METRIC_LIST:
            src = "accuracy" if k == "acc" else k
            cv_metrics[k].append(res[src])
        print(f"  Fold {fold+1}: " + ", ".join(f"{n}={res['accuracy' if k=='acc' else k]:.4f}"
                                              for k, n in METRIC_LIST))

    save_results_txt(cv_metrics, "results_5fold.txt", N_FOLDS)
    save_plot_data_txt(cv_metrics, "plot_data.txt", N_FOLDS)
    plot_cv_boxplot_violin(cv_metrics, "cv_boxplot.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
