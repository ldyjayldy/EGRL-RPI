"""Generator soft-edge quality evaluation (generator-only ranking)."""
import random
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from egrl.config import DEFAULT_CONFIG
from egrl.data import (load_sequences_from_fasta, load_encodings_from_file,
                       load_interactions, safe_train_test_split)
from egrl.graph import build_graph_for_fold
from egrl.modules import EGRLModel
from egrl.train import train_epoch
from egrl.evaluate import evaluate as eval_pair


N_RUNS = 10
HOLDOUT_FRAC = 0.2
TOPK_LIST = [5, 10]
SEED_BASE = 7000


def rank_metrics(labels, probs, tag):
    aupr = average_precision_score(labels, probs)
    auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    order = np.argsort(-probs)
    random_aupr = float(labels.mean())
    out = {"tag": tag, "n_pos": int(labels.sum()), "n_cand": len(labels),
           "aupr": aupr, "auroc": auroc, "random_aupr": random_aupr,
           "lift": aupr / max(random_aupr, 1e-8)}
    for k in TOPK_LIST:
        kk = min(k, len(order))
        out[f"p@{k}"] = float(labels[order[:kk]].sum() / kk)
    return out


@torch.no_grad()
def generator_ranking(model, x_meta, full_inter, N_rna, holdout_rna, holdout_prot, device):
    rna_part = x_meta[:N_rna]
    prot_part = x_meta[N_rna:]
    N_prot = prot_part.size(0)
    hold_r, hold_p = set(holdout_rna), set(holdout_prot)
    cand_p = torch.tensor([i for i in range(N_prot) if i not in hold_p], device=device)
    cand_r = torch.tensor([i for i in range(N_rna) if i not in hold_r], device=device)

    rna_recs, prot_recs = [], []
    for r in holdout_rna:
        labels = (full_inter[r, cand_p] > 0).cpu().numpy()
        if labels.sum() == 0:
            continue
        probs = model.generator(rna_part[r:r+1], prot_part[cand_p]).squeeze(0).cpu().numpy()
        rna_recs.append(rank_metrics(labels, probs, f"RNA_{r}"))
    for p in holdout_prot:
        labels = (full_inter[cand_r, p] > 0).cpu().numpy()
        if labels.sum() == 0:
            continue
        probs = model.generator(prot_part[p:p+1], rna_part[cand_r]).squeeze(0).cpu().numpy()
        prot_recs.append(rank_metrics(labels, probs, f"PROT_{p}"))
    return rna_recs, prot_recs


def train_one_run(valid_triples, full_inter, rna_embs, prot_embs,
                  rna_ids, prot_ids, device, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    interacted_r = torch.where(full_inter.sum(dim=1) > 0)[0].cpu().tolist()
    interacted_p = torch.where(full_inter.sum(dim=0) > 0)[0].cpu().tolist()
    rng = np.random.RandomState(seed)
    n_hr = max(1, int(len(interacted_r) * HOLDOUT_FRAC))
    n_hp = max(1, int(len(interacted_p) * HOLDOUT_FRAC))
    hold_r = sorted(rng.choice(interacted_r, n_hr, replace=False).tolist())
    hold_p = sorted(rng.choice(interacted_p, n_hp, replace=False).tolist())
    hr_set, hp_set = set(hold_r), set(hold_p)
    remaining = [t for t in valid_triples if t[0] not in hr_set and t[1] not in hp_set]
    subtrain, val = safe_train_test_split(remaining, test_size=0.2, random_state=seed, stratify=True)

    (train_inter, rna_meta_idx, rna_meta_w,
     prot_meta_idx_shift, prot_meta_w,
     rna_sim_edge, rna_sim_w,
     prot_sim_edge_shift, prot_sim_w,
     rp_edge, train_rna_mask, train_prot_mask) = build_graph_for_fold(
        subtrain, rna_embs, prot_embs, rna_ids, prot_ids, device, sim_k=cfg["sim_k"])

    N_rna, N_prot = len(rna_ids), len(prot_ids)
    N_total = N_rna + N_prot
    active_rna = torch.zeros(N_total, dtype=torch.bool, device=device)
    active_prot = torch.zeros(N_total, dtype=torch.bool, device=device)
    active_rna[:N_rna] = train_rna_mask
    active_prot[N_rna:] = train_prot_mask
    tr_r = torch.where(train_rna_mask)[0]; tr_p = torch.where(train_prot_mask)[0]

    model = EGRLModel(
        cfg["rna_in_dim"], cfg["prot_in_dim"], cfg["hidden_dim"],
        num_meta_relations=4, num_gat_relations=5,
        gat_heads=cfg["gat_heads"], num_gat_layers=cfg["num_gat_layers"],
        dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    best_auc, best_state = -1.0, None
    for _ in range(cfg["epochs"]):
        train_epoch(model, opt,
                    rna_embs, prot_embs,
                    rna_meta_idx, prot_meta_idx_shift, rp_edge,
                    subtrain, train_inter,
                    rna_meta_w=rna_meta_w, prot_meta_w=prot_meta_w, rp_w=None,
                    batch_size=cfg["batch_size"], clip_grad=cfg["clip_grad"],
                    lambda_gen=cfg["lambda_gen"], cold_ratio=cfg["cold_ratio"],
                    sim_k=cfg["sim_k"], sim_drop_ratio=cfg["sim_drop_ratio"],
                    gamma=cfg["gamma"], desc=f"seed={seed}")
        sch.step()
        val_res = eval_pair(
            model, rna_embs, prot_embs,
            rna_meta_idx, rna_sim_edge, prot_meta_idx_shift, prot_sim_edge_shift,
            rp_edge, val, threshold=cfg["threshold"],
            rna_meta_w=rna_meta_w, rna_sim_w=rna_sim_w,
            prot_meta_w=prot_meta_w, prot_sim_w=prot_sim_w, rp_w=None,
            active_rna_mask=active_rna, active_prot_mask=active_prot,
            train_rna_candidates=tr_r, train_prot_candidates=tr_p)
        if val_res["auc"] > best_auc:
            best_auc = val_res["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        x_meta = model.compute_x_meta(
            rna_embs, prot_embs,
            rna_meta_idx, rna_sim_edge, prot_meta_idx_shift, prot_sim_edge_shift,
            active_rna, active_prot)
    return model, hold_r, hold_p, x_meta


def main():
    protein_fasta = "../data/sequence/NPInter_protein_seq.fa"
    rna_fasta = "../data/sequence/NPInter_rna_seq.fa"
    protein_enc = "../data/encode_LM/NPInter_protein_encoded.txt"
    rna_enc = "../data/encode_LM/NPInter_RNA_encoded.txt"
    interactions = "../data/NPInter_pairs.txt"

    cfg = dict(DEFAULT_CONFIG)
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
    cfg["rna_in_dim"] = next(iter(rna_enc_dict.values())).shape[0] if rna_enc_dict else 128
    cfg["prot_in_dim"] = next(iter(prot_enc_dict.values())).shape[0] if prot_enc_dict else 128
    for sid in rna_ids + prot_ids:
        if sid not in encodings:
            encodings[sid] = torch.randn(cfg["rna_in_dim"] if sid in rna_ids else cfg["prot_in_dim"])

    rna_id2local = {rid: i for i, rid in enumerate(rna_ids)}
    prot_id2local = {pid: i for i, pid in enumerate(prot_ids)}
    triples = load_interactions(interactions)
    valid_triples = [(rna_id2local[r], prot_id2local[p], l)
                     for r, p, l in triples
                     if r in rna_id2local and p in prot_id2local]
    print("Valid interactions:", len(valid_triples))

    rna_embs_all = torch.stack([encodings[rid] for rid in rna_ids]).to(device)
    prot_embs_all = torch.stack([encodings[pid] for pid in prot_ids]).to(device)
    full_inter = torch.zeros(len(rna_ids), len(prot_ids), dtype=torch.float, device=device)
    for r, p, lab in valid_triples:
        if lab == 1:
            full_inter[r, p] = 1.0

    all_rna, all_prot = [], []
    for run in tqdm(range(N_RUNS), desc="generator quality runs"):
        seed = SEED_BASE + run
        model, hold_r, hold_p, x_meta = train_one_run(
            valid_triples, full_inter, rna_embs_all, prot_embs_all,
            rna_ids, prot_ids, device, cfg, seed)
        rna_recs, prot_recs = generator_ranking(
            model, x_meta, full_inter, len(rna_ids), hold_r, hold_p, device)
        for rec in rna_recs + prot_recs:
            rec["run"] = run
        all_rna.extend(rna_recs)
        all_prot.extend(prot_recs)

    def agg(records, prefix):
        out = {}
        for m in ["aupr", "auroc", "p@5", "p@10", "lift", "random_aupr"]:
            if not records:
                out[f"{prefix}_{m}_mean"] = float("nan")
                out[f"{prefix}_{m}_std"] = float("nan")
                out[f"{prefix}_{m}_ci95"] = float("nan")
                continue
            v = np.array([r[m] for r in records], dtype=float)
            n = len(v)
            mean = v.mean()
            std = v.std(ddof=1) if n > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            out[f"{prefix}_{m}_mean"] = mean
            out[f"{prefix}_{m}_std"] = std
            out[f"{prefix}_{m}_ci95"] = ci95
        return out

    agg_rna = agg(all_rna, "RNA")
    agg_prot = agg(all_prot, "PROT")

    with open("generator_reliability_results.txt", "w") as f:
        f.write("===== Generator reliability =====\n")
        for prefix, a in [("RNA", agg_rna), ("PROT", agg_prot)]:
            for m in ["aupr", "auroc", "p@5", "p@10"]:
                f.write(f"{prefix} {m}: {a[f'{prefix}_{m}_mean']:.4f} ± "
                        f"{a[f'{prefix}_{m}_std']:.4f} "
                        f"(CI95 ±{a[f'{prefix}_{m}_ci95']:.4f})\n")
            f.write(f"{prefix} random AUPR: {a[f'{prefix}_random_aupr_mean']:.4f} ± "
                    f"{a[f'{prefix}_random_aupr_std']:.4f}\n")
            f.write(f"{prefix} lift: {a[f'{prefix}_lift_mean']:.2f}x ± "
                    f"{a[f'{prefix}_lift_std']:.2f}\n\n")

    with open("generator_reliability_per_node.csv", "w") as f:
        f.write("tag,run,n_pos,n_cand,aupr,auroc,random_aupr,lift,p@5,p@10\n")
        for r in all_rna + all_prot:
            f.write(f"{r['tag']},{r['run']},{r['n_pos']},{r['n_cand']},"
                    f"{r['aupr']:.4f},{r['auroc']:.4f},{r['random_aupr']:.4f},"
                    f"{r['lift']:.4f},{r['p@5']:.4f},{r['p@10']:.4f}\n")
    print("Saved generator_reliability_results.txt and .csv")


if __name__ == "__main__":
    main()
