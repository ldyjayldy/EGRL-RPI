"""Cold-start molecule hold-out (NPInter2, 20% molecules)."""
import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm

from egrl.config import DEFAULT_CONFIG
from egrl.data import (load_sequences_from_fasta, load_encodings_from_file,
                       load_interactions)
from egrl.coldstart import run_one_coldstart
from egrl.plotting import plot_and_export_mean_curves


def main():
    protein_fasta = "../data/sequence/NPInter_protein_seq.fa"
    rna_fasta = "../data/sequence/NPInter_rna_seq.fa"
    protein_enc = "../data/encode_LM/NPInter_protein_encoded.txt"
    rna_enc = "../data/encode_LM/NPInter_RNA_encoded.txt"
    interactions = "../data/NPInter_pairs.txt"

    DATASET = "NPInter"
    OUT_DIR = f"Result/{DATASET}_EGRL_molecule_holdout"
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = dict(DEFAULT_CONFIG)
    TRAIN_SEEDS = [42, 43, 44]
    HOLDOUT_RATIO = 0.2

    GLOBAL_SEED = 42
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"TRAIN_SEEDS={TRAIN_SEEDS}, EPOCHS={cfg['epochs']}")

    rna_seqs = load_sequences_from_fasta(rna_fasta)
    prot_seqs = load_sequences_from_fasta(protein_fasta)
    rna_ids, prot_ids = list(rna_seqs.keys()), list(prot_seqs.keys())
    print(f"RNAs: {len(rna_ids)}, proteins: {len(prot_ids)}")

    rna_enc_dict = load_encodings_from_file(rna_enc, 128)
    prot_enc_dict = load_encodings_from_file(protein_enc, 128)
    encodings = {}
    if prot_enc_dict:
        encodings.update(prot_enc_dict)
    if rna_enc_dict:
        encodings.update(rna_enc_dict)
    cfg["rna_in_dim"] = (next(iter(rna_enc_dict.values())).shape[0]
                         if rna_enc_dict else 128)
    cfg["prot_in_dim"] = (next(iter(prot_enc_dict.values())).shape[0]
                          if prot_enc_dict else 128)
    for sid in rna_ids + prot_ids:
        if sid not in encodings:
            encodings[sid] = torch.randn(
                cfg["rna_in_dim"] if sid in rna_ids else cfg["prot_in_dim"])

    rna_id2local = {rid: i for i, rid in enumerate(rna_ids)}
    prot_id2local = {pid: i for i, pid in enumerate(prot_ids)}
    triples = load_interactions(interactions)
    valid_triples = [(rna_id2local[r], prot_id2local[p], l)
                     for r, p, l in triples
                     if r in rna_id2local and p in prot_id2local]
    print("Valid interactions:", len(valid_triples))

    rna_embs_all = torch.stack([encodings[rid] for rid in rna_ids]).to(device)
    prot_embs_all = torch.stack([encodings[pid] for pid in prot_ids]).to(device)
    full_inter = torch.zeros(len(rna_ids), len(prot_ids),
                             dtype=torch.float, device=device)
    for r, p, lab in valid_triples:
        if lab == 1:
            full_inter[r, p] = 1.0

    interacted_rna = torch.where(full_inter.sum(dim=1) > 0)[0].cpu().tolist()
    interacted_prot = torch.where(full_inter.sum(dim=0) > 0)[0].cpu().tolist()
    n_hold_rna = max(1, int(len(interacted_rna) * HOLDOUT_RATIO))
    n_hold_prot = max(1, int(len(interacted_prot) * HOLDOUT_RATIO))
    print(f"Interacted RNAs={len(interacted_rna)}, proteins={len(interacted_prot)}")
    print(f"Held-out RNAs/seed={n_hold_rna}, proteins/seed={n_hold_prot}")

    metric_keys = ["auc", "aupr", "accuracy", "precision", "recall", "f1", "mcc"]
    results = {k: [] for k in metric_keys}
    valid_runs = []
    run_meta = {}  # seed -> per-run metadata

    for seed_val in tqdm(TRAIN_SEEDS, desc="cold-start seeds"):
        sampling_seed = seed_val
        training_seed = seed_val

        rng = np.random.RandomState(sampling_seed)
        hold_rna = rng.choice(interacted_rna, size=n_hold_rna, replace=False).tolist()
        hold_prot = rng.choice(interacted_prot, size=n_hold_prot, replace=False).tolist()

        run_label = seed_val
        tqdm.write(f"  Seed {run_label}: holdout {len(hold_rna)} RNAs, "
                   f"{len(hold_prot)} proteins "
                   f"(sampling_seed={sampling_seed}, training_seed={training_seed})")

        out = run_one_coldstart(hold_rna, hold_prot, valid_triples, full_inter,
                                rna_embs_all, prot_embs_all, rna_ids, prot_ids,
                                cfg, device, seed=training_seed)
        if out is None:
            tqdm.write("    [skip] no valid test triples")
            continue
        res, scores, labels, test_triples = out

        # ============================================================
        # ✅ 写入查询 ID / 精确采样节点数 / 随机种子
        # ============================================================
        detail_file = os.path.join(
            OUT_DIR, f"{DATASET}_seed{run_label}_coldstart_details.txt")
        with open(detail_file, "w") as f:
            f.write(f"# Cold-start run seed={run_label} details\n")
            f.write(f"dataset={DATASET}\n")
            f.write(f"run_label={run_label}\n")
            f.write(f"global_seed={GLOBAL_SEED}\n")
            f.write(f"sampling_seed={sampling_seed}\n")
            f.write(f"training_seed={training_seed}\n")
            f.write(f"seed_mode=shared (sampling_seed==training_seed)\n")
            f.write(f"holdout_ratio={HOLDOUT_RATIO}\n")
            f.write(f"total_interacted_rna={len(interacted_rna)}\n")
            f.write(f"total_interacted_prot={len(interacted_prot)}\n")
            f.write(f"n_holdout_rna={len(hold_rna)}\n")
            f.write(f"n_holdout_prot={len(hold_prot)}\n")
            f.write(f"n_test_queries={len(test_triples)}\n")

            f.write("\n[holdout_rna_local_ids]\n")
            f.write(",".join(map(str, hold_rna)) + "\n")
            f.write("\n[holdout_rna_global_ids]\n")
            f.write(",".join(rna_ids[i] for i in hold_rna) + "\n")
            f.write("\n[holdout_prot_local_ids]\n")
            f.write(",".join(map(str, hold_prot)) + "\n")
            f.write("\n[holdout_prot_global_ids]\n")
            f.write(",".join(prot_ids[i] for i in hold_prot) + "\n")

            f.write("\n[test_queries]\n")
            f.write("RNA\tProtein\tY_true\n")
            for (r_local, p_local, label) in test_triples:
                f.write(f"{rna_ids[r_local]}\t{prot_ids[p_local]}\t{int(label)}\n")

        run_meta[run_label] = {
            "sampling_seed": sampling_seed,
            "training_seed": training_seed,
            "n_holdout_rna": len(hold_rna),
            "n_holdout_prot": len(hold_prot),
            "n_test_queries": len(test_triples),
        }
        tqdm.write(f"    details saved: {os.path.basename(detail_file)}")

        valid_runs.append(run_label)
        for k in metric_keys:
            results[k].append(res[k])
        tqdm.write(f"    AUROC={res['auc']:.4f}, AUPR={res['aupr']:.4f}")

        # --- AUROC curve ---
        fpr, tpr, th = roc_curve(labels, scores)
        with open(os.path.join(
                OUT_DIR, f"{DATASET}_seed{run_label}_AUROC_curve.txt"), "w") as f:
            f.write(f"# {DATASET} seed={run_label} AUROC={res['auc']:.6f}\n")
            f.write("FPR\tTPR\tThreshold\n")
            for a, b, c in zip(fpr, tpr, th):
                f.write(f"{a:.8f}\t{b:.8f}\t{c:.8f}\n")

        # --- AUPR curve ---
        prec, rec, th = precision_recall_curve(labels, scores)
        with open(os.path.join(
                OUT_DIR, f"{DATASET}_seed{run_label}_AUPR_curve.txt"), "w") as f:
            f.write(f"# {DATASET} seed={run_label} AUPR={res['aupr']:.6f}\n")
            f.write("Recall\tPrecision\tThreshold\n")
            for i in range(len(prec)):
                t = th[i] if i < len(th) else float("nan")
                f.write(f"{rec[i]:.8f}\t{prec[i]:.8f}\t"
                        f"{t:.8f}\n" if not np.isnan(t)
                        else f"{rec[i]:.8f}\t{prec[i]:.8f}\tnan\n")

        # --- per-pair predictions ---
        with open(os.path.join(
                OUT_DIR, f"{DATASET}_seed{run_label}_test_predictions.txt"), "w") as f:
            f.write("RNA\tProtein\tY_true\tY_score\n")
            for (r, p, lab), sc in zip(test_triples, scores):
                f.write(f"{rna_ids[r]}\t{prot_ids[p]}\t{int(lab)}\t{sc:.8f}\n")

    # ================= Summary =================
    out_txt = os.path.join(
        OUT_DIR, f"{DATASET}_EGRL_molecule_holdout_AUROC_AUPR.txt")
    with open(out_txt, "w") as f:
        f.write(f"dataset={DATASET}\n")
        f.write("protocol=EGRL_style_molecule_holdout_leakage_controlled\n")
        f.write(f"global_seed={GLOBAL_SEED}\n")
        f.write(f"holdout_ratio={HOLDOUT_RATIO}\n")
        f.write(f"N_SEEDS={len(TRAIN_SEEDS)}\n")
        f.write("seed_mode=shared (sampling_seed==training_seed)\n")
        f.write("sampling_seeds=" + ",".join(map(str, valid_runs)) + "\n")
        f.write("training_seeds=" + ",".join(map(str, valid_runs)) + "\n")
        f.write(f"heldout_RNA_per_seed={n_hold_rna}\n")
        f.write(f"heldout_protein_per_seed={n_hold_prot}\n")
        f.write("graph_isolation=only_train_positive_RPI_edges\n")
        f.write("soft_edges_restricted_to=train_candidate_pools\n")
        f.write("heldout_in_graph=no\n")
        f.write("test_edges_in_graph=no\n")
        f.write("valid_runs=" + ",".join(map(str, valid_runs)) + "\n")

        f.write("\n# file references per seed\n")
        for run_label in valid_runs:
            f.write(f"AUROC_curve_seed{run_label}="
                    f"{DATASET}_seed{run_label}_AUROC_curve.txt\n")
            f.write(f"AUPR_curve_seed{run_label}="
                    f"{DATASET}_seed{run_label}_AUPR_curve.txt\n")
            f.write(f"prediction_dump_seed{run_label}="
                    f"{DATASET}_seed{run_label}_test_predictions.txt\n")
            f.write(f"coldstart_details_seed{run_label}="
                    f"{DATASET}_seed{run_label}_coldstart_details.txt\n")

        f.write("\n")
        for i, run_label in enumerate(valid_runs):
            f.write(f"seed={run_label}\tAUROC={results['auc'][i]:.6f}\t"
                    f"AUPR={results['aupr'][i]:.6f}\n")
        for k in metric_keys:
            v = np.asarray(results[k])
            if len(v) > 0:
                dd = 1 if len(v) > 1 else 0
                f.write(f"{k}_mean_std: {v.mean():.6f} ± {v.std(ddof=dd):.6f}\n")
        aurocs = np.asarray(results["auc"], dtype=np.float64)
        auprs = np.asarray(results["aupr"], dtype=np.float64)
        if len(aurocs) > 1:
            f.write(
                "mean±std (ddof=1): "
                f"AUROC={aurocs.mean():.6f}±{aurocs.std(ddof=1):.6f}\t"
                f"AUPR={auprs.mean():.6f}±{auprs.std(ddof=1):.6f}\n"
            )

        f.write("\n# === per-run seed & sampling summary ===\n")
        f.write("# seed\tsampling_seed\ttraining_seed\t"
                "n_holdout_rna\tn_holdout_prot\tn_test_queries\n")
        for run_label in valid_runs:
            m = run_meta[run_label]
            f.write(f"{run_label}\t{m['sampling_seed']}\t{m['training_seed']}\t"
                    f"{m['n_holdout_rna']}\t{m['n_holdout_prot']}\t"
                    f"{m['n_test_queries']}\n")

    print("Results saved to", out_txt)

    if valid_runs:
        plot_and_export_mean_curves(OUT_DIR, DATASET, valid_runs, out_txt,
                                    n_points=201, protocol_label="molecule hold-out")


if __name__ == "__main__":
    main()