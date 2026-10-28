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
    N_COLD_RUNS = 3
    HOLDOUT_RATIO = 0.2
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
    full_inter = torch.zeros(len(rna_ids), len(prot_ids), dtype=torch.float, device=device)
    for r, p, lab in valid_triples:
        if lab == 1:
            full_inter[r, p] = 1.0

    interacted_rna = torch.where(full_inter.sum(dim=1) > 0)[0].cpu().tolist()
    interacted_prot = torch.where(full_inter.sum(dim=0) > 0)[0].cpu().tolist()
    n_hold_rna = max(1, int(len(interacted_rna) * HOLDOUT_RATIO))
    n_hold_prot = max(1, int(len(interacted_prot) * HOLDOUT_RATIO))

    metric_keys = ["auc", "aupr", "accuracy", "precision", "recall", "f1", "mcc"]
    results = {k: [] for k in metric_keys}
    valid_runs = []

    for run in tqdm(range(N_COLD_RUNS), desc="cold-start runs"):
        rng = np.random.RandomState(1000 + run)
        hold_rna = rng.choice(interacted_rna, size=n_hold_rna, replace=False).tolist()
        hold_prot = rng.choice(interacted_prot, size=n_hold_prot, replace=False).tolist()
        run_label = run + 1
        out = run_one_coldstart(hold_rna, hold_prot, valid_triples, full_inter,
                                rna_embs_all, prot_embs_all, rna_ids, prot_ids,
                                cfg, device, seed=2000 + run)
        if out is None:
            continue
        res, scores, labels, test_triples = out
        valid_runs.append(run_label)
        for k in metric_keys:
            results[k].append(res[k])

        fpr, tpr, th = roc_curve(labels, scores)
        with open(os.path.join(OUT_DIR, f"{DATASET}_seed{run_label}_AUROC_curve.txt"), "w") as f:
            f.write(f"# {DATASET} run={run_label} AUROC={res['auc']:.6f}\n")
            f.write("FPR\tTPR\tThreshold\n")
            for a, b, c in zip(fpr, tpr, th):
                f.write(f"{a:.8f}\t{b:.8f}\t{c:.8f}\n")

        prec, rec, th = precision_recall_curve(labels, scores)
        with open(os.path.join(OUT_DIR, f"{DATASET}_seed{run_label}_AUPR_curve.txt"), "w") as f:
            f.write(f"# {DATASET} run={run_label} AUPR={res['aupr']:.6f}\n")
            f.write("Recall\tPrecision\tThreshold\n")
            for i in range(len(prec)):
                t = th[i] if i < len(th) else float("nan")
                f.write(f"{rec[i]:.8f}\t{prec[i]:.8f}\t"
                        f"{t:.8f}\n" if not np.isnan(t) else f"{rec[i]:.8f}\t{prec[i]:.8f}\tnan\n")

        with open(os.path.join(OUT_DIR, f"{DATASET}_seed{run_label}_test_predictions.txt"), "w") as f:
            f.write("RNA\tProtein\tY_true\tY_score\n")
            for (r, p, lab), sc in zip(test_triples, scores):
                f.write(f"{rna_ids[r]}\t{prot_ids[p]}\t{int(lab)}\t{sc:.8f}\n")

    out_txt = os.path.join(OUT_DIR, f"{DATASET}_EGRL_molecule_holdout_AUROC_AUPR.txt")
    with open(out_txt, "w") as f:
        for i, run_label in enumerate(valid_runs):
            f.write(f"seed={run_label}\tAUROC={results['auc'][i]:.6f}\t"
                    f"AUPR={results['aupr'][i]:.6f}\n")
        for k in metric_keys:
            v = np.asarray(results[k])
            if len(v) > 0:
                dd = 1 if len(v) > 1 else 0
                f.write(f"{k}_mean_std: {v.mean():.6f} ± {v.std(ddof=dd):.6f}\n")
    print("Results saved to", out_txt)

    if valid_runs:
        plot_and_export_mean_curves(OUT_DIR, DATASET, valid_runs, out_txt,
                                    n_points=201, protocol_label="molecule hold-out")


if __name__ == "__main__":
    main()
