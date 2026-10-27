import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_LIST = [
    ("acc", "ACC"), ("precision", "PREC"), ("recall", "REC"),
    ("mcc", "MCC"), ("auc", "AUROC"), ("aupr", "AUPR"),
]


def save_results_txt(cv_metrics, out_path, n_folds):
    with open(out_path, "w") as f:
        f.write("===== Standard 5-Fold CV Results (mean ± std) =====\n")
        f.write(f"Number of folds: {n_folds}\n\n")
        for key, name in METRIC_LIST:
            vals = np.array(cv_metrics[key])
            f.write(f"{name}: {vals.mean():.5f} ± {vals.std():.5f}\n")


def save_plot_data_txt(cv_metrics, out_path, n_folds):
    with open(out_path, "w") as f:
        f.write("===== Per-Fold Metric Values =====\n")
        f.write("Metric\t" + "\t".join(f"Fold{i+1}" for i in range(n_folds)) + "\n")
        for key, name in METRIC_LIST:
            vals = cv_metrics[key]
            f.write(name + "\t" + "\t".join(f"{v:.5f}" for v in vals) + "\n")
        f.write("\n===== Boxplot Statistics =====\n")
        f.write("Metric\tMin\tQ1\tMedian\tQ3\tMax\tMean\tStd\n")
        for key, name in METRIC_LIST:
            vals = np.array(cv_metrics[key])
            f.write(f"{name}\t{np.min(vals):.5f}\t{np.percentile(vals, 25):.5f}\t"
                    f"{np.median(vals):.5f}\t{np.percentile(vals, 75):.5f}\t"
                    f"{np.max(vals):.5f}\t{vals.mean():.5f}\t{vals.std():.5f}\n")


def plot_cv_boxplot_violin(cv_metrics, out_pdf="cv_boxplot_violin.pdf"):
    labels = [name for _, name in METRIC_LIST]
    data = [np.array(cv_metrics[key]) for key, _ in METRIC_LIST]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bp = axes[0].boxplot(data, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    axes[0].set_xticks(np.arange(1, len(labels) + 1))
    axes[0].set_xticklabels(labels)
    axes[0].set_title("5-Fold CV: Boxplot")
    axes[0].set_ylabel("Score")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    axes[1].violinplot(data, showmeans=True, showmedians=True)
    axes[1].set_xticks(np.arange(1, len(labels) + 1))
    axes[1].set_xticklabels(labels)
    axes[1].set_title("5-Fold CV: Violin Plot")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _dedup_ascending(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    keep = np.ones(len(x), dtype=bool)
    keep[:-1] = x[:-1] != x[1:]
    return x[keep], y[keep]


def _load_curve_xy(path):
    data = np.loadtxt(path, skiprows=2, delimiter="\t", usecols=(0, 1))
    data = np.atleast_2d(data)
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def _resample_curves(xy_list, grid):
    ys = []
    for x, y in xy_list:
        x, y = _dedup_ascending(x, y)
        if x[0] > grid[0]:
            x = np.concatenate(([grid[0]], x))
            y = np.concatenate(([y[0]], y))
        if x[-1] < grid[-1]:
            x = np.concatenate((x, [grid[-1]]))
            y = np.concatenate((y, [y[-1]]))
        ys.append(np.interp(grid, x, y))
    return np.vstack(ys)


def _parse_metrics(result_txt):
    m = {}
    with open(result_txt) as f:
        for line in f:
            if line.startswith("seed="):
                parts = line.strip().split("\t")
                seed = int(parts[0].split("=")[1])
                auroc = float(parts[1].split("=")[1])
                aupr = float(parts[2].split("=")[1])
                m[seed] = (auroc, aupr)
    return m


def plot_and_export_mean_curves(out_dir, dataset, seeds, result_txt,
                                n_points=201, protocol_label="cold-start"):
    grid = np.linspace(0.0, 1.0, n_points)
    roc_list, pr_list = [], []
    for s in seeds:
        roc_list.append(_load_curve_xy(os.path.join(out_dir, f"{dataset}_seed{s}_AUROC_curve.txt")))
        rec, prec = _load_curve_xy(os.path.join(out_dir, f"{dataset}_seed{s}_AUPR_curve.txt"))
        if rec[0] > rec[-1]:
            rec, prec = rec[::-1], prec[::-1]
        pr_list.append((rec, prec))

    roc_mat = _resample_curves(roc_list, grid)
    pr_mat = _resample_curves(pr_list, grid)
    ddof = 1 if len(seeds) > 1 else 0
    roc_mean, roc_std = roc_mat.mean(0), roc_mat.std(0, ddof=ddof)
    pr_mean, pr_std = pr_mat.mean(0), pr_mat.std(0, ddof=ddof)

    metrics = _parse_metrics(result_txt)
    aurocs = np.array([metrics[s][0] for s in seeds])
    auprs = np.array([metrics[s][1] for s in seeds])
    mean_auroc, std_auroc = aurocs.mean(), aurocs.std(ddof=ddof)
    mean_aupr, std_aupr = auprs.mean(), auprs.std(ddof=ddof)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(grid, roc_mean, color="tab:blue", lw=2,
            label=f"Mean AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}")
    ax.fill_between(grid, roc_mean - roc_std, roc_mean + roc_std,
                    color="tab:blue", alpha=0.2, label="±1 std")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset} {protocol_label} — AUROC (n={len(seeds)})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    ax.plot(grid, pr_mean, color="tab:orange", lw=2,
            label=f"Mean AUPR = {mean_aupr:.4f} ± {std_aupr:.4f}")
    ax.fill_between(grid, pr_mean - pr_std, pr_mean + pr_std,
                    color="tab:orange", alpha=0.2, label="±1 std")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{dataset} {protocol_label} — AUPR (n={len(seeds)})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{dataset}_mean_AUROC_AUPR_curves.png"), dpi=300)
    plt.close(fig)
