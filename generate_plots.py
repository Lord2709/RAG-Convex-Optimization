"""
generate_plots.py
-----------------
Generates all publication-quality figures for the MSML604 final report and slides.
Reads from results/results.json — run after main.py completes.

Output files (saved to results/figures/):
  fig1_recall_at_k.png       — Recall@k curves for all systems
  fig2_precision_at_k.png    — Precision@k curves for all systems
  fig3_mrr_bar.png           — MRR comparison bar chart
  fig4_weights.png           — Learned weight distributions
  fig5_relaxation_gap.png    — Convex QP vs Non-Convex gap across k
  fig6_generation.png        — End-to-end generation EM and F1
  fig7_summary_heatmap.png   — Full metrics heatmap across all systems
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# System colours — consistent across all figures
COLORS = {
    "Convex QP":    "#2563EB",   # blue
    "Non-Convex":   "#16A34A",   # green
    "Equal Weights":"#9333EA",   # purple
    "Cosine Only":  "#EA580C",   # orange
    "BM25 Only":    "#DC2626",   # red
}
MARKERS = {
    "Convex QP":    "o",
    "Non-Convex":   "s",
    "Equal Weights":"^",
    "Cosine Only":  "D",
    "BM25 Only":    "P",
}
LINE_STYLES = {
    "Convex QP":    "-",
    "Non-Convex":   "--",
    "Equal Weights":":",
    "Cosine Only":  "-.",
    "BM25 Only":    (0, (3, 1, 1, 1)),
}

# ── Load results ──────────────────────────────────────────────────────────────
results_path = Path("results/results.json")
with open(results_path) as f:
    data = json.load(f)

retrieval   = data["retrieval"]
generation  = data["generation"]
weights     = data["weights"]
cfg         = data["config_snapshot"]

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

k_values = [1, 3, 5, 10]
systems  = list(retrieval.keys())


# ── Fig 1: Recall@k curves ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

for sys in systems:
    vals = [retrieval[sys][f"Recall@{k}"] for k in k_values]
    ax.plot(k_values, vals,
            color=COLORS[sys], marker=MARKERS[sys],
            linestyle=LINE_STYLES[sys], linewidth=2, markersize=7,
            label=sys)

ax.set_xlabel("k  (number of retrieved documents)")
ax.set_ylabel("Recall@k")
ax.set_title("Recall@k Across All Systems\n"
             f"TriviaQA (rc.wikipedia), {cfg['num_test_queries']} test queries, "
             f"corpus size = {cfg['corpus_size']}")
ax.set_xticks(k_values)
ax.set_ylim(0.40, 1.02)
ax.legend(loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.savefig(out_dir / "fig1_recall_at_k.png")
plt.close()
print("Saved fig1_recall_at_k.png")


# ── Fig 2: Precision@k curves ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

for sys in systems:
    vals = [retrieval[sys][f"Precision@{k}"] for k in k_values]
    ax.plot(k_values, vals,
            color=COLORS[sys], marker=MARKERS[sys],
            linestyle=LINE_STYLES[sys], linewidth=2, markersize=7,
            label=sys)

ax.set_xlabel("k  (number of retrieved documents)")
ax.set_ylabel("Precision@k")
ax.set_title("Precision@k Across All Systems\n"
             f"TriviaQA (rc.wikipedia), {cfg['num_test_queries']} test queries")
ax.set_xticks(k_values)
ax.set_ylim(0.0, 1.05)
ax.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
plt.savefig(out_dir / "fig2_precision_at_k.png")
plt.close()
print("Saved fig2_precision_at_k.png")


# ── Fig 3: MRR bar chart ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

mrr_vals = [retrieval[sys]["MRR"] for sys in systems]
bars = ax.bar(systems, mrr_vals,
              color=[COLORS[s] for s in systems],
              edgecolor="white", linewidth=0.8, width=0.55)

# Value labels on bars
for bar, val in zip(bars, mrr_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

ax.set_ylabel("Mean Reciprocal Rank (MRR)")
ax.set_title("MRR Comparison Across All Systems\n"
             "Higher is better  |  MRR = mean(1/rank of first relevant doc)")
ax.set_ylim(0, 1.12)
ax.set_xticklabels(systems, rotation=15, ha="right")
plt.tight_layout()
plt.savefig(out_dir / "fig3_mrr_bar.png")
plt.close()
print("Saved fig3_mrr_bar.png")


# ── Fig 4: Learned weights ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4))

feature_names = ["Cosine\nSimilarity", "BM25\nScore", "MMR\nDiversity"]
x = np.arange(len(feature_names))
width = 0.25

w_qp  = weights["convex_qp"]
w_nc  = weights["nonconvex"]
w_eq  = weights["equal"]

b1 = ax.bar(x - width, w_qp, width, label="Convex QP",
            color=COLORS["Convex QP"], edgecolor="white")
b2 = ax.bar(x,         w_nc, width, label="Non-Convex",
            color=COLORS["Non-Convex"], edgecolor="white")
b3 = ax.bar(x + width, w_eq, width, label="Equal Weights",
            color=COLORS["Equal Weights"], edgecolor="white", alpha=0.7)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(feature_names)
ax.set_ylabel("Learned Weight  w*")
ax.set_title("Learned Retrieval Signal Weights\n"
             "Simplex constraint: weights ≥ 0,  Σw = 1")
ax.set_ylim(0, 0.75)
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(out_dir / "fig4_weights.png")
plt.close()
print("Saved fig4_weights.png")


# ── Fig 5: Relaxation gap (Convex QP vs Non-Convex) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Recall gap
ax = axes[0]
qp_recall  = [retrieval["Convex QP"][f"Recall@{k}"]   for k in k_values]
nc_recall  = [retrieval["Non-Convex"][f"Recall@{k}"]  for k in k_values]
gap_recall = [nc - qp for nc, qp in zip(nc_recall, qp_recall)]

ax.bar(k_values, gap_recall, color="#F59E0B", edgecolor="white", width=2)
ax.axhline(0, color="black", linewidth=0.8)
for i, (k, g) in enumerate(zip(k_values, gap_recall)):
    ax.text(k, g + 0.001, f"+{g:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("k")
ax.set_ylabel("ΔRecall@k  (Non-Convex − Convex QP)")
ax.set_title("Recall@k Relaxation Gap")
ax.set_xticks(k_values)
ax.set_ylim(-0.01, max(gap_recall) * 1.6)

# MRR gap
ax = axes[1]
gap_labels = ["MRR", "Recall@5", "Recall@10"]
gaps = [
    retrieval["Non-Convex"]["MRR"]       - retrieval["Convex QP"]["MRR"],
    retrieval["Non-Convex"]["Recall@5"]  - retrieval["Convex QP"]["Recall@5"],
    retrieval["Non-Convex"]["Recall@10"] - retrieval["Convex QP"]["Recall@10"],
]
bars = ax.bar(gap_labels, gaps, color="#F59E0B", edgecolor="white", width=0.4)
ax.axhline(0, color="black", linewidth=0.8)
for bar, g in zip(bars, gaps):
    ax.text(bar.get_x() + bar.get_width() / 2, g + 0.001,
            f"+{g:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Δ Metric  (Non-Convex − Convex QP)")
ax.set_title("Key Metric Relaxation Gaps")
ax.set_ylim(-0.01, max(gaps) * 1.8)

fig.suptitle("Convex Relaxation Gap: Cost of the Convex Approximation\n"
             "Positive = Non-Convex outperforms; smaller gap = tighter relaxation",
             fontsize=11)
plt.tight_layout()
plt.savefig(out_dir / "fig5_relaxation_gap.png")
plt.close()
print("Saved fig5_relaxation_gap.png")


# ── Fig 6: Generation results ─────────────────────────────────────────────────
if generation:
    fig, ax = plt.subplots(figsize=(5.5, 4))

    gen_systems = list(generation.keys())
    em_vals = [generation[s]["ExactMatch"] for s in gen_systems]
    f1_vals = [generation[s]["F1"]         for s in gen_systems]

    x = np.arange(len(gen_systems))
    b1 = ax.bar(x - 0.18, em_vals, 0.32, label="Exact Match",
                color="#2563EB", edgecolor="white")
    b2 = ax.bar(x + 0.18, f1_vals, 0.32, label="Token F1",
                color="#16A34A", edgecolor="white")

    for bar, val in [(b, v) for bars, vals in [(b1, em_vals), (b2, f1_vals)]
                     for b, v in zip(bars, vals)]:
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10.5,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(gen_systems, rotation=10, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("End-to-End Generation Quality")
    ax.set_ylim(0, 1.0)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_generation.png")
    plt.close()
    print("Saved fig6_generation.png")


# ── Fig 7: Full metrics heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))

metric_keys = ["Recall@1", "Recall@3", "Recall@5", "Recall@10",
               "Precision@1", "Precision@5", "MRR"]
matrix = np.array([[retrieval[sys][m] for m in metric_keys] for sys in systems])

im = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0.0, vmax=1.0)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Score")

ax.set_xticks(range(len(metric_keys)))
ax.set_xticklabels(metric_keys, rotation=30, ha="right")
ax.set_yticks(range(len(systems)))
ax.set_yticklabels(systems)

# Annotate cells
for i in range(len(systems)):
    for j in range(len(metric_keys)):
        val = matrix[i, j]
        color = "white" if val > 0.75 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=9.5, color=color, fontweight="bold")

ax.set_title("Complete Metrics Heatmap: All Systems × All Metrics\n"
             "Darker = higher score", pad=12)
plt.tight_layout()
plt.savefig(out_dir / "fig7_summary_heatmap.png")
plt.close()
print("Saved fig7_summary_heatmap.png")


# ── Fig 8: Lambda sensitivity (optional — requires lambda_results.json) ───────
lambda_results_path = Path("results/lambda_results.json")

if lambda_results_path.exists():
    with open(lambda_results_path) as f:
        lam_records = json.load(f)

    log_lambdas = [r["log10_lambda"] for r in lam_records]
    lambdas_raw = [r["lambda"] for r in lam_records]
    w_cosine    = [r["w_cosine"] for r in lam_records]
    w_bm25      = [r["w_bm25"] for r in lam_records]
    w_diversity = [r["w_diversity"] for r in lam_records]
    obj_vals    = [r["optimal_value"] for r in lam_records]
    recall_vals = [r["recall_at_k"] for r in lam_records]
    k_lam       = lam_records[0]["k"]

    FEAT_COLORS = ["#2563EB", "#DC2626", "#16A34A"]
    FEAT_NAMES  = ["Cosine Similarity", "BM25 Score", "MMR Diversity"]
    default_log = np.log10(cfg["lambda_reg"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Lambda (λ) Regularization Sensitivity Analysis\n"
        "QP: min ‖Xw−y‖² + λ‖w‖²  s.t. w≥0, Σw=1"
        f"  |  TriviaQA: {cfg['num_train_queries']} train / {cfg['num_test_queries']} test queries",
        fontsize=11, y=1.01,
    )

    # (a) Regularization path
    ax = axes[0, 0]
    ax.plot(log_lambdas, w_cosine,   "o-", color=FEAT_COLORS[0], lw=2, ms=6, label=FEAT_NAMES[0])
    ax.plot(log_lambdas, w_bm25,     "s-", color=FEAT_COLORS[1], lw=2, ms=6, label=FEAT_NAMES[1])
    ax.plot(log_lambdas, w_diversity, "^-", color=FEAT_COLORS[2], lw=2, ms=6, label=FEAT_NAMES[2])
    ax.axhline(1/3, color="gray", linestyle=":", lw=1.2, label="Equal weight (1/3)")
    ax.axvline(default_log, color="black", linestyle="--", lw=1, alpha=0.6,
               label=f"λ={cfg['lambda_reg']} (default)")
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("Learned Weight  w*")
    ax.set_title("(a) Regularization Path")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # (b) Objective value
    ax = axes[0, 1]
    ax.plot(log_lambdas, obj_vals, "D-", color="#9333EA", lw=2, ms=6)
    ax.axvline(default_log, color="black", linestyle="--", lw=1, alpha=0.6)
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("QP Objective  ‖Xw*−y‖² + λ‖w*‖²")
    ax.set_title("(b) Objective Value vs λ\n(increases as regularization tightens)")

    # (c) Recall@k
    ax = axes[1, 0]
    ax.plot(log_lambdas, recall_vals, "o-", color="#EA580C", lw=2, ms=6)
    ax.axvline(default_log, color="black", linestyle="--", lw=1, alpha=0.6,
               label=f"λ={cfg['lambda_reg']} (default)")
    max_idx = int(np.argmax(recall_vals))
    recall_range = max(recall_vals) - min(recall_vals)
    if recall_range > 0.001:   # only annotate if there's visible variation
        offset_x = -1.5 if log_lambdas[max_idx] > 0 else 0.5
        ax.annotate(
            f"peak @ λ={lambdas_raw[max_idx]:.4f}\nRecall={recall_vals[max_idx]:.3f}",
            xy=(log_lambdas[max_idx], recall_vals[max_idx]),
            xytext=(log_lambdas[max_idx] + offset_x, recall_vals[max_idx] - recall_range * 0.5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel(f"Recall@{k_lam}  (training set)")
    ax.set_title(f"(c) Training Recall@{k_lam} vs λ\n(proxy for retrieval quality)")
    ax.legend(loc="lower left", fontsize=9)

    # (d) Stacked area
    ax = axes[1, 1]
    log_arr = np.array(log_lambdas)
    ax.stackplot(log_arr,
                 np.array(w_cosine), np.array(w_bm25), np.array(w_diversity),
                 labels=FEAT_NAMES, colors=FEAT_COLORS, alpha=0.75)
    ax.axvline(default_log, color="black", linestyle="--", lw=1.2, alpha=0.8,
               label=f"λ={cfg['lambda_reg']} (default)")
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("Weight Composition")
    ax.set_title("(d) Weight Composition vs λ\n(how regularization blends retrieval signals)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_dir / "fig8_lambda_sensitivity.png")
    plt.close()
    print("Saved fig8_lambda_sensitivity.png")
else:
    print("Skipping fig8: results/lambda_results.json not found. Run main.py to generate it.")


print(f"\nAll figures saved to: {out_dir.resolve()}")
