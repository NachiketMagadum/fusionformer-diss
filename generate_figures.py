"""
Generate the four Chapter 4 figures from the actual experimental output.

Reads notes/all_runs.csv (produced by analyse_wilcoxon.py) and writes:
  figures/fig_4_1_smd_paired.png
  figures/fig_4_2_skab_paired.png
  figures/fig_4_3_paired_diffs.png
  figures/fig_4_4_cross_benchmark.png

Every plotted value is a real Test AUROC from a completed training run.
Nothing is synthesised.

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)
CSV_PATH = Path("notes/all_runs.csv")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif",
})

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREY = "#4a5568"


def load():
    rows = []
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            r["auroc"] = float(r["auroc"])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def paired(rows, dataset):
    full, ablated = {}, {}
    for r in rows:
        if r["dataset"] != dataset:
            continue
        key = (r["id"], r["seed"])
        if r["variant"] == "full":
            full[key] = r["auroc"]
        elif r["variant"] == "no_mswea":
            ablated[key] = r["auroc"]
    keys = sorted(set(full) & set(ablated))
    return keys, np.array([full[k] for k in keys]), np.array([ablated[k] for k in keys])


def fig_smd(rows):
    keys, full, abl = paired(rows, "smd")
    labels = [f"{k[0].replace('machine-', 'm')}\ns{k[1]}" for k in keys]
    x = np.arange(len(keys)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, full, w, label="Full Fusionformer FAM", color=BLUE)
    ax.bar(x + w/2, abl, w, label="MSWEA-off ablation", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test AUROC"); ax.set_ylim(0.75, 1.0)
    ax.axhline(full.mean(), ls=":", c=BLUE, alpha=0.6, lw=1,
               label=f"full mean {full.mean():.4f}")
    ax.axhline(abl.mean(), ls=":", c=ORANGE, alpha=0.6, lw=1,
               label=f"no_mswea mean {abl.mean():.4f}")
    ax.set_title("Figure 4.1: Paired MSWEA ablation on SMD "
                 f"(n = {len(keys)} machine-seed pairs)\n"
                 "Wilcoxon W = 48.5, p = 0.513, r = 0.169 (small, not significant)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_4_1_smd_paired.png"); plt.close()
    print("wrote fig_4_1_smd_paired.png")


def fig_skab(rows):
    keys, full, abl = paired(rows, "skab")
    labels = [f"f{k[0]}\ns{k[1]}" for k in keys]
    x = np.arange(len(keys)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w/2, full, w, label="Full Fusionformer FAM", color=BLUE)
    ax.bar(x + w/2, abl, w, label="MSWEA-off ablation", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("Test AUROC"); ax.set_ylim(0.7, 1.0)
    ax.axhline(full.mean(), ls=":", c=BLUE, alpha=0.6, lw=1,
               label=f"full mean {full.mean():.4f}")
    ax.axhline(abl.mean(), ls=":", c=ORANGE, alpha=0.6, lw=1,
               label=f"no_mswea mean {abl.mean():.4f}")
    ax.set_title("Figure 4.2: Paired MSWEA ablation on SKAB "
                 f"(n = {len(keys)} file-seed pairs)\n"
                 "Wilcoxon W = 9.00, p = 0.002, r = 0.797 (large, significant at 0.01)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_4_2_skab_paired.png"); plt.close()
    print("wrote fig_4_2_skab_paired.png")


def fig_diffs(rows):
    _, smd_full, smd_abl = paired(rows, "smd")
    _, skab_full, skab_abl = paired(rows, "skab")
    smd_d = smd_full - smd_abl
    skab_d = skab_full - skab_abl

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, d, name, colour in [
        (axes[0], smd_d, f"SMD (n = {len(smd_d)})", BLUE),
        (axes[1], skab_d, f"SKAB (n = {len(skab_d)})", ORANGE),
    ]:
        ax.hist(d, bins=10, color=colour, alpha=0.8, edgecolor="black")
        ax.axvline(0, ls="--", c="red", lw=1.5, label="no difference")
        ax.axvline(d.mean(), ls="-", c="black", lw=1.5,
                   label=f"mean Δ = {d.mean():+.4f}")
        ax.set_xlabel("Full AUROC − no_mswea AUROC")
        ax.set_ylabel("count")
        ax.set_title(name)
        ax.legend(fontsize=8, frameon=False)
    plt.suptitle("Figure 4.3: Paired AUROC differences by benchmark", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_4_3_paired_diffs.png"); plt.close()
    print("wrote fig_4_3_paired_diffs.png")


def fig_cross(rows):
    # Bars: two benchmarks, side by side, p-value and effect size annotated
    _, smd_full, smd_abl = paired(rows, "smd")
    _, skab_full, skab_abl = paired(rows, "skab")
    ds_names = ["SMD", "SKAB"]
    n_vals = [len(smd_full), len(skab_full)]
    fulls = [smd_full.mean(), skab_full.mean()]
    ablated = [smd_abl.mean(), skab_abl.mean()]
    diffs = [f - a for f, a in zip(fulls, ablated)]
    p_vals = [0.513, 0.002]  # from analyse_wilcoxon.py
    r_vals = [0.169, 0.797]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ds_names)); w = 0.35
    ax.bar(x - w/2, fulls, w, label="Full", color=BLUE)
    ax.bar(x + w/2, ablated, w, label="no_mswea", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(ds_names, fontsize=11)
    ax.set_ylabel("Mean Test AUROC")
    ax.set_ylim(0.8, 1.0)
    for i, (d, p, r) in enumerate(zip(diffs, p_vals, r_vals)):
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        ax.text(i, max(fulls[i], ablated[i]) + 0.01,
                f"Δ = {d:+.4f}\np = {p:.3f}, r = {r:.2f}  {sig}",
                ha="center", fontsize=9)
    ax.set_title("Figure 4.4: MSWEA contribution across benchmarks\n"
                 "SKAB: significant (** p<0.01, large effect). SMD: not significant.")
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_4_4_cross_benchmark.png"); plt.close()
    print("wrote fig_4_4_cross_benchmark.png")


if __name__ == "__main__":
    rows = load()
    print(f"loaded {len(rows)} runs from {CSV_PATH}")
    fig_smd(rows)
    fig_skab(rows)
    fig_diffs(rows)
    fig_cross(rows)
    print(f"\n4 figures written to {FIGDIR}/")
