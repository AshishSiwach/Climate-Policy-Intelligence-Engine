"""
Generate publishable charts for the 4-config retrieval ablation.

Produces:
    docs/charts/ablation_01_aggregate.png     — Table 1
    docs/charts/ablation_02_by_query_type.png — Table 2
    docs/charts/ablation_03_retrieval.png     — Table 3
    docs/charts/ablation_04_negatives.png     — Table 4
    docs/charts/ablation_composite.png        — 2x2 grid for LinkedIn

Run:
    uv run python scripts/plot_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette; config-locked so colors are consistent
# across every chart in this file.
CONFIG_COLORS = {
    "bm25":          "#999999",   # neutral gray — weakest
    "dense":         "#56B4E9",   # light blue
    "hybrid":        "#009E73",   # green — v1 winner
    "hybrid_rerank": "#E69F00",   # orange — attention
}
CONFIG_LABELS = {
    "bm25":          "BM25 only",
    "dense":         "Dense only",
    "hybrid":        "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + Rerank",
}
CONFIG_ORDER = ["bm25", "dense", "hybrid", "hybrid_rerank"]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.8,
})

SOURCE_PATH = Path("data/eval/results/ablation_20260807T144600Z.json")
OUT_DIR = Path("docs/charts")
FOOTNOTE = "CPIE ablation · 47 ground-truth queries · gpt-4o-mini synthesis · gpt-5.4-mini judge"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ablation():
    with open(SOURCE_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_latency(data, cfg):
    return data["per_config"][cfg]["retrieval_latency_ms_mean"]


def get_overall(data, cfg, metric):
    return data["per_config"][cfg]["overall_judge"].get(metric, 0.0)


def get_type_metric(data, cfg, qtype, metric):
    return data["per_config"][cfg]["by_query_type"].get(qtype, {}).get(metric, 0.0)


def get_negatives(data, cfg):
    return data["per_config"][cfg]["negatives"]


# ---------------------------------------------------------------------------
# Shared plotting helper — grouped bars
# ---------------------------------------------------------------------------

def _grouped_bars(ax, categories, series, colors, labels, ylim=None, ylabel=None,
                  fmt="{:.2f}", annotate=True):
    """Draw grouped bar chart with value annotations above each bar."""
    n_cat = len(categories)
    n_ser = len(series)
    bar_w = 0.8 / n_ser
    x = np.arange(n_cat)

    for i, (ser_name, values) in enumerate(zip(labels, series)):
        offset = (i - (n_ser - 1) / 2) * bar_w
        bars = ax.bar(x + offset, values, bar_w,
                      color=colors[i], label=ser_name, edgecolor="white",
                      linewidth=0.6)
        if annotate:
            for bar, v in zip(bars, values):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (ylim[1] * 0.01 if ylim else 0.02),
                            fmt.format(v), ha="center", va="bottom", fontsize=7.5,
                            color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylim:
        ax.set_ylim(ylim)
    if ylabel:
        ax.set_ylabel(ylabel)


# ---------------------------------------------------------------------------
# Chart 1 — Aggregate results (Table 1)
# ---------------------------------------------------------------------------

def chart_01_aggregate(data, ax=None, standalone=True):
    metrics = ["correctness", "faithfulness", "completeness", "refusal_appropriateness"]
    metric_labels = ["Correctness", "Faithfulness", "Completeness", "Refusal Appr."]

    series = [
        [get_overall(data, cfg, m) for m in metrics]
        for cfg in CONFIG_ORDER
    ]
    colors = [CONFIG_COLORS[c] for c in CONFIG_ORDER]

    # Latency-annotated config labels
    labels = [
        f"{CONFIG_LABELS[c]} · {get_latency(data, c):.0f}ms"
        for c in CONFIG_ORDER
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
    else:
        fig = None

    _grouped_bars(ax, metric_labels, series, colors, labels,
                  ylim=(0, 5.4), ylabel="Judge score (1–5)")

    ax.set_title("Aggregate judge scores across 47 queries",
                 loc="left", fontweight="bold", fontsize=12, pad=12)
    ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=8.5)
    ax.set_axisbelow(True)

    if standalone:
        fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color="#666")
        fig.tight_layout()
        out = OUT_DIR / "ablation_01_aggregate.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out}")
        return out


# ---------------------------------------------------------------------------
# Chart 2 — Correctness by query type (Table 2)
# ---------------------------------------------------------------------------

def chart_02_by_query_type(data, ax=None, standalone=True):
    qtypes = ["factual", "cross_document", "numeric", "summarisation", "negative"]
    qtype_labels = [
        "factual\n(n=28)", "cross_document\n(n=4)", "numeric\n(n=4)",
        "summarisation\n(n=2)", "negative\n(n=9)",
    ]

    series = [
        [get_type_metric(data, cfg, qt, "correctness") for qt in qtypes]
        for cfg in CONFIG_ORDER
    ]
    colors = [CONFIG_COLORS[c] for c in CONFIG_ORDER]
    labels = [CONFIG_LABELS[c] for c in CONFIG_ORDER]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5.5))
    else:
        fig = None

    _grouped_bars(ax, qtype_labels, series, colors, labels,
                  ylim=(0, 5.4), ylabel="Correctness (1–5)")

    ax.set_title("Correctness by query type — where the aggregate tie breaks",
                 loc="left", fontweight="bold", fontsize=12, pad=12)
    ax.legend(loc="lower right", ncol=4, frameon=False, fontsize=8.5)

    if standalone:
        fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color="#666")
        fig.tight_layout()
        out = OUT_DIR / "ablation_02_by_query_type.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out}")
        return out


# ---------------------------------------------------------------------------
# Chart 3 — Retrieval metrics (Table 3)
# ---------------------------------------------------------------------------

def chart_03_retrieval(data, ax=None, standalone=True):
    metrics = ["hit@5", "recall@5", "precision@5", "mrr@5", "ndcg@5"]
    metric_labels = ["Hit@5", "Recall@5", "Precision@5", "MRR@5", "nDCG@5"]

    series = [
        [get_overall(data, cfg, m) for m in metrics]
        for cfg in CONFIG_ORDER
    ]
    colors = [CONFIG_COLORS[c] for c in CONFIG_ORDER]
    labels = [CONFIG_LABELS[c] for c in CONFIG_ORDER]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5.5))
    else:
        fig = None

    _grouped_bars(ax, metric_labels, series, colors, labels,
                  ylim=(0, 1.05), ylabel="Score (0–1)",
                  fmt="{:.2f}")

    ax.set_title("Retrieval metrics — chunk-level ranking quality",
                 loc="left", fontweight="bold", fontsize=12, pad=12)
    ax.legend(loc="lower right", ncol=4, frameon=False, fontsize=8.5)

    if standalone:
        fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color="#666")
        fig.tight_layout()
        out = OUT_DIR / "ablation_03_retrieval.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out}")
        return out


# ---------------------------------------------------------------------------
# Chart 4 — Negatives (Table 4)
# ---------------------------------------------------------------------------

def chart_04_negatives(data, ax=None, standalone=True):
    rates = [get_negatives(data, c)["handled_well_rate"] for c in CONFIG_ORDER]
    means = [get_negatives(data, c)["mean_refusal_appropriateness"] for c in CONFIG_ORDER]
    labels = [CONFIG_LABELS[c] for c in CONFIG_ORDER]
    colors = [CONFIG_COLORS[c] for c in CONFIG_ORDER]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = None

    x = np.arange(len(labels))
    bars = ax.bar(x, rates, 0.6, color=colors, edgecolor="white", linewidth=0.8)
    for bar, r, m in zip(bars, rates, means):
        # rate on top
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{r:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        # mean refusal_appr inside bar
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"mean refusal appr.\n{m:.2f}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Handled-well rate (refusal appr. ≥ 4)")
    ax.set_title("Out-of-corpus negatives — did the system refuse when it should?",
                 loc="left", fontweight="bold", fontsize=12, pad=12)
    ax.axhline(y=1.0, linestyle=":", color="#999", linewidth=0.8)
    ax.text(len(labels) - 0.5, 1.01, "target = 100%",
            ha="right", va="bottom", fontsize=8, color="#999")

    if standalone:
        fig.text(0.5, -0.02, FOOTNOTE + "  · n=9 negatives", ha="center",
                 fontsize=7.5, color="#666")
        fig.tight_layout()
        out = OUT_DIR / "ablation_04_negatives.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out}")
        return out


# ---------------------------------------------------------------------------
# Composite — 2x2 grid for LinkedIn
# ---------------------------------------------------------------------------

def chart_composite(data):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.18,
                          left=0.06, right=0.98, top=0.90, bottom=0.06)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    chart_01_aggregate(data, ax=axes[0], standalone=False)
    chart_02_by_query_type(data, ax=axes[1], standalone=False)
    chart_03_retrieval(data, ax=axes[2], standalone=False)
    chart_04_negatives(data, ax=axes[3], standalone=False)

    fig.suptitle("CPIE — 4-config retrieval ablation",
                 x=0.06, y=0.965, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.935,
             "47 ground-truth queries · Hybrid RRF wins on Recall@5, cross-doc, and negatives · "
             "Reranker: 5× slower, no aggregate gain",
             ha="left", fontsize=10.5, color="#444")
    fig.text(0.5, 0.015, FOOTNOTE, ha="center", fontsize=8.5, color="#666")

    out = OUT_DIR / "ablation_composite.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not SOURCE_PATH.exists():
        raise SystemExit(f"Ablation source not found: {SOURCE_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_ablation()
    print(f"Loaded {data['n_queries']} queries × {data['n_configs']} configs from {SOURCE_PATH.name}")
    print(f"Writing to {OUT_DIR}/")

    chart_01_aggregate(data)
    chart_02_by_query_type(data)
    chart_03_retrieval(data)
    chart_04_negatives(data)
    chart_composite(data)

    print("Done.")


if __name__ == "__main__":
    main()
