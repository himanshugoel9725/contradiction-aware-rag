#!/usr/bin/env python3
"""
Generate publication-quality figures from experiment results.

USAGE:
    # Figures for a single run (per-run figures only)
    python scripts/generate_figures.py --run-id <run_id>

    # Cross-run figures (strategies, ablation, backbone, retrieval, cross-dataset)
    python scripts/generate_figures.py --figure-set strategies
    python scripts/generate_figures.py --figure-set ablation
    python scripts/generate_figures.py --figure-set backbone
    python scripts/generate_figures.py --figure-set retrieval
    python scripts/generate_figures.py --figure-set cross_dataset
    python scripts/generate_figures.py --figure-set all

    # Override output directory
    python scripts/generate_figures.py --figure-set all --output-dir figures/

FIGURES:
    Fig 1  CBR baseline bar       runs/experiment_1_1_*/metrics.json
    Fig 2  Strategy comparison    runs/gap1_strategy_{A,B,C,D}*/metrics.json
    Fig 3  Ablation waterfall     runs/gap4_*/metrics.json
    Fig 4  Backbone comparison    runs/gap5_*/metrics.json
    Fig 5  Retrieval comparison   runs/gap7_retrieval_*/metrics.json
    Fig 6  Cross-dataset HC vs SciFact
    Fig 7  Pipeline architecture  (static, rendered with matplotlib)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.paper4.utils.env_setup  # noqa: F401, E402

import typer  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402

app = typer.Typer(help="Generate publication figures")

# ── Colour palette ─────────────────────────────────────────────────────────────
_PALETTE = {
    "A": "#e74c3c",
    "B": "#e67e22",
    "C": "#2ecc71",
    "D": "#3498db",
    "bm25": "#9b59b6",
    "embedding": "#1abc9c",
    "hybrid": "#2ecc71",
    "mmr": "#f39c12",
    "random": "#95a5a6",
    "healthcontradict": "#2980b9",
    "scifact": "#27ae60",
    "baseline": "#e74c3c",
    "default": "#34495e",
}
_METRIC_COLORS = {"CBR": "#e74c3c", "VCS": "#2ecc71", "EAA": "#3498db",
                  "CAS": "#9b59b6", "SEAA": "#f39c12"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _extract_point(metrics: dict, key: str) -> float:
    """Extract point estimate from metrics dict."""
    entry = metrics.get(key, {})
    if isinstance(entry, dict):
        ci = entry.get("bootstrap_ci", {})
        if ci:
            return float(ci.get("point_estimate", 0.0))
        return float(entry.get("mean", entry.get(key, 0.0)))
    return float(entry)


def _extract_ci(metrics: dict, key: str) -> tuple[float, float]:
    """Return (lower_err, upper_err) for error bars."""
    entry = metrics.get(key, {})
    if isinstance(entry, dict):
        ci = entry.get("bootstrap_ci", {})
        pt = float(ci.get("point_estimate", 0.0))
        return pt - float(ci.get("ci_lower", pt)), float(ci.get("ci_upper", pt)) - pt
    return 0.0, 0.0


def _save(fig: plt.Figure, output_dir: Path, name: str, fmt: str = "both") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt in ("png", "both"):
        fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    if fmt in ("pdf", "both"):
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def _glob_latest(runs_dir: Path, pattern: str) -> list[Path]:
    """Glob for run directories matching pattern, sorted by name."""
    return sorted(runs_dir.glob(pattern))


# ── Fig 1: CBR Baseline ────────────────────────────────────────────────────────

def fig1_cbr_baseline(runs_dir: Path, output_dir: Path, fmt: str) -> None:
    """Fig 1: CBR bar for the 1_1 CBR baseline run."""
    dirs = _glob_latest(runs_dir, "experiment_1_1_*")
    if not dirs:
        print("[Fig 1] No experiment_1_1_* run found, skipping.")
        return

    metrics = _load_metrics(dirs[-1] / "metrics.json")
    cbr_val = _extract_point(metrics, "cbr")
    yerr = _extract_ci(metrics, "cbr")

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["CBR Baseline"], [cbr_val], color=[_PALETTE["baseline"]], width=0.4,
                  yerr=[[yerr[0]], [yerr[1]]], capsize=6, error_kw={"linewidth": 1.5})
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6, label="Random (0.5)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CBR (lower is better)")
    ax.set_title("Fig 1 — CBR Baseline")
    ax.legend(fontsize=9)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                f"{cbr_val:.3f}", ha="center", fontsize=10, fontweight="bold")

    _save(fig, output_dir, "fig1_cbr_baseline", fmt)


# ── Fig 2: Strategy Comparison ─────────────────────────────────────────────────

def fig2_strategy_comparison(runs_dir: Path, output_dir: Path, fmt: str) -> None:
    """Fig 2: Metrics for strategies A–D side by side."""
    strategies = ["A", "B", "C", "D"]
    metric_keys = ["cbr", "vcs", "eaa"]
    metric_labels = ["CBR", "VCS", "EAA"]

    data: dict[str, list[float]] = {s: [] for s in strategies}
    errs: dict[str, list[tuple[float, float]]] = {s: [] for s in strategies}
    found: list[str] = []

    for s in strategies:
        dirs = _glob_latest(runs_dir, f"gap1_strategy_{s}*")
        if not dirs:
            dirs = _glob_latest(runs_dir, f"experiment_gap1_strategy_{s}*")
        if not dirs:
            print(f"[Fig 2] No run found for strategy {s}, filling zeros.")
            m = {}
        else:
            m = _load_metrics(dirs[-1] / "metrics.json")
            found.append(s)

        for mk in metric_keys:
            data[s].append(_extract_point(m, mk))
            errs[s].append(_extract_ci(m, mk))

    if not found:
        print("[Fig 2] No strategy runs found, skipping.")
        return

    n_metrics = len(metric_keys)
    n_strategies = len(strategies)
    x = np.arange(n_metrics)
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(strategies):
        offset = (i - (n_strategies - 1) / 2) * width
        vals = data[s]
        lo = [e[0] for e in errs[s]]
        hi = [e[1] for e in errs[s]]
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=_PALETTE.get(s, _PALETTE["default"]), label=f"Strategy {s}",
                      yerr=[lo, hi], capsize=4, error_kw={"linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Fig 2 — Generation Strategy Comparison")
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=4, fontsize=9, loc="upper right")
    ax.axhline(0.0, color="black", linewidth=0.5)

    _save(fig, output_dir, "fig2_strategy_comparison", fmt)


# ── Fig 3: Ablation Waterfall ──────────────────────────────────────────────────

def fig3_ablation_waterfall(runs_dir: Path, output_dir: Path, fmt: str) -> None:
    """Fig 3: Contribution of each pipeline component to VCS."""
    conditions = [
        ("full", "Full pipeline"),
        ("minus_pico", "−PICO"),
        ("minus_stance", "−Stance"),
        ("minus_quality", "−Quality"),
        ("minus_selection", "−Selection"),
    ]

    vals: list[float] = []
    labels: list[str] = []

    for cond_id, cond_label in conditions:
        dirs = _glob_latest(runs_dir, f"gap4_{cond_id}*")
        if not dirs:
            dirs = _glob_latest(runs_dir, f"*ablation*{cond_id}*")
        if dirs:
            m = _load_metrics(dirs[-1] / "metrics.json")
            vals.append(_extract_point(m, "vcs"))
        else:
            vals.append(0.0)
        labels.append(cond_label)

    if all(v == 0.0 for v in vals):
        print("[Fig 3] No ablation runs found, skipping.")
        return

    # Waterfall: bars start from prior value
    fig, ax = plt.subplots(figsize=(9, 5))
    full_val = vals[0]
    colors = []
    bottoms = []
    heights = []

    for i, v in enumerate(vals):
        if i == 0:
            heights.append(v)
            bottoms.append(0.0)
            colors.append(_PALETTE["C"])
        else:
            delta = v - full_val
            heights.append(abs(delta))
            bottoms.append(min(v, full_val))
            colors.append(_PALETTE["baseline"] if delta < 0 else _PALETTE["embedding"])

    ax.bar(range(len(vals)), heights, bottom=bottoms, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(full_val, ls="--", color="gray", alpha=0.5, linewidth=1)

    for i, (v, label) in enumerate(zip(vals, labels)):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("VCS (higher is better)")
    ax.set_title("Fig 3 — Ablation Study: Component Contributions to VCS")
    ax.set_ylim(0, 1.15)

    _save(fig, output_dir, "fig3_ablation_waterfall", fmt)


# ── Fig 4: Backbone Comparison ─────────────────────────────────────────────────

def fig4_backbone_comparison(runs_dir: Path, output_dir: Path, fmt: str) -> None:
    """Fig 4: VCS/EAA/CAS for different LLM backbones."""
    backbone_runs = [
        ("sonnet", "Claude Sonnet"),
        ("haiku", "Claude Haiku"),
        ("gpt4o_mini", "GPT-4o mini"),
    ]
    metric_keys = ["vcs", "eaa"]
    metric_labels = ["VCS", "EAA"]

    data: dict[str, list[float]] = {}
    found: list[str] = []

    for bk_id, bk_label in backbone_runs:
        dirs = (
            _glob_latest(runs_dir, f"gap5_{bk_id}*") or
            _glob_latest(runs_dir, f"gap5_backbone_{bk_id}*") or
            _glob_latest(runs_dir, f"*backbone*{bk_id}*")
        )
        if dirs:
            m = _load_metrics(dirs[-1] / "metrics.json")
            found.append(bk_label)
        else:
            m = {}
        data[bk_label] = [_extract_point(m, mk) for mk in metric_keys]

    if not found:
        print("[Fig 4] No backbone runs found, skipping.")
        return

    x = np.arange(len(metric_keys))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, bk_label) in enumerate(backbone_runs):
        offset = (i - 1) * width
        color = list(_PALETTE.values())[i % len(_PALETTE)]
        ax.bar(x + offset, data[bk_label], width=width * 0.9, color=color, label=bk_label)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Fig 4 — LLM Backbone Comparison")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)

    _save(fig, output_dir, "fig4_backbone_comparison", fmt)


# ── Fig 5: Retrieval Strategy Comparison ──────────────────────────────────────

def fig5_retrieval_comparison(runs_dir: Path, output_dir: Path, fmt: str) -> None:
    """Fig 5: VCS + stance coverage across retrieval strategies."""
    retrieval_strats = ["bm25", "embedding", "hybrid", "mmr", "random"]
    labels = ["BM25", "Embedding", "Hybrid", "MMR", "Random"]
    metric_keys = ["vcs", "eaa"]

    data: dict[str, dict[str, float]] = {}
    found = []

    for strat, lbl in zip(retrieval_strats, labels):
        dirs = (
            _glob_latest(runs_dir, f"gap7_{strat}*") or
            _glob_latest(runs_dir, f"gap7_retrieval_{strat}*") or
            _glob_latest(runs_dir, f"*retrieval*{strat}*")
        )
        if dirs:
            m = _load_metrics(dirs[-1] / "metrics.json")
            found.append(lbl)
        else:
            m = {}
        data[lbl] = {mk: _extract_point(m, mk) for mk in metric_keys}

    if not found:
        print("[Fig 5] No retrieval strategy runs found, skipping.")
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    vcs_vals = [data[l]["vcs"] for l in labels]
    eaa_vals = [data[l]["eaa"] for l in labels]

    ax.bar(x - width / 2, vcs_vals, width=width * 0.95, label="VCS",
           color=[_PALETTE.get(s, _PALETTE["default"]) for s in retrieval_strats])
    ax.bar(x + width / 2, eaa_vals, width=width * 0.95, label="EAA",
           color=[_PALETTE.get(s, _PALETTE["default"]) for s in retrieval_strats],
           alpha=0.65, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Fig 5 — Retrieval Strategy Comparison")
    ax.set_ylim(0, 1.15)
    vcs_patch = mpatches.Patch(color="gray", label="VCS (solid)")
    eaa_patch = mpatches.Patch(color="gray", hatch="//", alpha=0.65, label="EAA (hatched)")
    ax.legend(handles=[vcs_patch, eaa_patch], fontsize=9)

    _save(fig, output_dir, "fig5_retrieval_comparison", fmt)


# ── Fig 6: Cross-dataset ───────────────────────────────────────────────────────

def fig6_cross_dataset(runs_dir: Path, output_dir: Path, fmt: str) -> None:
    """Fig 6: HC vs SciFact metric comparison for Strategy C."""
    hc_dirs = (
        _glob_latest(runs_dir, "gap1_strategy_C*") or
        _glob_latest(runs_dir, "experiment_gap1_strategy_C*")
    )
    sf_dirs = (
        _glob_latest(runs_dir, "gap8_scifact_gold*") or
        _glob_latest(runs_dir, "gap8_scifact*")
    )

    hc_metrics = _load_metrics(hc_dirs[-1] / "metrics.json") if hc_dirs else {}
    sf_metrics = _load_metrics(sf_dirs[-1] / "metrics.json") if sf_dirs else {}

    if not hc_metrics and not sf_metrics:
        print("[Fig 6] No cross-dataset runs found, skipping.")
        return

    metric_keys = ["cbr", "vcs", "eaa"]
    metric_labels = ["CBR", "VCS", "EAA"]
    hc_vals = [_extract_point(hc_metrics, mk) for mk in metric_keys]
    sf_vals = [_extract_point(sf_metrics, mk) for mk in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, hc_vals, width=width * 0.95, label="HealthContradict",
           color=_PALETTE["healthcontradict"])
    ax.bar(x + width / 2, sf_vals, width=width * 0.95, label="SciFact",
           color=_PALETTE["scifact"])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Fig 6 — Cross-Dataset Generalisation (Strategy C)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)

    _save(fig, output_dir, "fig6_cross_dataset", fmt)


# ── Fig 7: Pipeline Architecture ──────────────────────────────────────────────

def fig7_pipeline_architecture(output_dir: Path, fmt: str) -> None:
    """Fig 7: Static pipeline architecture flowchart."""
    stages = [
        ("Query", "#bdc3c7"),
        ("PICO Extraction", "#3498db"),
        ("Retrieval\n(BM25/Embed/Hybrid)", "#9b59b6"),
        ("Stance\nClassification", "#e67e22"),
        ("Evidence\nSelection", "#e74c3c"),
        ("GRADE\nQuality Tagging", "#f39c12"),
        ("Synthesis\n(Strategy A–D)", "#2ecc71"),
        ("Post-check\nVerification", "#1abc9c"),
        ("Response", "#bdc3c7"),
    ]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, len(stages) + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")

    box_w, box_h = 0.85, 0.8
    for i, (label, color) in enumerate(stages):
        x_center = i + 0.75
        rect = mpatches.FancyBboxPatch(
            (x_center - box_w / 2, 0.2), box_w, box_h,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor="#2c3e50", facecolor=color, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x_center, 0.6, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold", wrap=True)

        if i < len(stages) - 1:
            ax.annotate(
                "", xy=(x_center + box_w / 2 + 0.12, 0.6),
                xytext=(x_center + box_w / 2, 0.6),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5),
            )

    ax.set_title("Fig 7 — Contradiction-Aware RAG Pipeline Architecture",
                 fontsize=11, fontweight="bold", y=1.02)

    _save(fig, output_dir, "fig7_pipeline_architecture", fmt)


# ── Entry point ────────────────────────────────────────────────────────────────

FigureSet = Literal["strategies", "ablation", "backbone", "retrieval", "cross_dataset", "architecture", "all"]


@app.command()
def generate(
    run_id: str = typer.Option(None, "--run-id", "-r", help="Single run ID (for per-run figures)"),
    runs_dir: str = typer.Option("./runs", "--runs-dir"),
    output_dir: str = typer.Option("./figures", "--output-dir", "-o", help="Top-level figure output directory"),
    figure_set: str = typer.Option("all", "--figure-set", "-s",
                                   help="Which figures: strategies, ablation, backbone, retrieval, cross_dataset, architecture, all"),
    format: str = typer.Option("both", "--format", "-f", help="Output format: png, pdf, or both"),
) -> None:
    """Generate publication figures for one or all experiments."""
    out = Path(output_dir)
    rdir = Path(runs_dir)

    print(f"[Figures] Output: {out}  |  figure-set: {figure_set}  |  format: {format}")

    do_all = figure_set == "all"

    if do_all or figure_set == "strategies":
        fig1_cbr_baseline(rdir, out, format)
        fig2_strategy_comparison(rdir, out, format)

    if do_all or figure_set == "ablation":
        fig3_ablation_waterfall(rdir, out, format)

    if do_all or figure_set == "backbone":
        fig4_backbone_comparison(rdir, out, format)

    if do_all or figure_set == "retrieval":
        fig5_retrieval_comparison(rdir, out, format)

    if do_all or figure_set == "cross_dataset":
        fig6_cross_dataset(rdir, out, format)

    if do_all or figure_set == "architecture":
        fig7_pipeline_architecture(out, format)

    print(f"\n[Figures] Done. Files in {out}/")


if __name__ == "__main__":
    app()

