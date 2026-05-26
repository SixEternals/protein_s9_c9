#!/usr/bin/env python3
"""Generate publication-style prediction comparison figures.

The script reads one YAML config, validates that all prediction files contain
the same test-set labels in the same order, and saves Figure A and Figure C as
PNG/PDF. The config is intentionally JSON-compatible YAML so the script can run
even in plotting environments that do not have PyYAML installed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

mpl_config_dir_value = os.environ.get("MPLCONFIGDIR")
mpl_config_dir = Path(mpl_config_dir_value) if mpl_config_dir_value else Path()
if not mpl_config_dir_value or not os.access(mpl_config_dir, os.W_OK):
    mpl_config_dir = Path("/data/zwf/tmp/matplotlib")
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
mpl_config_dir.mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


OBSERVED = "observed_positive"
UNOBSERVED = "unobserved_candidate"


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        config = yaml.safe_load(text)
    except ModuleNotFoundError:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise TypeError(f"Config must define a mapping: {path}")
    return config


def resolve_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def probability_column(frame: pd.DataFrame, configured_column: str | None) -> str:
    if configured_column:
        if configured_column not in frame.columns:
            raise ValueError(f"Configured probability column is missing: {configured_column}")
        return configured_column

    candidates = [column for column in frame.columns if column != "label"]
    if len(candidates) != 1:
        raise ValueError(
            "Prediction CSV must have exactly one probability column besides label "
            f"when probability_column is omitted. Found: {candidates}"
        )
    return candidates[0]


def load_predictions(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    models = config["models"]
    labels: pd.Series | None = None
    columns: dict[str, pd.Series] = {}
    model_labels: dict[str, str] = {}

    for key, model_config in models.items():
        path = resolve_path(model_config["path"], root)
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        if "label" not in frame.columns:
            raise ValueError(f"Missing label column in {path}")

        current_labels = frame["label"].astype("int8")
        if labels is None:
            labels = current_labels
        elif not labels.equals(current_labels):
            raise ValueError(f"Label column is not aligned with the first model: {path}")

        prob_col = probability_column(frame, model_config.get("probability_column"))
        probabilities = pd.to_numeric(frame[prob_col], errors="raise").astype("float32")
        outside_unit = probabilities[(probabilities < 0.0) | (probabilities > 1.0)]
        if not outside_unit.empty:
            raise ValueError(
                f"Predictions in {path} must be in [0, 1]; "
                f"observed min={probabilities.min()} max={probabilities.max()}"
            )

        columns[key] = probabilities
        model_labels[key] = model_config.get("label", key)

    if labels is None:
        raise ValueError("No models were configured.")

    data = {"label": labels}
    data.update(columns)
    df = pd.DataFrame(data)
    return df, model_labels


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 12,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )


def prepare_plot_samples(
    df: pd.DataFrame, sample_size: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed = df.loc[df["label"] == 1].copy()
    unobserved_all = df.loc[df["label"] == 0].copy()
    if len(unobserved_all) > sample_size:
        unobserved_plot = unobserved_all.sample(n=sample_size, random_state=seed)
    else:
        unobserved_plot = unobserved_all
    return observed, unobserved_all, unobserved_plot


def format_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ticks = np.linspace(0.0, 1.0, 6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])
    ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)


def add_agreement_reference(ax: plt.Axes) -> None:
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", lw=0.8, alpha=0.5)
    ax.text(
        0.58,
        0.40,
        "Perfect Agreement",
        rotation=45,
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
        ha="left",
        va="center",
        alpha=0.9,
    )


def draw_figure_a(
    df: pd.DataFrame,
    observed: pd.DataFrame,
    unobserved_plot: pd.DataFrame,
    model_labels: dict[str, str],
    config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    figure_config = config["figure_a"]
    colors = config["colors"]
    fig, axes = plt.subplots(1, 3, figsize=tuple(figure_config["figsize"]), constrained_layout=False)

    stats_rows: list[dict[str, Any]] = []
    with_kde = bool(figure_config.get("with_kde", True))

    for ax, (x_col, y_col) in zip(axes, figure_config["pairs"]):
        if with_kde:
            sns.kdeplot(
                data=unobserved_plot,
                x=x_col,
                y=y_col,
                levels=6,
                color=colors.get("density", "#9AA0A6"),
                linewidths=0.45,
                alpha=0.28,
                fill=False,
                thresh=0.08,
                ax=ax,
                zorder=0,
            )

        ax.scatter(
            unobserved_plot[x_col],
            unobserved_plot[y_col],
            c=colors[UNOBSERVED],
            s=2,
            alpha=0.15,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        ax.scatter(
            observed[x_col],
            observed[y_col],
            c=colors[OBSERVED],
            s=3,
            alpha=0.4,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

        add_agreement_reference(ax)
        r_value, p_value = stats.pearsonr(df[x_col], df[y_col])
        stats_rows.append(
            {
                "figure": "A",
                "x_model": x_col,
                "y_model": y_col,
                "x_label": model_labels[x_col],
                "y_label": model_labels[y_col],
                "pearson_r": r_value,
                "pearson_p_value": p_value,
                "n_total": len(df),
                "n_observed_positive": int((df["label"] == 1).sum()),
                "n_unobserved_candidate_sampled": len(unobserved_plot),
            }
        )
        ax.text(
            0.95,
            0.05,
            f"Pearson r = {r_value:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666666",
        )

        x_label = f"{model_labels[x_col]} predicted probability"
        y_label = f"{model_labels[y_col]} predicted probability"
        format_axes(ax, x_label, y_label)
        ax.set_title(f"{model_labels[x_col]} vs {model_labels[y_col]}", fontweight="bold", pad=8)

    axes[0].text(
        -0.18,
        1.08,
        "A",
        transform=axes[0].transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=colors[OBSERVED],
            markeredgecolor="none",
            markersize=5,
            alpha=0.8,
            label=f"{OBSERVED} (n={int((df['label'] == 1).sum()):,})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=colors[UNOBSERVED],
            markeredgecolor="none",
            markersize=5,
            alpha=0.55,
            label=f"{UNOBSERVED} (n={len(unobserved_plot):,} sampled)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        handletextpad=0.6,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.22, wspace=0.34)

    basename = figure_config["basename"]
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{basename}.{suffix}", dpi=config["dpi"], bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def build_violin(ax: plt.Axes, values: np.ndarray, center: float, width: float) -> dict[str, Any]:
    parts = ax.violinplot(
        [values],
        positions=[center],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = parts["bodies"][0]
    return {"parts": parts, "body": body}


def draw_group_violin(
    ax: plt.Axes,
    values: np.ndarray,
    center: float,
    color: str,
    hatch: str | None,
    width: float = 0.32,
) -> dict[str, float]:
    violin = build_violin(ax, values, center, width)
    body = violin["body"]
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_linewidth(0.5)
    body.set_alpha(0.92)
    if hatch:
        body.set_hatch(hatch)

    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    box_half_width = width * 0.16
    line_half_width = width * 0.28
    x0, x1 = center - box_half_width, center + box_half_width

    rect = Rectangle(
        (x0, q1),
        x1 - x0,
        q3 - q1,
        facecolor="white",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.22,
        zorder=4,
    )
    ax.add_patch(rect)
    ax.hlines(median, center - line_half_width, center + line_half_width, colors="white", linewidth=1.5, zorder=5)
    ax.hlines([q1, q3], x0, x1, colors="black", linewidth=0.45, alpha=0.45, zorder=5)
    return {"q1": float(q1), "median": float(median), "q3": float(q3)}


def draw_figure_c(
    df: pd.DataFrame,
    model_labels: dict[str, str],
    config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    figure_config = config["figure_c"]
    colors = config["colors"]
    model_order = figure_config["model_order"]
    positions = np.arange(len(model_order), dtype=float)

    fig, ax = plt.subplots(figsize=tuple(figure_config["figsize"]), constrained_layout=False)
    stats_rows: list[dict[str, Any]] = []
    violin_width = 0.32
    pair_offset = 0.24

    for position, model_key in zip(positions, model_order):
        observed_values = df.loc[df["label"] == 1, model_key].to_numpy(dtype=float)
        unobserved_values = df.loc[df["label"] == 0, model_key].to_numpy(dtype=float)

        observed_center = position - pair_offset
        unobserved_center = position + pair_offset
        observed_stats = draw_group_violin(
            ax,
            observed_values,
            center=observed_center,
            color=colors[OBSERVED],
            hatch=None,
            width=violin_width,
        )
        unobserved_stats = draw_group_violin(
            ax,
            unobserved_values,
            center=unobserved_center,
            color=colors[UNOBSERVED],
            hatch="////",
            width=violin_width,
        )

        label_specs = [
            (observed_stats["median"], observed_center, colors[OBSERVED], OBSERVED),
            (unobserved_stats["median"], unobserved_center, colors[UNOBSERVED], UNOBSERVED),
        ]
        for median, text_x, text_color, group in label_specs:
            text_y = min(0.985, max(0.035, median + (0.035 if median < 0.86 else -0.045)))
            ax.text(
                text_x,
                text_y,
                f"median={median:.2f}",
                fontsize=8,
                color="black",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": text_color,
                    "linewidth": 0.4,
                    "alpha": 0.78,
                },
                zorder=7,
            )
            stats_rows.append(
                {
                    "figure": "C",
                    "model": model_key,
                    "model_label": model_labels[model_key],
                    "group": group,
                    "n": int((df["label"] == (1 if group == OBSERVED else 0)).sum()),
                    "q1": observed_stats["q1"] if group == OBSERVED else unobserved_stats["q1"],
                    "median": median,
                    "q3": observed_stats["q3"] if group == OBSERVED else unobserved_stats["q3"],
                }
            )

    format_axes(ax, "Model", "Predicted probability")
    ax.set_xlim(-0.65, len(model_order) - 0.05)
    ax.set_xticks(positions)
    ax.set_xticklabels([model_labels[key] for key in model_order])
    ax.set_title("")
    ax.text(
        -0.11,
        1.03,
        "C",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    handles = [
        Patch(facecolor=colors[OBSERVED], edgecolor="black", linewidth=0.5, label=OBSERVED),
        Patch(
            facecolor=colors[UNOBSERVED],
            edgecolor="black",
            linewidth=0.5,
            hatch="////",
            label=UNOBSERVED,
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.12)

    basename = figure_config["basename"]
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{basename}.{suffix}", dpi=config["dpi"], bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def write_summary(
    df: pd.DataFrame,
    stats_a: pd.DataFrame,
    stats_c: pd.DataFrame,
    output_dir: Path,
) -> None:
    label_counts = df["label"].map({0: UNOBSERVED, 1: OBSERVED}).value_counts().rename_axis("group")
    label_counts.to_csv(output_dir / "prediction_figure_label_counts.csv", header=["n"])
    stats_a.to_csv(output_dir / "figure_A_prediction_correlation_stats.csv", index=False)
    stats_c.to_csv(output_dir / "figure_C_prediction_violin_stats.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/figure_prediction_distributions.yaml",
        help="Path to the JSON-compatible YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    config_path = resolve_path(args.config, root)
    config = load_config(config_path)

    configure_style()
    output_dir = resolve_path(config["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, model_labels = load_predictions(config, root)
    observed, _, unobserved_plot = prepare_plot_samples(
        df,
        sample_size=int(config["unobserved_sample_size"]),
        seed=int(config["seed"]),
    )

    stats_a = draw_figure_a(df, observed, unobserved_plot, model_labels, config, output_dir)
    stats_c = draw_figure_c(df, model_labels, config, output_dir)
    write_summary(df, stats_a, stats_c, output_dir)

    print(f"Loaded {len(df):,} aligned test samples")
    print(f"{OBSERVED}: {(df['label'] == 1).sum():,}")
    print(f"{UNOBSERVED}: {(df['label'] == 0).sum():,}")
    print(f"Figure outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
