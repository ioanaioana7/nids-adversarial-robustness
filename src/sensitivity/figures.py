"""Graficele O4, in aceleasi conventii ca O3 (paleta validata, small multiples)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.evasion.figures import (GRID_COLOR, SEQUENTIAL_HUE, TEXT_PRIMARY,
                                 TEXT_SECONDARY, _style_axes)

# Paleta categorica, ordine fixa per model (validata: CVD dE 9.2 all-pairs pe 3 sloturi).
MODEL_COLORS = {"RandomForest": "#2a78d6", "XGBoost": "#eb6834", "Transformer": "#1baf7a"}
MODEL_MARKERS = {"RandomForest": "o", "XGBoost": "s", "Transformer": "^"}


def plot_importance(importance: pd.DataFrame, path,
                    top_n: int = config.SENSITIVITY_TOP_FEATURES) -> None:
    """Importanta prin permutare: bare orizontale, un panou per model.

    Caracteristicile pe care O2 le poate perturba sunt marcate, ca legatura
    dintre "important" si "atacabil" sa fie vizibila direct in grafic.
    """
    perturbable = set(config.SOURCE_CONTROLLED + config.CONNECTION_RATE_FEATURES
                      + config.DERIVED_FEATURES)
    models = sorted(importance["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 6.4))
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        data = (importance[importance["model"] == model]
                .sort_values("detection_drop", ascending=False).head(top_n).iloc[::-1])
        colors = [MODEL_COLORS[model] if f in perturbable else "#b8b7b1"
                  for f in data["feature"]]
        ax.barh(range(len(data)), data["detection_drop"], color=colors, height=0.72,
                xerr=data["detection_drop_std"], error_kw={"ecolor": "#8a8984", "lw": 0.8})
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data["feature"], fontsize=8, color=TEXT_SECONDARY)
        _style_axes(ax)
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
        ax.grid(False, axis="y")
        ax.set_title(model, color=TEXT_PRIMARY, fontsize=11, pad=8)
        ax.set_xlabel("scaderea ratei de detectie (puncte %)",
                      color=TEXT_SECONDARY, fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color="#52514e"),
               plt.Rectangle((0, 0), 1, 1, color="#b8b7b1")]
    fig.legend(handles, ["perturbabila de O2", "neperturbabila"],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=TEXT_SECONDARY, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Importanta prin permutare — cat pierde detectia daca amestecam o caracteristica",
                 color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def plot_importance_vs_evasion(merged: pd.DataFrame, correlations: pd.DataFrame,
                               path) -> None:
    """Masa de importanta atinsa vs rata de evaziune produsa, un panou per model."""
    models = sorted(merged["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 4.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        data = merged[merged["model"] == model]
        ax.scatter(data["importance_mass_weighted"], data["evasion_rate"],
                   s=70, color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                   edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
        # O singura eticheta directa, pe varianta cu cea mai mare evaziune.
        # Doua sau mai multe se suprapun: punctele extreme sunt aproape coincidente.
        if not data.empty:
            row = data.nlargest(1, "evasion_rate").iloc[0]
            ax.annotate(row["variant"], (row["importance_mass_weighted"], row["evasion_rate"]),
                        textcoords="offset points", xytext=(-8, -14), ha="right",
                        fontsize=7.5, color=TEXT_SECONDARY)
        rho = correlations.loc[correlations["model"] == model,
                               "spearman_importance_mass_weighted"]
        label = f"Spearman rho = {rho.iloc[0]:.2f}" if len(rho) and pd.notna(rho.iloc[0]) else ""
        _style_axes(ax)
        ax.grid(True, axis="both", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
        ax.set_title(f"{model}\n{label}", color=TEXT_PRIMARY, fontsize=10, pad=8)
        ax.set_xlabel("masa de importanta atinsa (ponderata)", color=TEXT_SECONDARY, fontsize=9)

    axes[0].set_ylabel("rata de evaziune (%)", color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle("Perturbarile care ating caracteristici importante produc mai multa evaziune?",
                 color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def plot_outcome_destinations(destinations: pd.DataFrame, path, variant: str,
                              model: str) -> None:
    """Unde aluneca fluxurile confundate cu alt atac — harta granitelor fragile."""
    if destinations.empty:
        return
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL_HUE)
    fig, ax = plt.subplots(figsize=(1.05 * destinations.shape[1] + 3.2,
                                    0.52 * destinations.shape[0] + 2.8))
    image = ax.imshow(destinations.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(destinations.shape[1]))
    ax.set_xticklabels(destinations.columns, rotation=45, ha="right",
                       fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(destinations.shape[0]))
    ax.set_yticklabels(destinations.index, fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("clasa prezisa dupa perturbare", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("clasa reala", color=TEXT_SECONDARY, fontsize=9)

    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("% din fluxurile confundate", color=TEXT_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    bar.outline.set_visible(False)

    ax.set_title(f"Destinatiile confuziei (rezultatul (c)) — {model}, {variant}",
                 color=TEXT_PRIMARY, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
