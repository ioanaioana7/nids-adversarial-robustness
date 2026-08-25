"""Graficele O3.

Alegeri de reprezentare, motivate:

  - Fiecare model primeste propriul panou (small multiples). Suprapunerea a trei
    modele x sapte tipuri de perturbare intr-un singur grafic ar amesteca doua
    dimensiuni diferite in aceeasi culoare si ar deveni ilizibila.

  - Perechile hold/mimic se deseneaza ca BANDA intre cele doua margini, nu ca doua
    linii separate. Banda e chiar rezultatul de raportat (ct_state_ttl nu poate fi
    recalculat, vezi README), deci reprezentarea transmite metodologia in loc sa o
    ascunda; in plus reduce numarul de serii de la 7 la 5.

  - Culorile sunt atribuite in ordine fixa per tip de perturbare, niciodata ciclic,
    ca un tip sa pastreze aceeasi culoare in toate figurile. Textul ramane in
    culori de text, nu in culoarea seriei.

  - Cele trei rezultate folosesc culori de STARE (corect / confundat / evadat),
    distincte de paleta categorica, insotite mereu de eticheta.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

# Paleta categorica, in ordine fixa (validata: CVD dE 9.1, normal-vision 19.6).
SERIES_COLORS = {
    "ttl": "#2a78d6",              # blue
    "padding": "#eb6834",          # orange
    "timing": "#1baf7a",           # aqua
    "connection_rate": "#eda100",  # yellow
    "combined": "#e87ba4",         # magenta
}
SERIES_MARKERS = {"ttl": "o", "padding": "s", "timing": "^",
                  "connection_rate": "D", "combined": "v"}

# Culori de stare pentru cele trei rezultate — nu se amesteca cu paleta categorica.
OUTCOME_COLORS = {
    config.OUTCOME_CORRECT: "#0ca30c",                 # good
    config.OUTCOME_MISCLASSIFIED_ATTACK: "#fab219",    # warning
    config.OUTCOME_EVADED: "#d03b3b",                  # critical
}
OUTCOME_LABELS = {
    config.OUTCOME_CORRECT: "(a) clasificat corect",
    config.OUTCOME_MISCLASSIFIED_ATTACK: "(c) confundat cu alt atac",
    config.OUTCOME_EVADED: "(b) evaziune -> Normal",
}

SEQUENTIAL_HUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"


def _style_axes(ax) -> None:
    """Grid si axe recesive, ca marcajele de date sa domine vizual."""
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def plot_evasion_curves(summary: pd.DataFrame, path) -> None:
    """Rata de evaziune in functie de nivelul de intensitate, un panou per model.

    Perechile hold/mimic apar ca banda intre marginea conservatoare si cea optimista.
    """
    models = sorted(summary["model"].unique())
    bounded = {lo for lo, hi in config.EVASION_BOUND_PAIRS.values()} | \
              {hi for lo, hi in config.EVASION_BOUND_PAIRS.values()}
    singles = [t for t in ["padding", "timing", "connection_rate"]
               if t in set(summary["type"])]

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        data = summary[(summary["model"] == model) & (summary["level"] > 0)
                       & summary["realizable"]]

        for family, (lower_type, upper_type) in config.EVASION_BOUND_PAIRS.items():
            low = data[data["type"] == lower_type].sort_values("level")
            high = data[data["type"] == upper_type].sort_values("level")
            if low.empty or high.empty:
                continue
            colour = SERIES_COLORS[family]
            ax.fill_between(low["level"], low["evasion_rate"], high["evasion_rate"],
                            color=colour, alpha=0.18, linewidth=0)
            for frame, style in ((low, "--"), (high, "-")):
                ax.plot(frame["level"], frame["evasion_rate"], style, color=colour,
                        linewidth=2, marker=SERIES_MARKERS[family], markersize=5,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.0)

        for ptype in singles:
            frame = data[data["type"] == ptype].sort_values("level")
            if frame.empty:
                continue
            ax.plot(frame["level"], frame["evasion_rate"], "-", color=SERIES_COLORS[ptype],
                    linewidth=2, marker=SERIES_MARKERS[ptype], markersize=5,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.0)

        _style_axes(ax)
        ax.set_title(model, color=TEXT_PRIMARY, fontsize=11, pad=8)
        ax.set_xlabel("nivel de intensitate", color=TEXT_SECONDARY, fontsize=9)
        ax.set_xticks(sorted(data["level"].unique()))

    axes[0].set_ylabel("rata de evaziune (%)", color=TEXT_SECONDARY, fontsize=9)

    handles = [plt.Line2D([], [], color=SERIES_COLORS[f], marker=SERIES_MARKERS[f],
                          linewidth=2, markersize=5,
                          label=f"{f} (banda hold-mimic)" if f in config.EVASION_BOUND_PAIRS
                          else f)
               for f in ["ttl", "combined"] + singles]
    # Fara asta, cititorul nu poate sti care margine a benzii e care.
    handles += [
        plt.Line2D([], [], color=TEXT_SECONDARY, linestyle="-", linewidth=2,
                   label="margine mimic (optimista)"),
        plt.Line2D([], [], color=TEXT_SECONDARY, linestyle="--", linewidth=2,
                   label="margine hold (conservatoare)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=TEXT_SECONDARY, bbox_to_anchor=(0.5, -0.10))

    fig.suptitle("Rata de evaziune pe niveluri de perturbare (fluxuri eligibile)",
                 color=TEXT_PRIMARY, fontsize=12)
    # Nivelurile TTL sunt valori tinta, NU o scara de intensitate crescatoare:
    # nivelul 2 (sttl=62) e o valoare asociata atacurilor, nu traficului normal,
    # deci nu ajuta atacatorul. Fara nota, scaderea de la nivelul 2 pare o eroare.
    ttl_note = (f"Niveluri TTL = valori tinta sttl {tuple(config.TTL_LEVELS)}, nu intensitate "
                f"crescatoare: nivelul 2 (sttl={config.TTL_LEVELS[1]}) este o valoare asociata "
                f"traficului de atac, nu celui normal — de aici scaderea.")
    fig.text(0.5, 0.925, ttl_note, ha="center", va="top",
             color=TEXT_SECONDARY, fontsize=8.5)

    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def plot_outcome_composition(composition: pd.DataFrame, path, model: str) -> None:
    """Compozitia celor trei rezultate, bare stivuite per varianta realizabila."""
    data = composition[(composition["model"] == model) & composition["realizable"]
                       & (composition["level"] > 0)].copy()
    data = data.sort_values(["type", "level"])
    labels = [f"{r.type}\nL{r.level}" for r in data.itertuples()]

    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(labels)), 4.6))
    bottom = np.zeros(len(data))
    positions = np.arange(len(data))
    for name in config.OUTCOME_ORDER:
        values = data[f"pct_{name}"].to_numpy()
        ax.bar(positions, values, bottom=bottom, width=0.74,
               color=OUTCOME_COLORS[name], label=OUTCOME_LABELS[name],
               edgecolor="#fcfcfb", linewidth=1.2)   # 2px surface gap between segments
        bottom += values

    _style_axes(ax)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7, color=TEXT_SECONDARY)
    ax.set_ylabel("procent din fluxurile eligibile", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(f"Compozitia rezultatelor perturbarii — {model}",
                 color=TEXT_PRIMARY, fontsize=12, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False,
              fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def plot_per_class_heatmap(per_class: pd.DataFrame, path, model: str) -> None:
    """Rata de evaziune per clasa de atac x varianta (magnitudine -> ramp secvential)."""
    data = per_class[(per_class["model"] == model) & (per_class["level"] > 0)]
    data = data[~data["type"].isin(config.EVASION_NON_REALIZABLE_TYPES)]
    if data.empty:
        return
    table = data.pivot_table(index="attack", columns="variant", values="evasion_rate")

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL_HUE)
    fig, ax = plt.subplots(figsize=(max(9, 0.42 * table.shape[1]), 0.45 * table.shape[0] + 2.6))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=90, fontsize=7, color=TEXT_SECONDARY)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index, fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    bar.set_label("rata de evaziune (%)", color=TEXT_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    bar.outline.set_visible(False)

    ax.set_title(f"Evaziune per clasa de atac — {model}", color=TEXT_PRIMARY,
                 fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
