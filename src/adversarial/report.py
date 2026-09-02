"""Tabelele finale O6 si verdictul fata de criteriile pre-inregistrate.

CE INSEAMNA "CEL MAI RAU CAZ". Un atacator alege varianta care ii convine, nu una
la intamplare. Cifra de robustete raportata aici este prin urmare MAXIMUL ratei de
evaziune peste variantele realizabile, nu media lor: o aparare care reduce media
dar lasa o varianta la 95% nu a aparat nimic. ttl_both ramane exclus, fiind
marcat nerealizabil inca din O2.

CE INSEAMNA "IMBUNATATIRE". Diferenta fata de C1, nu fata de O1. C1 adauga exact
acelasi numar de randuri de atac, dar neperturbate, deci orice castig care apare
si la C1 se datoreaza volumului de date, nu perturbarii. Se raporteaza si
diferenta fata de O1, ca sa se vada ambele componente.
"""

import numpy as np
import pandas as pd

from src import config

WORST_CASE_LABEL = "worst_case_realizable"


def _realizable(frame: pd.DataFrame) -> pd.DataFrame:
    """Doar variantele realizabile si non-identitate."""
    return frame[frame["realizable"] & (frame["level"] > 0)]


def worst_case_table(per_seed: pd.DataFrame, column: str = "evasion_rate") -> pd.DataFrame:
    """Cel mai rau caz per (model, brat, seed), apoi media si amplitudinea pe seed-uri.

    Parameters:
        per_seed (pd.DataFrame): ratele per (model, brat, seed, varianta), pe cohorta.
        column (str): coloana masurata (globala sau per clasa).

    Returns:
        pd.DataFrame: o linie per (model, brat), cu media, abaterea si extremele
            maximului peste variantele realizabile, plus varianta care il atinge.
    """
    realizable = _realizable(per_seed)
    if column not in realizable.columns:
        return pd.DataFrame(columns=["model", "arm", "metric", "n_seeds", "worst_case_mean",
                                     "worst_case_std", "worst_case_min", "worst_case_max",
                                     "worst_variants"])
    realizable = realizable.dropna(subset=[column])
    per_run = (realizable.loc[realizable.groupby(["model", "arm", "seed"])[column].idxmax()]
               [["model", "arm", "seed", "variant", column]]
               .rename(columns={column: "worst_case", "variant": "worst_variant"}))

    rows = []
    for (model, arm), group in per_run.groupby(["model", "arm"]):
        rows.append({
            "model": model, "arm": arm, "metric": column,
            "n_seeds": int(len(group)),
            "worst_case_mean": round(float(group["worst_case"].mean()), 4),
            "worst_case_std": round(float(group["worst_case"].std(ddof=1)), 4)
                              if len(group) > 1 else 0.0,
            "worst_case_min": round(float(group["worst_case"].min()), 4),
            "worst_case_max": round(float(group["worst_case"].max()), 4),
            "worst_variants": ",".join(sorted(set(group["worst_variant"]))),
        })
    return pd.DataFrame(rows).sort_values(["metric", "model", "arm"]).reset_index(drop=True)


def clean_summary(clean_rows: list) -> pd.DataFrame:
    """Media si amplitudinea metricilor pe trafic curat, per (model, brat)."""
    frame = pd.DataFrame(clean_rows)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    grouped = frame.groupby(["model", "arm"], as_index=False)
    aggregated = grouped.agg({m: ["mean", "std", "min", "max"] for m in metrics})
    aggregated.columns = ["_".join(c).rstrip("_") for c in aggregated.columns]
    aggregated["n_seeds"] = grouped.size()["size"].to_numpy()
    return aggregated.round(4)


def per_class_f1_table(per_class: dict) -> pd.DataFrame:
    """Tabelul F1 per clasa, mediat peste seed-uri, per (model, brat)."""
    rows = []
    for key, values in per_class.items():
        family, arm, seed = key.split("|")
        for cls, score in values.items():
            rows.append({"model": family, "arm": arm, "seed": int(seed),
                         "attack": cls, "f1": score})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["model", "arm", "attack",
                                     "f1_mean", "f1_std", "f1_min", "f1_max"])
    return (frame.groupby(["model", "arm", "attack"], as_index=False)
            .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                 f1_min=("f1", "min"), f1_max=("f1", "max")).round(4))


def _lookup(table: pd.DataFrame, model: str, arm: str, metric: str, column: str):
    """O singura valoare din tabelul de cel-mai-rau-caz, sau NaN daca lipseste."""
    match = table[(table["model"] == model) & (table["arm"] == arm)
                  & (table["metric"] == metric)]
    return float(match.iloc[0][column]) if not match.empty else float("nan")


def verdict(worst_global: pd.DataFrame, worst_class: pd.DataFrame,
            clean: pd.DataFrame, per_class: pd.DataFrame,
            cohort_sizes: dict, model: str = "Transformer") -> dict:
    """Confrunta rezultatele cu criteriile pre-inregistrate si da verdictul.

    Criteriile au fost fixate in manifest INAINTE de rulare. Un esec e la randul
    lui un rezultat, nu o eroare de executie: se raporteaza ca atare.

    Returns:
        dict: fiecare criteriu cu valoarea masurata, pragul si verdictul.
    """
    metric_class = f"evasion_{config.O6_CRITERION_CLASS}"
    o1, c1, o6 = config.O6_ARM_BASELINE, config.O6_ARM_CONTROL, config.O6_ARM_DEFENDED

    c1_class = _lookup(worst_class, model, c1, metric_class, "worst_case_mean")
    o6_class = _lookup(worst_class, model, o6, metric_class, "worst_case_mean")
    o1_class = _lookup(worst_class, model, o1, metric_class, "worst_case_mean")
    improvement = c1_class - o6_class

    def clean_value(arm, column="macro_f1_mean"):
        match = clean[(clean["model"] == model) & (clean["arm"] == arm)]
        return float(match.iloc[0][column]) if not match.empty else float("nan")

    macro_o1, macro_c1, macro_o6 = clean_value(o1), clean_value(c1), clean_value(o6)

    safety = per_class[(per_class["model"] == model)]
    baseline_f1 = safety[safety["arm"] == o1].set_index("attack")["f1_mean"]
    defended_f1 = safety[safety["arm"] == o6].set_index("attack")["f1_mean"]
    drops = (baseline_f1 - defended_f1).dropna().sort_values(ascending=False)
    worst_class_drop = float(drops.iloc[0]) if len(drops) else float("nan")

    return {
        "model": model,
        "cohort_n": cohort_sizes.get(model, {}).get("n"),
        "primary_robustness": {
            "criterion": f"evaziunea {config.O6_CRITERION_CLASS} (cel mai rau caz realizabil, "
                         f"cohorta comuna) scade cu >= "
                         f"{config.O6_CRITERION_MIN_EVASION_DROP_PP} pp fata de C1",
            "o1_pct": round(o1_class, 3), "c1_pct": round(c1_class, 3),
            "o6_pct": round(o6_class, 3),
            "improvement_over_c1_pp": round(improvement, 3),
            "improvement_over_o1_pp": round(o1_class - o6_class, 3),
            "threshold_pp": config.O6_CRITERION_MIN_EVASION_DROP_PP,
            "passed": bool(improvement >= config.O6_CRITERION_MIN_EVASION_DROP_PP),
        },
        "primary_clean_cost": {
            "criterion": f"macro-F1 pe test curat scade cu <= "
                         f"{config.O6_CRITERION_MAX_MACRO_F1_DROP} fata de C1 si O1",
            "o1_macro_f1": round(macro_o1, 4), "c1_macro_f1": round(macro_c1, 4),
            "o6_macro_f1": round(macro_o6, 4),
            "drop_vs_c1": round(macro_c1 - macro_o6, 4),
            "drop_vs_o1": round(macro_o1 - macro_o6, 4),
            "threshold": config.O6_CRITERION_MAX_MACRO_F1_DROP,
            "passed": bool(max(macro_c1 - macro_o6, macro_o1 - macro_o6)
                           <= config.O6_CRITERION_MAX_MACRO_F1_DROP),
        },
        "safety": {
            "criterion": f"nicio clasa nu pierde mai mult de "
                         f"{config.O6_CRITERION_MAX_PER_CLASS_F1_DROP} din F1 fata de O1",
            "worst_class": str(drops.index[0]) if len(drops) else None,
            "worst_drop": round(worst_class_drop, 4),
            "all_drops": {str(k): round(float(v), 4) for k, v in drops.items()},
            "threshold": config.O6_CRITERION_MAX_PER_CLASS_F1_DROP,
            "passed": bool(worst_class_drop <= config.O6_CRITERION_MAX_PER_CLASS_F1_DROP),
        },
        "stretch_target": {
            "criterion": f"evaziunea {config.O6_CRITERION_CLASS} in cel mai rau caz "
                         f"<= {config.O6_STRETCH_EVASION_ABS}% absolut",
            "value_pct": round(o6_class, 3),
            "threshold_pct": config.O6_STRETCH_EVASION_ABS,
            "met": bool(o6_class <= config.O6_STRETCH_EVASION_ABS),
            "note": "tinta aspirationala, nu criteriu de trecere",
        },
        "global_worst_case": {
            arm: round(_lookup(worst_global, model, arm, "evasion_rate", "worst_case_mean"), 3)
            for arm in (o1, c1, o6)
        },
    }


def holdout_comparison(per_seed: pd.DataFrame, model: str = "Transformer") -> pd.DataFrame:
    """Generalizarea la un tip de perturbare NEVAZUT la antrenare.

    Bratul o6_loo a fost antrenat fara nicio varianta din familia TTL, iar aici se
    masoara tocmai pe ea. Robustetea pe un tip nevazut e o dovada mult mai
    puternica decat robustetea pe unul antrenat: arata ca modelul a invatat
    invarianta, nu transformarile concrete.
    """
    held_out = tuple(config.O6_HOLDOUT_TYPES)
    subset = _realizable(per_seed)
    subset = subset[(subset["model"] == model) & (subset["type"].isin(held_out))]
    if subset.empty:
        return pd.DataFrame()
    per_run = subset.loc[subset.groupby(["arm", "seed"])["evasion_rate"].idxmax()]
    return (per_run.groupby("arm", as_index=False)
            .agg(worst_case_mean=("evasion_rate", "mean"),
                 worst_case_min=("evasion_rate", "min"),
                 worst_case_max=("evasion_rate", "max"),
                 n_seeds=("seed", "nunique"))
            .round(3))


def save_figures(per_seed: pd.DataFrame, clean: pd.DataFrame, directory) -> list:
    """Figurile O6: curbele de evaziune per brat si compromisul cost/robustete."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    arm_style = {config.O6_ARM_BASELINE: ("tab:red", "O1 — original"),
                 config.O6_ARM_CONTROL: ("tab:orange", "C1 — control (duplicate)"),
                 config.O6_ARM_DEFENDED: ("tab:green", "O6 — augmentare perturbata"),
                 config.O6_ARM_HOLDOUT: ("tab:blue", "O6-LOO — fara familia TTL")}

    # ---- Curbele per familie de perturbari, un panou per model ----
    realizable = _realizable(per_seed)
    families = sorted(realizable["type"].unique())
    for model in sorted(realizable["model"].unique()):
        subset = realizable[realizable["model"] == model]
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
        for ax, family in zip(axes.ravel(), families):
            block = subset[subset["type"] == family]
            for arm, (color, label) in arm_style.items():
                arm_block = block[block["arm"] == arm]
                if arm_block.empty:
                    continue
                curve = arm_block.groupby("level")["evasion_rate"].agg(["mean", "min", "max"])
                ax.plot(curve.index, curve["mean"], marker="o", color=color, label=label)
                ax.fill_between(curve.index, curve["min"], curve["max"], color=color, alpha=0.15)
            ax.set_title(family, fontsize=10)
            ax.set_xlabel("nivel")
            ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("rata de evaziune (%)")
        axes[1, 0].set_ylabel("rata de evaziune (%)")
        for ax in axes.ravel()[len(families):]:
            ax.axis("off")
        handles, labels_ = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="lower right", fontsize=10)
        fig.suptitle(f"O6 — evaziune pe cohorta comuna, {model} "
                     f"(medie peste seed-uri, banda = min–max)", fontsize=13)
        fig.tight_layout()
        path = directory / f"o6_evasion_curves_{model.lower()}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

    # ---- Compromisul: cost pe trafic curat vs cel mai rau caz de evaziune ----
    worst = worst_case_table(per_seed)
    merged = worst.merge(clean, on=["model", "arm"], how="inner")
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"RandomForest": "o", "XGBoost": "s", "Transformer": "^"}
    for _, row in merged.iterrows():
        color = arm_style.get(row["arm"], ("gray", row["arm"]))[0]
        ax.scatter(row["macro_f1_mean"], row["worst_case_mean"],
                   color=color, marker=markers.get(row["model"], "o"), s=110,
                   edgecolor="black", linewidth=0.5)
        ax.annotate(f"{row['model'][:3]}/{row['arm']}",
                    (row["macro_f1_mean"], row["worst_case_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("macro-F1 pe test curat")
    ax.set_ylabel("cel mai rau caz de evaziune, cohorta comuna (%)")
    ax.set_title("O6 — costul pe trafic curat fata de castigul de robustete")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = directory / "o6_cost_vs_robustness.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)
    return written
