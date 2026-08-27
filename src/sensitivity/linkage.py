"""Legatura dintre "ce a fost perturbat" si "cat de important era".

Intrebarea centrala din O4: perturbarile care ating caracteristici importante
produc mai multa evaziune? Raspunsul cere trei surse puse cap la cap:

  results/perturbation/feature_deltas.csv  -> ce caracteristici schimba fiecare varianta
  importanta prin permutare (src.sensitivity.importance)  -> cat conteaza fiecare
  results/evasion/evasion_summary.csv      -> cata evaziune a produs fiecare varianta
"""

import numpy as np
import pandas as pd
from scipy import stats

from src import config


def load_feature_deltas() -> pd.DataFrame:
    """Incarca ce caracteristici modifica fiecare varianta, produs de O2.

    Returns:
        pd.DataFrame: coloane [variant, feature, rows_changed, pct_rows_changed, ...].

    Raises:
        FileNotFoundError: daca O2 nu a fost rulat.
    """
    path = config.PERTURBATION_DIR / "feature_deltas.csv"
    if not path.exists():
        raise FileNotFoundError(f"lipseste {path}; ruleaza mai intai `python run_perturbation.py`")
    return pd.read_csv(path)


def importance_mass(deltas: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    """Cat "capital de importanta" atinge fiecare varianta, per model.

    Masa se calculeaza in doua feluri, ambele raportate:
      - `importance_mass`: suma bruta a importantei caracteristicilor atinse;
      - `importance_mass_weighted`: aceeasi suma, dar fiecare caracteristica
        ponderata cu fractia de randuri pe care chiar s-a schimbat. O varianta
        care atinge o caracteristica importanta pe 2% dintre randuri nu e
        echivalenta cu una care o schimba peste tot.

    Parameters:
        deltas (pd.DataFrame): rezultatul lui load_feature_deltas.
        importance (pd.DataFrame): importanta prin permutare, per (model, feature).

    Returns:
        pd.DataFrame: o linie per (model, variant).
    """
    rows = []
    for model, model_importance in importance.groupby("model"):
        lookup = model_importance.set_index("feature")["detection_drop"].clip(lower=0)
        for variant, group in deltas.groupby("variant"):
            touched = group[group["feature"].isin(lookup.index)]
            weights = touched["pct_rows_changed"].to_numpy() / 100.0
            values = lookup.reindex(touched["feature"]).to_numpy()
            rows.append({
                "model": model,
                "variant": variant,
                "n_features_touched": int(len(touched)),
                "features_touched": ",".join(sorted(touched["feature"])),
                "importance_mass": float(np.nansum(values)),
                "importance_mass_weighted": float(np.nansum(values * weights)),
                "top_feature_touched": (touched.assign(imp=values)
                                        .sort_values("imp", ascending=False)["feature"].iloc[0]
                                        if len(touched) else ""),
            })
    return pd.DataFrame(rows)


def correlate_with_evasion(mass: pd.DataFrame, evasion: pd.DataFrame) -> tuple:
    """Coreleaza masa de importanta atinsa cu rata de evaziune observata.

    Se foloseste corelatia de rang (Spearman), nu Pearson: relatia nu are motiv
    sa fie liniara, esantionul e mic (numarul de variante) si ne intereseaza doar
    daca ordinea se pastreaza — variantele care ating mai multa importanta produc
    mai multa evaziune?

    Se exclud varianta-identitate (evaziune 0 prin constructie) si variantele
    nerealizabile.

    Parameters:
        mass (pd.DataFrame): rezultatul lui importance_mass.
        evasion (pd.DataFrame): evasion_summary.csv.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (tabelul imbinat, corelatiile per model).
    """
    realizable = evasion[(evasion["level"] > 0) & evasion["realizable"]]
    merged = realizable.merge(mass, on=["model", "variant"], how="inner")

    correlations = []
    for model, group in merged.groupby("model"):
        row = {"model": model, "n_variants": int(len(group))}
        for column in ("importance_mass", "importance_mass_weighted"):
            if group[column].nunique() > 1:
                rho, p_value = stats.spearmanr(group[column], group["evasion_rate"])
            else:
                rho, p_value = np.nan, np.nan
            row[f"spearman_{column}"] = round(float(rho), 4)
            row[f"pvalue_{column}"] = round(float(p_value), 6)
        correlations.append(row)

    columns = ["model", "variant", "type", "level", "evasion_rate", "n_features_touched",
               "importance_mass", "importance_mass_weighted", "top_feature_touched",
               "features_touched"]
    return merged[columns].sort_values(["model", "evasion_rate"], ascending=[True, False]), \
        pd.DataFrame(correlations)


def single_feature_attribution(importance: pd.DataFrame,
                               perturbable: list[str]) -> pd.DataFrame:
    """Cat din importanta totala sta in caracteristicile pe care O2 le poate atinge.

    Pune in perspectiva rezultatul principal: daca o singura caracteristica
    (sttl) concentreaza o parte disproportionata din importanta si e si trivial
    de modificat de atacator, atunci fragilitatea nu e o surpriza, ci o consecinta.

    Parameters:
        importance (pd.DataFrame): importanta prin permutare, per (model, feature).
        perturbable (list[str]): caracteristicile efectiv modificate de O2.

    Returns:
        pd.DataFrame: o linie per model.
    """
    rows = []
    for model, group in importance.groupby("model"):
        positive = group.assign(value=group["detection_drop"].clip(lower=0))
        total = positive["value"].sum()
        touched = positive[positive["feature"].isin(perturbable)]["value"].sum()
        top = positive.sort_values("value", ascending=False).iloc[0]
        rows.append({
            "model": model,
            "total_detection_drop": round(float(total), 4),
            "perturbable_share_pct": round(100.0 * touched / total, 2) if total else 0.0,
            "top_feature": top["feature"],
            "top_feature_drop": round(float(top["value"]), 4),
            "top_feature_share_pct": round(100.0 * top["value"] / total, 2) if total else 0.0,
        })
    return pd.DataFrame(rows)
