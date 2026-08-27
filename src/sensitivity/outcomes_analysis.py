"""Interpretarea celor trei rezultate ale perturbarii.

O3 a numarat rezultatele; O4 le interpreteaza. Doua intrebari:

  1. Cand un flux NU evadeaza dar isi schimba eticheta (rezultatul (c)), spre ce
     clasa aluneca? Destinatiile arata unde sunt subtiri granitele de decizie
     dintre clasele de atac — informatia pe care obiectivele lucrarii o cer
     explicit de la distinctia (c).

  2. Fluxurile care evadeaza erau deja aproape de granita pe traficul curat, sau
     perturbarea le-a mutat de la o decizie ferma? Daca modelul era deja nesigur
     pe ele, fragilitatea e mai putin ingrijoratoare decat daca erau clasificate
     cu incredere mare.
"""

import numpy as np
import pandas as pd

from src import config


def outcome_destinations(per_row: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Spre ce clase aluneca fluxurile confundate cu alt atac (rezultatul (c)).

    Parameters:
        per_row (pd.DataFrame): rezultatele per flux pentru un model.
        variant (str): varianta analizata.

    Returns:
        pd.DataFrame: matrice clasa reala x clasa prezisa, in procente pe rand.
    """
    subset = per_row[(per_row["variant"] == variant)
                     & (per_row["outcome"] == config.OUTCOME_MISCLASSIFIED_ATTACK)]
    if subset.empty:
        return pd.DataFrame()
    table = pd.crosstab(subset["true_label"], subset["prediction"])
    return (100.0 * table.div(table.sum(axis=1), axis=0)).round(2)


def boundary_proximity(per_row: pd.DataFrame, baseline_proba: pd.DataFrame,
                       classes: np.ndarray, variant: str) -> dict:
    """Compara increderea de pe traficul CURAT a fluxurilor care evadeaza si a celor care nu.

    Daca fluxurile care cedeaza aveau deja o incredere scazuta, perturbarea doar
    a impins peste granita cazuri deja marginale. Daca aveau incredere ridicata,
    modelul chiar se baza pe caracteristica perturbata.

    Parameters:
        per_row (pd.DataFrame): rezultatele per flux pentru un model.
        baseline_proba (pd.DataFrame): probabilitatile pe test curat, indexate pe sample_id.
        classes (np.ndarray): ordinea claselor din matricea de probabilitati.
        variant (str): varianta analizata.

    Returns:
        dict: increderea medie pe traficul curat, separat pentru evadate si ne-evadate.
    """
    subset = per_row[per_row["variant"] == variant]
    if subset.empty:
        return {}

    proba = baseline_proba.reindex(subset["sample_id"])
    class_list = list(classes)
    true_index = np.array([class_list.index(label) for label in subset["true_label"]])
    p_true_clean = proba.to_numpy(dtype=float)[np.arange(len(subset)), true_index]

    evaded = (subset["outcome"] == config.OUTCOME_EVADED).to_numpy()
    if not evaded.any() or evaded.all():
        return {"variant": variant, "n_evaded": int(evaded.sum()),
                "note": "grup gol — comparatie imposibila"}

    return {
        "variant": variant,
        "n_evaded": int(evaded.sum()),
        "n_not_evaded": int((~evaded).sum()),
        "mean_p_true_clean_evaded": round(float(p_true_clean[evaded].mean()), 4),
        "mean_p_true_clean_not_evaded": round(float(p_true_clean[~evaded].mean()), 4),
        "confidence_gap": round(float(p_true_clean[~evaded].mean()
                                      - p_true_clean[evaded].mean()), 4),
    }


def divergence_analysis(importance_by_class: pd.DataFrame,
                        target_class: str = config.SENSITIVITY_DIVERGENCE_CLASS) -> pd.DataFrame:
    """Explica divergenta dintre modele pe o clasa anume.

    O3 a semnalat ca pe `Generic` arborii rezista (1,5-5,5% evaziune) in timp ce
    Transformer-ul se prabuseste (97,3%), desi toate trei o clasifica aproape
    perfect pe traficul curat. Daca importanta restransa la acea clasa arata ca
    Transformer-ul se sprijina pe TTL iar arborii nu, divergenta e explicata.

    Parameters:
        importance_by_class (pd.DataFrame): importanta prin permutare calculata
            doar pe fluxurile clasei tinta.
        target_class (str): clasa analizata.

    Returns:
        pd.DataFrame: primele caracteristici per model, pentru clasa tinta.
    """
    rows = []
    for model, group in importance_by_class.groupby("model"):
        ranked = group.sort_values("detection_drop", ascending=False)
        total = ranked["detection_drop"].clip(lower=0).sum()
        for rank, (_, entry) in enumerate(ranked.head(5).iterrows(), start=1):
            rows.append({
                "attack_class": target_class,
                "model": model,
                "rank": rank,
                "feature": entry["feature"],
                "detection_drop": round(float(entry["detection_drop"]), 4),
                "share_pct": round(100.0 * max(entry["detection_drop"], 0) / total, 2)
                if total else 0.0,
            })
    return pd.DataFrame(rows)
