"""Taxonomia celor trei rezultate ale unei perturbari si metricile derivate.

Din obiectivele lucrarii: intr-un cadru multi-clasa, o varianta perturbata poate
  (a) sa ramana clasificata corect            -> OUTCOME_CORRECT
  (b) sa devina "trafic normal"               -> OUTCOME_EVADED  (evaziune propriu-zisa)
  (c) sa fie confundata cu alt tip de atac    -> OUTCOME_MISCLASSIFIED_ATTACK

Distinctia (c) e cea care spune ceva despre fragilitatea granitelor de decizie
DINTRE clasele de atac: fluxul e in continuare semnalat ca malitios, deci
alarma exista, dar eticheta e gresita.

DENOMINATORUL CONTEAZA. Rata de evaziune se calculeaza exclusiv peste fluxurile
ELIGIBILE — cele pentru care modelul a ridicat o alarma pe traficul CURAT
(a prezis orice altceva decat "Normal"). Un flux pe care modelul nu l-a prins
niciodata nu poate "evada": daca l-am include, am numara esecuri preexistente
ale modelului drept succese ale atacatorului si am umfla artificial rezultatul.
"""

import numpy as np
import pandas as pd

from src import config


def classify_outcomes(true_labels, predictions) -> np.ndarray:
    """Incadreaza fiecare flux perturbat in una dintre cele trei categorii.

    Parameters:
        true_labels (array-like): etichetele reale (attack_cat) ale fluxurilor.
        predictions (array-like): etichetele prezise pe varianta perturbata.

    Returns:
        np.ndarray: valori din config.OUTCOME_ORDER, cate una per rand.
    """
    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(predictions)

    outcomes = np.full(len(pred_arr), config.OUTCOME_MISCLASSIFIED_ATTACK, dtype=object)
    outcomes[pred_arr == "Normal"] = config.OUTCOME_EVADED
    outcomes[(pred_arr == true_arr) & (pred_arr != "Normal")] = config.OUTCOME_CORRECT
    return outcomes


def evasion_summary(outcomes) -> dict:
    """Agregă rezultatele unui set de fluxuri intr-un rand de raport.

    Parameters:
        outcomes (array-like): valori produse de classify_outcomes.

    Returns:
        dict: numarul si proportia pentru fiecare categorie, plus rata de evaziune.
    """
    arr = np.asarray(outcomes)
    n = len(arr)
    counts = {name: int((arr == name).sum()) for name in config.OUTCOME_ORDER}
    summary = {"n_eligible": n}
    for name, count in counts.items():
        summary[f"n_{name}"] = count
        summary[f"pct_{name}"] = round(100.0 * count / n, 4) if n else 0.0
    summary["evasion_rate"] = summary[f"pct_{config.OUTCOME_EVADED}"]
    return summary


def confidence_shift(baseline_proba: np.ndarray, variant_proba: np.ndarray,
                     classes: np.ndarray, true_labels) -> dict:
    """Cuantifica deplasarea increderii modelului intre traficul curat si cel perturbat.

    Doua marimi complementare:
      - P(Normal): cat de mult creste "convingerea" ca fluxul e benign;
      - P(clasa reala): cat de mult scade suportul pentru eticheta corecta.
    Ambele sunt utile chiar si acolo unde eticheta finala nu s-a schimbat inca,
    pentru ca arata cat de aproape de granita a fost impins fluxul.

    Parameters:
        baseline_proba (np.ndarray): probabilitati pe traficul curat [n, n_classes].
        variant_proba (np.ndarray): probabilitati pe varianta perturbata [n, n_classes].
        classes (np.ndarray): ordinea claselor, comuna ambelor matrici.
        true_labels (array-like): etichetele reale, pentru P(clasa reala).

    Returns:
        dict: mediile inainte/dupa si deplasarile.
    """
    class_list = list(classes)
    normal_idx = class_list.index("Normal")
    true_idx = np.array([class_list.index(label) for label in np.asarray(true_labels)])
    rows = np.arange(len(true_idx))

    base_normal = baseline_proba[:, normal_idx]
    var_normal = variant_proba[:, normal_idx]
    base_true = baseline_proba[rows, true_idx]
    var_true = variant_proba[rows, true_idx]

    return {
        "mean_p_normal_clean": float(base_normal.mean()),
        "mean_p_normal_perturbed": float(var_normal.mean()),
        "mean_p_normal_shift": float((var_normal - base_normal).mean()),
        "mean_p_true_clean": float(base_true.mean()),
        "mean_p_true_perturbed": float(var_true.mean()),
        "mean_p_true_shift": float((var_true - base_true).mean()),
    }


def per_class_summary(true_labels, outcomes) -> pd.DataFrame:
    """Defalca rezultatele pe clasa de atac.

    Clasele cu suport mic (Worms are 44 de randuri in test) sunt marcate, ca
    rata lor sa nu fie citita ca o masuratoare stabila.

    Parameters:
        true_labels (array-like): etichetele reale.
        outcomes (array-like): valori produse de classify_outcomes.

    Returns:
        pd.DataFrame: o linie per clasa de atac.
    """
    frame = pd.DataFrame({"attack": np.asarray(true_labels), "outcome": np.asarray(outcomes)})
    rows = []
    for attack, group in frame.groupby("attack", sort=True):
        summary = evasion_summary(group["outcome"].to_numpy())
        summary["attack"] = attack
        summary["low_support"] = summary["n_eligible"] < config.EVASION_MIN_CLASS_SUPPORT
        rows.append(summary)
    columns = ["attack", "n_eligible", "evasion_rate", "low_support"]
    result = pd.DataFrame(rows)
    return result[columns + [c for c in result.columns if c not in columns]]
