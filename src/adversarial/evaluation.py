"""Evaluarea bratelor O6: performanta pe trafic curat si evaziune pe cohorta comuna.

PROBLEMA DENOMINATORULUI. O3 raporteaza evaziunea peste fluxurile ELIGIBILE ale
fiecarui model — cele pe care modelul respectiv le-a semnalat ca malitioase pe
traficul curat. Doua modele diferite au multimi de fluxuri eligibile diferite,
deci "22% inainte" si "9% dupa" sunt procente din populatii diferite si nu se
pot compara direct: un model care semnaleaza mai putine fluxuri are un
denominator mai mic si poate parea mai robust fara sa fie.

SOLUTIA. Cohorta comuna: intersectia fluxurilor eligibile pentru TOATE bratele si
toate seed-urile aceleiasi familii de modele. Pe aceasta multime fixa, ratele
sunt masurate pe exact aceleasi fluxuri si sunt comparabile pereche cu pereche.
Se raporteaza si ratele brute (fiecare pe eligibilii proprii), dar cifra
principala e cea pe cohorta comuna, intotdeauna insotita de marimea n a cohortei.

CONTEXTUL DE EVALUARE. Variantele pe care se masoara evaziunea se genereaza cu
ACELASI context ca in O3 — build_context(train, test) — ca perturbarile masurate
sa fie identice, rand cu rand, cu cele din capitolul de rezultate. Aici nu exista
scurgere: modelele sunt deja antrenate si inghetate, iar contextul intervine doar
in construirea intrarilor de test. Contextul TRAIN-ONLY este cel folosit la
generarea datelor de ANTRENARE (vezi augment.py); cele doua roluri sunt distincte
si nu trebuie confundate.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

from src import config
from src.evasion import outcomes as outcome_utils
from src.perturbation import generators

# Codificarea compacta a celor trei rezultate, ca matricea per-rand sa incapa
# lejer in memorie si pe disc (45.332 fluxuri x 29 variante x 27 rulari).
OUTCOME_CODES = {config.OUTCOME_CORRECT: 0,
                 config.OUTCOME_MISCLASSIFIED_ATTACK: 1,
                 config.OUTCOME_EVADED: 2}
CODE_EVADED = OUTCOME_CODES[config.OUTCOME_EVADED]


def run_key(model_family: str, arm: str, seed: int) -> str:
    """Identificatorul textual al unei rulari (familie de model, brat, seed)."""
    return f"{model_family}|{arm}|{seed}"


def outcomes_path(model_family: str, arm: str, seed: int):
    """Fisierul in care se salveaza matricea de rezultate per rand a unei rulari."""
    return config.ADVERSARIAL_DIR / "outcomes" / f"{model_family.lower()}_{arm}_s{seed}.npz"


def clean_metrics(model, X_test: pd.DataFrame, y_test, labels: list) -> dict:
    """Metrici pe setul de test oficial CURAT, in acelasi format ca in O1.

    Returns:
        dict: metrici agregate, F1 per clasa si predictiile brute.
    """
    predictions = np.asarray(model.predict(X_test))
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    per_class_f1 = {cls: float(vals["f1-score"]) for cls, vals in report.items()
                    if cls in set(labels)}
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "per_class_f1": per_class_f1,
        "predictions": predictions,
    }


def sweep_outcomes(model, X_attack: pd.DataFrame, true_labels: np.ndarray,
                   ctx: dict, logger=None, label: str = "") -> tuple:
    """Parcurge grila O2 si codifica rezultatul fiecarui flux, la fiecare varianta.

    Se masoara pe TOATE fluxurile de atac, nu doar pe eligibilii modelului:
    restrangerea la o cohorta se face ulterior, ca aceeasi matrice sa poata fi
    citita atat pe eligibilii proprii, cat si pe cohorta comuna.

    Returns:
        tuple: (lista variantelor, matrice int8 [n_variante, n_fluxuri]).
    """
    grid = generators.variant_grid()
    matrix = np.empty((len(grid), len(X_attack)), dtype=np.int8)
    variants = []

    for row, (ptype, level) in enumerate(grid):
        variant, _ = generators.generate_variant(X_attack, ptype, level, ctx)
        predictions = np.asarray(model.predict(variant))
        coded = outcome_utils.classify_outcomes(true_labels, predictions)
        for name, code in OUTCOME_CODES.items():
            matrix[row][coded == name] = code
        variants.append({"variant": f"{ptype}_L{level}", "type": ptype, "level": level,
                         "realizable": ptype not in config.EVASION_NON_REALIZABLE_TYPES})

    if logger is not None:
        logger.info(f"  [{label}] {len(grid)} variante masurate pe {len(X_attack):,} fluxuri")
    return variants, matrix


def eligible_from_matrix(variants: list, matrix: np.ndarray) -> np.ndarray:
    """Masca de eligibilitate, citita din varianta-identitate.

    Eligibil = modelul a ridicat o alarma pe fluxul CURAT, adica a prezis orice
    altceva decat "Normal". In codificarea folosita aici asta inseamna orice cod
    diferit de CODE_EVADED pe randul identitatii.
    """
    identity_row = next(i for i, v in enumerate(variants) if v["level"] == 0)
    return matrix[identity_row] != CODE_EVADED


def common_cohort(masks: list) -> np.ndarray:
    """Intersectia mastilor de eligibilitate ale tuturor rularilor unei familii."""
    cohort = np.ones_like(masks[0], dtype=bool)
    for mask in masks:
        cohort &= mask
    return cohort


def rates_on_cohort(variants: list, matrix: np.ndarray, cohort: np.ndarray,
                    true_labels: np.ndarray) -> pd.DataFrame:
    """Rata de evaziune si compozitia rezultatelor pe o cohorta data.

    Parameters:
        variants (list): descrierea variantelor, in ordinea randurilor matricei.
        matrix (np.ndarray): rezultate codificate [n_variante, n_fluxuri].
        cohort (np.ndarray): masca de fluxuri pe care se face raportarea.
        true_labels (np.ndarray): etichetele reale, pentru defalcarea per clasa.

    Returns:
        pd.DataFrame: o linie per varianta, cu rata globala si cate o coloana de
            rata per clasa de atac.
    """
    n = int(cohort.sum())
    classes = sorted(set(true_labels[cohort])) if n else []
    rows = []
    for index, meta in enumerate(variants):
        values = matrix[index][cohort]
        entry = dict(meta)
        entry["n_cohort"] = n
        entry["n_evaded"] = int((values == CODE_EVADED).sum())
        entry["evasion_rate"] = round(100.0 * entry["n_evaded"] / n, 4) if n else np.nan
        entry["pct_correct"] = round(100.0 * (values == 0).sum() / n, 4) if n else np.nan
        entry["pct_misclassified_attack"] = round(100.0 * (values == 1).sum() / n, 4) if n else np.nan
        for cls in classes:
            cls_mask = true_labels[cohort] == cls
            support = int(cls_mask.sum())
            entry[f"evasion_{cls}"] = (
                round(100.0 * (values[cls_mask] == CODE_EVADED).sum() / support, 4)
                if support else np.nan)
            entry[f"n_{cls}"] = support
        rows.append(entry)
    return pd.DataFrame(rows)


def aggregate_over_seeds(frames: dict, value_columns: list) -> pd.DataFrame:
    """Media si amplitudinea peste seed-uri, per (familie, brat, varianta).

    Un singur seed supraestimeaza certitudinea, mai ales la retea; se raporteaza
    media, abaterea standard si extremele.

    Parameters:
        frames (dict): (familie, brat, seed) -> cadrul returnat de rates_on_cohort.
        value_columns (list[str]): coloanele pe care se agrega.

    Returns:
        pd.DataFrame: o linie per (familie, brat, varianta).
    """
    stacked = []
    for (family, arm, seed), frame in frames.items():
        tagged = frame.copy()
        tagged.insert(0, "model", family)
        tagged.insert(1, "arm", arm)
        tagged.insert(2, "seed", seed)
        stacked.append(tagged)
    combined = pd.concat(stacked, ignore_index=True)

    grouped = combined.groupby(["model", "arm", "variant", "type", "level", "realizable"],
                               as_index=False)
    aggregated = grouped.agg({col: ["mean", "std", "min", "max"] for col in value_columns})
    aggregated.columns = ["_".join(c).rstrip("_") for c in aggregated.columns]
    counts = grouped.size().rename(columns={"size": "n_seeds"})
    merged = aggregated.merge(
        counts, on=["model", "arm", "variant", "type", "level", "realizable"])
    return combined, merged


def check_identity_invariant(variants: list, matrix: np.ndarray, cohort: np.ndarray,
                             label: str) -> None:
    """Invariantul mostenit din O3: pe cohorta, identitatea nu poate produce evaziune.

    Cohorta e construita din fluxuri pe care fiecare model le-a semnalat pe
    trafic curat, iar varianta-identitate nu modifica nimic. O valoare nenula ar
    insemna ca masuratoarea nu se aliniaza cu predictiile de referinta.

    Raises:
        AssertionError: daca invariantul e incalcat.
    """
    identity_row = next(i for i, v in enumerate(variants) if v["level"] == 0)
    evaded = int((matrix[identity_row][cohort] == CODE_EVADED).sum())
    if evaded:
        raise AssertionError(
            f"[{label}] varianta-identitate produce {evaded} evaziuni pe cohorta comuna; "
            f"masuratoarea nu e aliniata cu eligibilitatea")
