"""Importanta caracteristicilor prin permutare, comparabila intre cele trei modele.

DE CE PERMUTARE SI NU IMPORTANTELE NATIVE
    Random Forest are importanta prin impuritate si SHAP; XGBoost are "gain";
    Transformer-ul are ponderi de atentie. Sunt trei marimi diferite, pe scale
    diferite, care nu se pot compara intre ele. Cum intrebarea din O4 este
    "corelati evaziunea cu importanta caracteristicilor afectate" — deci o
    comparatie INTRE modele — avem nevoie de o singura definitie, aplicata
    identic tuturor. Permutarea e model-agnostica: amesteca o coloana si masoara
    cat pierde modelul. Se aplica pe modelele inghetate, fara reantrenare.

METRICA PRINCIPALA: SCADEREA RATEI DE DETECTIE
    Rata de detectie = procentul fluxurilor de atac inca semnalate ca malitioase
    (prezise diferit de "Normal"). E deliberat exact marimea complementara celei
    masurate de O3: evaziunea inseamna ca un flux de atac devine "Normal".
    Astfel, "cat de mult se bazeaza modelul pe caracteristica X" si "cata
    evaziune produce perturbarea lui X" sunt exprimate in aceleasi unitati si
    pot fi corelate direct, nu prin analogie.

    Se raporteaza si scaderea de macro-F1, ca masura generala de referinta.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import config


def detection_rate(predictions, is_attack: np.ndarray) -> float:
    """Procentul fluxurilor de atac inca semnalate ca malitioase.

    Parameters:
        predictions (array-like): etichetele prezise.
        is_attack (np.ndarray): masca fluxurilor de atac din esantion.

    Returns:
        float: rata de detectie in procente.
    """
    attack_predictions = np.asarray(predictions)[is_attack]
    if not len(attack_predictions):
        return 0.0
    return 100.0 * float((attack_predictions != "Normal").mean())


def _score(predictions, y_true: np.ndarray, is_attack: np.ndarray) -> tuple[float, float]:
    """Cele doua metrici de referinta pentru un set de predictii."""
    return (detection_rate(predictions, is_attack),
            100.0 * f1_score(y_true, predictions, average="macro", zero_division=0))


def permutation_importance(model, X: pd.DataFrame, y_true: np.ndarray,
                           columns: list[str] | None = None,
                           n_repeats: int = config.SENSITIVITY_N_REPEATS,
                           seed: int = config.RANDOM_SEED,
                           logger=None, model_name: str = "",
                           donor: pd.DataFrame | None = None) -> pd.DataFrame:
    """Importanta prin permutare pentru un model inghetat.

    Pentru fiecare coloana: se inlocuiesc valorile ei (rupand legatura cu
    eticheta, dar pastrand distributia marginala), se reprezice si se masoara cat
    s-a pierdut fata de scorul de referinta. Repetat de n_repeats ori cu seed-uri
    diferite, se raporteaza media si abaterea standard.

    DONOR — necesar cand analiza e restransa la o singura clasa.
        Permutarea clasica amesteca valorile DIN X. Daca X e restrans la o clasa
        in care coloana e cvasi-constanta, amestecarea nu schimba nimic si
        importanta iese ~0 chiar daca modelul depinde masiv de ea. In clasa
        Generic, de exemplu, sttl e 254 pentru 98,8% dintre randuri si dttl e 0
        pentru 97,1% — permutarea interna ar fi practic o operatie nula.

        Cu `donor` setat, valorile de inlocuire se trag din distributia acelui
        cadru (tipic tot setul de test), nu din X. Intrebarea devine "daca fluxul
        asta ar avea TTL-ul unui flux oarecare, l-ar mai recunoaste modelul?" —
        exact paralela cu ce face perturbarea din O2.

    Parameters:
        model: obiect cu .predict(DataFrame) -> etichete (din src.inference).
        X (pd.DataFrame): esantionul de evaluare, cu schema bruta UNSW-NB15.
        y_true (np.ndarray): etichetele reale ale esantionului.
        columns (list[str] | None): coloanele de evaluat; implicit toate.
        n_repeats (int): numarul de permutari per coloana.
        seed (int): seed-ul de baza.
        logger: logger optional pentru progres.
        model_name (str): eticheta modelului, pentru raport.
        donor (pd.DataFrame | None): sursa valorilor de inlocuire; implicit X insusi.

    Returns:
        pd.DataFrame: o linie per caracteristica, cu scaderile medii si abaterile.
    """
    columns = list(columns or X.columns)
    is_attack = y_true != "Normal"

    base_detection, base_macro_f1 = _score(model.predict(X), y_true, is_attack)
    if logger:
        source = "donor extern" if donor is not None else "permutare interna"
        logger.info(f"  {model_name}: referinta detectie={base_detection:.2f}%  "
                    f"macroF1={base_macro_f1:.2f}%  ({len(X):,} randuri, "
                    f"{len(columns)} caracteristici x {n_repeats} permutari, {source})")

    rows = []
    for index, column in enumerate(columns):
        detection_drops, f1_drops = [], []
        for repeat in range(n_repeats):
            rng = np.random.default_rng(seed + 1000 * index + repeat)
            shuffled = X.copy()
            if donor is None:
                shuffled[column] = X[column].to_numpy()[rng.permutation(len(X))]
            else:
                pool = donor[column].to_numpy()
                shuffled[column] = pool[rng.integers(0, len(pool), size=len(X))]
            detection, macro_f1 = _score(model.predict(shuffled), y_true, is_attack)
            detection_drops.append(base_detection - detection)
            f1_drops.append(base_macro_f1 - macro_f1)

        rows.append({
            "model": model_name,
            "feature": column,
            "detection_drop": float(np.mean(detection_drops)),
            "detection_drop_std": float(np.std(detection_drops)),
            "macro_f1_drop": float(np.mean(f1_drops)),
            "macro_f1_drop_std": float(np.std(f1_drops)),
        })

    result = pd.DataFrame(rows).sort_values("detection_drop", ascending=False)
    result["detection_drop_share"] = (
        result["detection_drop"].clip(lower=0)
        / max(result["detection_drop"].clip(lower=0).sum(), 1e-12))
    return result.reset_index(drop=True)


def stratified_sample(test: pd.DataFrame, size: int = config.SENSITIVITY_SAMPLE_SIZE,
                      seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Esantion stratificat pe attack_cat din setul de test.

    Stratificarea pastreaza proportiile claselor, inclusiv ale celor rare, ca
    scaderea de macro-F1 sa ramana interpretabila.

    Parameters:
        test (pd.DataFrame): setul de test complet.
        size (int): numarul aproximativ de randuri dorite.
        seed (int): seed pentru reproducibilitate.

    Returns:
        pd.DataFrame: esantionul, cu indexul resetat.
    """
    # Se selecteaza INDICI per clasa, nu grupuri: groupby().apply() consuma
    # coloana de grupare si ar sterge attack_cat din rezultat.
    fraction = min(1.0, size / len(test))
    selected = []
    for _, group in test.groupby(config.TARGET, sort=True):
        n = max(1, int(round(len(group) * fraction)))
        selected.append(group.sample(n, random_state=seed).index)
    return test.loc[np.concatenate(selected)].reset_index(drop=True)
