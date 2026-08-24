"""Punctul UNIC de intrare pentru predictie cu modelele inghetate.

DE CE EXISTA ACEST MODUL
    RandomForest-ul inghetat are 44 de randuri pe muchie de cutit in setul de
    atac: 43 la egalitate exacta intre primele doua clase si unul cu diferenta
    de un singur ULP — sample_id=49676, unde Exploits=0.392499999999999905 si
    Reconnaissance=0.392500000000000016 (ambele inseamna 78,5 voturi din 200).

    Cu n_jobs=-1, voturile arborilor se insumeaza in ordine variabila intre fire
    de executie, iar rotunjirea in virgula mobila decide care clasa iese cu un
    ULP mai sus. Rezultatul: acelasi model, aceleasi date, predictii diferite
    intre rulari.

    Pericolul concret pentru O3: daca verificarea (varianta-identitate) si
    masurarea (variantele perturbate) ar folosi cai de predictie diferite,
    un flux ca 49676 s-ar putea rezolva "Reconnaissance" la verificare si
    "Exploits" la masurare, si ar fi numarat ca schimbare de clasa provocata de
    perturbare — cand de fapt e doar planificarea firelor de executie.

    De aceea TOATE consumatoarele (verificarea din O2, masurarea din O3) trebuie
    sa treaca prin acest modul, nu sa incarce modelele direct cu joblib.load.

INTERFATA UNIFORMA
    load_frozen_models() returneaza obiecte care expun toate acelasi API —
    .predict(X) -> etichete string, .predict_proba(X) -> probabilitati,
    .classes_ -> ordinea claselor. O3 nu are nevoie de ramificatii per model.
"""

import numpy as np
import pandas as pd

from src import config

TREE_MODEL_NAMES = ("RandomForest", "XGBoost")
ALL_MODEL_NAMES = ("RandomForest", "XGBoost", "Transformer")


class XGBoostLabelAdapter:
    """Uniformizeaza XGBoost cu celelalte modele.

    XGBoost a fost antrenat pe etichete numerice, deci pipeline-ul salvat
    returneaza coduri; adaptorul le decodeaza inapoi in etichete string, ca
    .predict() sa se comporte identic cu RandomForest si Transformer.
    """

    def __init__(self, pipeline, label_encoder):
        self.pipeline = pipeline
        self.label_encoder = label_encoder

    @property
    def classes_(self) -> np.ndarray:
        """Etichetele de clasa, in aceeasi ordine ca la celelalte modele."""
        return self.label_encoder.classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Eticheta de clasa prezisa (string)."""
        return self.label_encoder.inverse_transform(self.pipeline.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilitati per clasa, in ordinea din classes_."""
        return self.pipeline.predict_proba(X)


def force_deterministic_inference(pipeline,
                                  n_jobs: int = config.DETERMINISTIC_INFERENCE_N_JOBS) -> bool:
    """Forteaza predictia pe un singur fir pentru estimatorul dintr-un pipeline.

    NU reantreneaza si NU modifica ponderile invatate: schimba doar numarul de
    fire folosite la inferenta, deci rezultatele stocate raman valabile
    (verificat: n_jobs=1 reproduce exact baseline_predictions.csv).

    Parameters:
        pipeline: pipeline sklearn cu un pas final "clf".
        n_jobs (int): numarul de fire pentru inferenta.

    Returns:
        bool: True daca s-a putut seta n_jobs pe estimatorul final.
    """
    estimator = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else pipeline
    if estimator is not None and hasattr(estimator, "n_jobs"):
        estimator.n_jobs = n_jobs
        return True
    return False


def load_frozen_models(include_transformer: bool = False,
                       deterministic: bool = config.DETERMINISTIC_INFERENCE) -> dict:
    """Incarca modelele inghetate, cu predictie determinista.

    Nu se reantreneaza nimic; modelele vin de pe disc exact asa cum au fost
    salvate de O1 si de etapa Transformer.

    Parameters:
        include_transformer (bool): include si FT-Transformer-ul. Importul torch
            se face lazy, ca O2 sa nu depinda de PyTorch daca nu e nevoie.
        deterministic (bool): forteaza inferenta pe un singur fir la arbori.

    Returns:
        dict: nume model -> obiect cu .predict / .predict_proba / .classes_.

    Raises:
        FileNotFoundError: daca modelul Transformer e cerut dar nu a fost antrenat.
    """
    import joblib

    rf = joblib.load(config.RF_MODEL_PATH)
    xgb_bundle = joblib.load(config.XGB_MODEL_PATH)
    xgb = XGBoostLabelAdapter(xgb_bundle["pipeline"], xgb_bundle["label_encoder"])

    if deterministic:
        force_deterministic_inference(rf)
        force_deterministic_inference(xgb.pipeline)

    models = {"RandomForest": rf, "XGBoost": xgb}

    if include_transformer:
        if not config.TRANSFORMER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"modelul Transformer lipseste ({config.TRANSFORMER_MODEL_PATH}); "
                f"ruleaza mai intai `python train_transformer.py`")
        from src.transformer.predictor import TransformerNIDS
        models["Transformer"] = TransformerNIDS.load(config.TRANSFORMER_MODEL_PATH)

    return models


def predict(models: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Predictii (etichete string) cu fiecare model, prin calea determinista.

    Parameters:
        models (dict): rezultatul lui load_frozen_models().
        X (pd.DataFrame): features brute UNSW-NB15 (inclusiv variante perturbate din O2).

    Returns:
        dict[str, np.ndarray]: nume model -> etichete prezise.
    """
    return {name: model.predict(X) for name, model in models.items()}


def predict_proba(models: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Probabilitati per clasa cu fiecare model, prin calea determinista.

    Parameters:
        models (dict): rezultatul lui load_frozen_models().
        X (pd.DataFrame): features brute UNSW-NB15.

    Returns:
        dict[str, np.ndarray]: nume model -> matrice [n_samples, n_classes].
    """
    return {name: model.predict_proba(X) for name, model in models.items()}
