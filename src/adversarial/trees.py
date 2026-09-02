"""Reantrenarea ansamblurilor de arbori pe seturile augmentate (brate O6 / C1 / O1).

Pipeline-urile sunt IDENTICE cu cele din O1 (aceeasi grupare a categoriilor rare,
acelasi encoder, aceiasi hiperparametri). Se schimba exact doua lucruri:
  - setul de antrenare (original / original+duplicate / original+perturbate);
  - seed-ul, ca sa se poata estima variabilitatea.
Ponderile de clasa sunt cele inghetate din src.adversarial.weights, nu
recalculate pe numaratorile augmentate.

Modelele se salveaza in models/adversarial/ cu nume care contin bratul si
seed-ul. Artefactele O1 (models/rf_baseline.joblib, xgb_baseline.joblib) nu sunt
atinse niciodata.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src import config
from src.preprocessing import RareCategoryGrouper, build_preprocessor

TREE_MODELS = ("RandomForest", "XGBoost")

# Padurea antrenata pe setul augmentat ocupa ~1,1 GB neomprimata (baseline-ul O1,
# pe 175.341 de randuri, are 633 MB). Cu 9 paduri de salvat, nivelul 1 de zlib e
# compromisul potrivit: reduce de cateva ori dimensiunea la o fractiune din timpul
# nivelului 3, care ar adauga ore la o rulare deja lunga.
DUMP_COMPRESS = 1


def model_path(model_name: str, arm: str, seed: int):
    """Calea fisierului salvat pentru un (model, brat, seed)."""
    prefix = "rf" if model_name == "RandomForest" else "xgb"
    return config.ADVERSARIAL_MODELS_DIR / f"{prefix}_{arm}_s{seed}.joblib"


def build_rf(numeric_cols, class_weight: dict, seed: int) -> Pipeline:
    """Pipeline-ul Random Forest din O1, cu ponderi de clasa explicite.

    Diferenta fata de src.models.build_rf_pipeline: class_weight primeste
    dictionarul inghetat in loc de sirul "balanced", ca ponderile sa nu se
    recalculeze pe numaratorile augmentate.
    """
    return Pipeline([
        ("rare", RareCategoryGrouper(columns=config.CATEGORICAL_COLUMNS,
                                     min_freq=config.MIN_CATEGORY_FREQUENCY)),
        ("prep", build_preprocessor(numeric_cols)),
        ("clf", RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            class_weight=class_weight,
            n_jobs=config.RF_N_JOBS,
            random_state=seed)),
    ])


def build_xgb(numeric_cols, seed: int) -> Pipeline:
    """Pipeline-ul XGBoost din O1, neschimbat in afara de seed.

    XGBoost nu primeste ponderi de clasa nici in O1, deci obiectivul e deja
    identic intre brate si nu exista nimic de inghetat.
    """
    return Pipeline([
        ("rare", RareCategoryGrouper(columns=config.CATEGORICAL_COLUMNS,
                                     min_freq=config.MIN_CATEGORY_FREQUENCY)),
        ("prep", build_preprocessor(numeric_cols)),
        ("clf", XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            subsample=config.XGB_SUBSAMPLE,
            tree_method=config.XGB_TREE_METHOD,
            n_jobs=config.XGB_N_JOBS,
            random_state=seed)),
    ])


def kept_categories(pipeline: Pipeline) -> dict:
    """Categoriile pastrate de RareCategoryGrouper, pentru comparatie intre brate.

    Augmentarea schimba frecventele categoriilor, deci teoretic ar putea schimba
    si multimea celor pastrate — ceea ce ar fi un al doilea canal prin care
    bratele difera. Se extrage explicit ca sa poata fi verificata egalitatea.
    """
    grouper = pipeline.named_steps["rare"]
    return {col: sorted(map(str, values)) for col, values in grouper.keep_.items()}


def train_tree_arm(model_name: str, arm: str, seed: int, X_train, y_train,
                   numeric_cols, class_weight: dict, logger) -> dict:
    """Antreneaza si salveaza un singur (model, brat, seed).

    Returns:
        dict: calea modelului, timpul de antrenare si categoriile pastrate.
    """
    import time

    path = model_path(model_name, arm, seed)
    started = time.time()

    if model_name == "RandomForest":
        pipeline = build_rf(numeric_cols, class_weight, seed)
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, path, compress=DUMP_COMPRESS)
    else:
        encoder = LabelEncoder().fit(y_train)
        pipeline = build_xgb(numeric_cols, seed)
        pipeline.fit(X_train, encoder.transform(y_train))
        joblib.dump({"pipeline": pipeline, "label_encoder": encoder},
                    path, compress=DUMP_COMPRESS)

    elapsed = time.time() - started
    logger.info(f"  [{arm}/s{seed}] {model_name:<14} antrenat pe {len(X_train):,} randuri "
                f"in {elapsed / 60:.1f} min -> {path.name}")
    return {"path": str(path), "rows": int(len(X_train)),
            "seconds": round(elapsed, 1), "kept_categories": kept_categories(pipeline)}


def load_tree_model(model_name: str, arm: str, seed: int):
    """Reincarca un model salvat, cu acelasi adaptor de etichete ca src.inference.

    Predictia e fortata pe un singur fir din acelasi motiv ca in O1/O3: cu
    n_jobs=-1 ordinea de insumare a voturilor variaza intre rulari si randurile
    aflate pe muchie de cutit pot oscila.
    """
    from src.inference import XGBoostLabelAdapter, force_deterministic_inference

    loaded = joblib.load(model_path(model_name, arm, seed))
    if model_name == "RandomForest":
        force_deterministic_inference(loaded)
        return loaded
    adapter = XGBoostLabelAdapter(loaded["pipeline"], loaded["label_encoder"])
    force_deterministic_inference(adapter.pipeline)
    return adapter
