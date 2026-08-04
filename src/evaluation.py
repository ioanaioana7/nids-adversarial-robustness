"""Evaluare model: metrici, matrice de confuzie, importanta caracteristicilor,
export de predictii/probabilitati si validarea modelelor salvate pe disc."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src import config


def evaluate(name: str, y_test, y_pred, labels: list) -> dict:
    """Calculeaza metricile de evaluare pentru un model, fara a le afisa.

    Parameters:
        name (str): numele modelului (ex. "RandomForest").
        y_test (array-like): etichetele reale.
        y_pred (array-like): etichetele prezise.
        labels (list): lista ordonata a claselor.

    Returns:
        dict: "name", "classification_report" (dict), "macro_f1", "weighted_f1",
            "accuracy", "balanced_accuracy", "confusion_matrix" (normalizata pe randuri),
            "labels".
    """
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
    return {
        "name": name,
        "classification_report": report,
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "confusion_matrix": cm,
        "labels": labels,
    }


def log_evaluation(metrics: dict, logger) -> None:
    """Afiseaza (consola + training.log) raportul de clasificare si scorurile agregate."""
    name = metrics["name"]
    logger.info(f"\n{'=' * 60}\nEVALUARE — {name}\n{'=' * 60}")

    report_no_accuracy = {k: v for k, v in metrics["classification_report"].items() if k != "accuracy"}
    report_df = pd.DataFrame(report_no_accuracy).T
    logger.info("\n" + report_df.to_string(float_format=lambda x: f"{x:.4f}"))

    logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Macro F1:          {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1:       {metrics['weighted_f1']:.4f}")


def save_metrics(metrics: dict) -> None:
    """Salveaza metricile agregate si tabelul per-clasa ca CSV in config.METRICS_DIR.

    Fisiere: results/metrics/<model>_metrics.csv (rezumat) si
    results/metrics/<model>_per_class.csv (precizie/recall/F1/support per clasa,
    tabel util direct pentru tabelele din lucrare).
    """
    name = metrics["name"]
    summary = pd.DataFrame([{
        "model": name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }])
    summary.to_csv(config.METRICS_DIR / f"{name.lower()}_metrics.csv", index=False)

    report = metrics["classification_report"]
    per_class = pd.DataFrame([
        {"attack": cls, "precision": vals["precision"], "recall": vals["recall"],
         "f1": vals["f1-score"], "support": vals["support"]}
        for cls, vals in report.items()
        if cls not in ("accuracy", "macro avg", "weighted avg")
    ])
    per_class.to_csv(config.METRICS_DIR / f"{name.lower()}_per_class.csv", index=False)


def save_confusion_matrix(metrics: dict) -> None:
    """Salveaza matricea de confuzie normalizata ca CSV (results/metrics) si PNG (results/figures)."""
    name = metrics["name"]
    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
    cm_df.to_csv(config.METRICS_DIR / f"confusion_matrix_{name.lower()}.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(metrics["confusion_matrix"], display_labels=metrics["labels"]).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", values_format=".2f")
    ax.set_title(f"Matrice de confuzie — {name} (normalizata pe randuri)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"confusion_matrix_{name.lower()}.png", dpi=150)
    plt.close()


def save_feature_importance(name: str, prep, importances, top_n: int = 15) -> pd.Series:
    """Salveaza importanta caracteristicilor (model-based) ca CSV, ordonata descrescator.

    Parameters:
        name (str): numele modelului.
        prep: ColumnTransformer-ul fitted (pentru get_feature_names_out()).
        importances (array-like): feature_importances_ ale clasificatorului.
        top_n (int): cate intrari se returneaza pentru afisare in consola.

    Returns:
        pd.Series: top_n cele mai importante caracteristici (pentru logging).
    """
    names = prep.get_feature_names_out()
    series = pd.Series(importances, index=names).sort_values(ascending=False)
    series.rename_axis("feature").reset_index(name="importance").to_csv(
        config.FEATURE_IMPORTANCE_DIR / f"{name.lower()}.csv", index=False)
    return series.head(top_n)


def export_predictions(ids, y_true, predictions_by_model: dict) -> None:
    """Exporta predictiile pe setul de test curat: sample_id, eticheta reala, predictie per model.

    Devine referinta baseline pentru masurarea evaziunii in O3.

    Parameters:
        ids (array-like): identificatorii originali (coloana 'id') ai randurilor de test.
        y_true (array-like): etichetele reale (attack_cat).
        predictions_by_model (dict[str, array-like]): predictii, cheie = nume model.
    """
    df = pd.DataFrame({"sample_id": np.asarray(ids), "true_label": np.asarray(y_true)})
    for model_name, preds in predictions_by_model.items():
        df[f"{model_name.lower()}_prediction"] = np.asarray(preds)
    df.to_csv(config.PREDICTIONS_DIR / "baseline_predictions.csv", index=False)


def export_probabilities(name: str, ids, proba: np.ndarray, classes) -> None:
    """Exporta probabilitatile prezise per clasa pentru un model.

    Va permite compararea nivelului de incredere al modelului inainte/dupa perturbare (O3/O4).

    Parameters:
        name (str): numele modelului.
        ids (array-like): identificatorii originali (coloana 'id') ai randurilor de test.
        proba (np.ndarray): matrice (n_samples, n_classes) de probabilitati.
        classes (array-like): numele claselor, in ordinea coloanelor din proba.
    """
    df = pd.DataFrame(proba, columns=[f"P({c})" for c in classes])
    df.insert(0, "sample_id", np.asarray(ids))
    df.to_csv(config.PREDICTIONS_DIR / f"baseline_probabilities_{name.lower()}.csv", index=False)


def validate_saved_model(name: str, path, X_sample: pd.DataFrame, in_memory_predictions,
                          logger, n: int = 100) -> None:
    """Reincarca modelul salvat de pe disc si verifica identitatea predictiilor pe primele n mostre.

    Esueaza explicit daca serializarea (joblib) a schimbat comportamentul modelului.

    Parameters:
        name (str): numele modelului, pentru logging.
        path (Path): calea catre fisierul .joblib salvat.
        X_sample (pd.DataFrame): setul de test brut (se folosesc primele n randuri).
        in_memory_predictions (array-like): predictiile modelului antrenat, pe X_sample.
        logger (logging.Logger): logger pentru mesaje.
        n (int): numarul de mostre verificate.

    Raises:
        AssertionError: daca predictiile reincarcate difera de cele originale.
    """
    loaded = joblib.load(path)
    is_wrapped = isinstance(loaded, dict)
    pipeline = loaded["pipeline"] if is_wrapped else loaded

    reloaded_predictions = pipeline.predict(X_sample.head(n))
    if is_wrapped:
        reloaded_predictions = loaded["label_encoder"].inverse_transform(reloaded_predictions)

    expected = np.asarray(in_memory_predictions[:n])
    if not np.array_equal(np.asarray(reloaded_predictions), expected):
        raise AssertionError(f"[{name}] predictiile modelului reincarcat difera de cele originale!")

    logger.info(f"[validare] {name}: predictiile modelului salvat coincid cu cele originale ({n} mostre)")
