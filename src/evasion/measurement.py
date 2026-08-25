"""Parcurgerea grilei de variante si masurarea rezultatelor per flux.

Incarcarea datelor de referinta (eligibilitate, predictii si probabilitati pe
traficul curat) se face din artefactele deja produse de O1/O2/Transformer, nu
prin recalculare, ca masuratoarea sa se raporteze exact la aceleasi valori pe
care le-au verificat etapele anterioare.
"""

import numpy as np
import pandas as pd

from src import config, inference
from src.evasion import outcomes as outcome_utils
from src.perturbation import generators

# nume model -> (fisier eligibilitate, coloana flag, fisier predictii, coloana predictie)
MODEL_SOURCES = {
    "RandomForest": (config.PERTURBATION_DIR / "eligible_flags.csv", "rf_eligible",
                     config.PREDICTIONS_DIR / "baseline_predictions.csv",
                     "randomforest_prediction"),
    "XGBoost": (config.PERTURBATION_DIR / "eligible_flags.csv", "xgb_eligible",
                config.PREDICTIONS_DIR / "baseline_predictions.csv",
                "xgboost_prediction"),
    "Transformer": (config.TRANSFORMER_ELIGIBLE_FLAGS_PATH, "transformer_eligible",
                    config.TRANSFORMER_PREDICTIONS_PATH, "transformer_prediction"),
}


def load_reference(model_name: str, ids: np.ndarray) -> dict:
    """Incarca eligibilitatea, predictiile si probabilitatile pe traficul curat.

    Parameters:
        model_name (str): "RandomForest" | "XGBoost" | "Transformer".
        ids (np.ndarray): identificatorii fluxurilor de atac, in ordinea de lucru.

    Returns:
        dict: "eligible" (mask bool), "clean_predictions", "clean_proba", "proba_classes".

    Raises:
        FileNotFoundError: daca lipseste un artefact de referinta.
    """
    flags_path, flag_col, preds_path, pred_col = MODEL_SOURCES[model_name]
    for path in (flags_path, preds_path):
        if not path.exists():
            raise FileNotFoundError(f"artefact de referinta lipsa pentru {model_name}: {path}")

    flags = pd.read_csv(flags_path).set_index("sample_id").reindex(ids)
    preds = pd.read_csv(preds_path).set_index("sample_id").reindex(ids)

    proba_path = config.PREDICTIONS_DIR / f"baseline_probabilities_{model_name.lower()}.csv"
    proba_frame = pd.read_csv(proba_path).set_index("sample_id").reindex(ids)
    proba_classes = np.array([c[2:-1] for c in proba_frame.columns])  # "P(Normal)" -> "Normal"

    return {
        "eligible": flags[flag_col].to_numpy(dtype=bool),
        "clean_predictions": preds[pred_col].to_numpy(),
        "clean_proba": proba_frame.to_numpy(dtype=float),
        "proba_classes": proba_classes,
    }


def measure_variant(models: dict, references: dict, X: pd.DataFrame,
                    true_labels: np.ndarray, ids: np.ndarray,
                    variant_type: str, level: int, ctx: dict) -> tuple[list, list]:
    """Masoara o singura varianta perturbata, pentru toate modelele.

    Parameters:
        models (dict): modelele incarcate prin src.inference.
        references (dict): datele de referinta per model, din load_reference.
        X (pd.DataFrame): fluxurile de atac neperturbate.
        true_labels (np.ndarray): etichetele reale.
        ids (np.ndarray): identificatorii fluxurilor.
        variant_type (str): tipul de perturbare.
        level (int): nivelul de intensitate (0 = identitate).
        ctx (dict): contextul generatoarelor.

    Returns:
        tuple[list, list]: (randuri de rezumat, cadre per-rand).
    """
    variant, _ = generators.generate_variant(X, variant_type, level, ctx)
    variant_label = f"{variant_type}_L{level}"

    summaries, per_row_frames = [], []
    for model_name, model in models.items():
        reference = references[model_name]
        mask = reference["eligible"]

        predictions = model.predict(variant)
        probabilities = model.predict_proba(variant)

        # Totul se restrange la fluxurile eligibile (alarma pe traficul curat).
        eligible_true = true_labels[mask]
        eligible_pred = np.asarray(predictions)[mask]
        row_outcomes = outcome_utils.classify_outcomes(eligible_true, eligible_pred)

        summary = outcome_utils.evasion_summary(row_outcomes)
        summary.update(outcome_utils.confidence_shift(
            reference["clean_proba"][mask], probabilities[mask],
            reference["proba_classes"], eligible_true))
        summary.update({
            "model": model_name, "variant": variant_label,
            "type": variant_type, "level": level,
            "realizable": variant_type not in config.EVASION_NON_REALIZABLE_TYPES,
        })
        summaries.append(summary)

        class_frame = outcome_utils.per_class_summary(eligible_true, row_outcomes)
        class_frame.insert(0, "model", model_name)
        class_frame.insert(1, "variant", variant_label)
        class_frame.insert(2, "type", variant_type)
        class_frame.insert(3, "level", level)
        summaries[-1]["_per_class"] = class_frame

        if config.EVASION_PERSIST_PER_ROW:
            normal_idx = list(reference["proba_classes"]).index("Normal")
            per_row_frames.append(pd.DataFrame({
                "sample_id": ids[mask],
                "true_label": eligible_true,
                "model": model_name,
                "variant": variant_label,
                "type": variant_type,
                "level": level,
                "prediction": eligible_pred,
                "outcome": row_outcomes,
                "p_normal": probabilities[mask][:, normal_idx].round(6),
            }))

    return summaries, per_row_frames


def run_sweep(models: dict, references: dict, X: pd.DataFrame, true_labels: np.ndarray,
              ids: np.ndarray, ctx: dict, logger) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Parcurge intreaga grila de variante si aduna rezultatele.

    Returns:
        tuple: (rezumat per varianta/model, defalcare per clasa, cadre per-rand).
    """
    grid = generators.variant_grid()
    logger.info(f"\nMasurare: {len(grid)} variante x {len(models)} modele "
                f"x {len(X):,} fluxuri de atac")

    summary_rows, class_frames, per_row_frames = [], [], []
    for variant_type, level in grid:
        summaries, rows = measure_variant(models, references, X, true_labels, ids,
                                          variant_type, level, ctx)
        for summary in summaries:
            class_frames.append(summary.pop("_per_class"))
            summary_rows.append(summary)
        per_row_frames.extend(rows)

        rates = "  ".join(f"{s['model'][:3]}={s['evasion_rate']:5.2f}%" for s in summaries)
        logger.info(f"  {variant_type}_L{level:<2} {rates}")

    return pd.DataFrame(summary_rows), pd.concat(class_frames, ignore_index=True), per_row_frames
