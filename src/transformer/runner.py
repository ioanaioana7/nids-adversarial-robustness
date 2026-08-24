"""Orchestrarea antrenarii FT-Transformer-ului si scrierea artefactelor.

Reutilizeaza functiile de evaluare din O1 (src.evaluation), astfel incat
metricile, matricea de confuzie si probabilitatile sa aiba EXACT acelasi format
ca pentru RF/XGBoost. Predictiile si flagurile de eligibilitate se scriu in
fisiere PARALELE per model, ca sa nu atinga artefactele detinute de O1/O2.
"""

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config, data_loader, evaluation, utils
from src.transformer import training
from src.transformer.model import FTTransformer
from src.transformer.predictor import TransformerNIDS, resolve_device
from src.transformer.preprocessing import TabularPreprocessor

MODEL_NAME = "Transformer"


def _git_commit() -> str | None:
    """Commit-ul curent, daca proiectul e un repo git; altfel None."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _assert_label_mapping_matches(encoder: LabelEncoder, logger) -> None:
    """Confirma ca maparea eticheta->cod e identica cu cea folosita de XGBoost.

    LabelEncoder sorteaza clasele, deci ar trebui sa coincida oricum; verificam
    explicit fata de results/label_mapping.json ca sa nu existe divergente tacute
    intre modele.

    Raises:
        AssertionError: daca maparile difera.
    """
    path = config.RESULTS_DIR / "label_mapping.json"
    if not path.exists():
        logger.info("  [etichete] label_mapping.json lipseste — se sare verificarea")
        return
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = {cls: int(i) for i, cls in enumerate(encoder.classes_)}
    if actual != expected:
        raise AssertionError(f"maparea etichetelor difera de XGBoost: {actual} vs {expected}")
    logger.info(f"  [etichete] mapare identica cu XGBoost ({len(actual)} clase)")


def validate_saved_model(predictor: TransformerNIDS, X_test: pd.DataFrame,
                         in_memory_predictions: np.ndarray, logger,
                         n: int = config.TRANSFORMER_RELOAD_CHECK_ROWS) -> dict:
    """Reincarca modelul salvat si verifica predictii identice (oglindeste O1).

    Raises:
        AssertionError: daca serializarea a schimbat comportamentul modelului.
    """
    reloaded = TransformerNIDS.load(config.TRANSFORMER_MODEL_PATH)
    reloaded_predictions = reloaded.predict(X_test.head(n))
    expected = np.asarray(in_memory_predictions[:n])
    mismatches = int((np.asarray(reloaded_predictions) != expected).sum())
    if mismatches:
        raise AssertionError(
            f"[{MODEL_NAME}] predictiile modelului reincarcat difera: {mismatches}/{n}")
    logger.info(f"  [validare] {MODEL_NAME}: predictiile modelului salvat coincid "
                f"cu cele originale ({n} mostre)")
    return {"rows": int(n), "mismatches": 0}


def verify_o2_identity(predictor: TransformerNIDS, test: pd.DataFrame,
                       clean_predictions: np.ndarray, logger) -> dict:
    """Verifica varianta-identitate din O2 fata de predictiile pe test curat.

    Foloseste API-ul public al lui O2 (generate_variant) fara sa il modifice.
    Daca predictiile difera, harnessul de perturbare introduce drift pentru acest
    model si orice masuratoare ulterioara ar fi un artefact.

    Raises:
        AssertionError: la orice nepotrivire.
    """
    from src.perturbation import generators, schema as perturbation_schema

    train_raw, test_raw = data_loader.load_data()
    feature_cols = perturbation_schema.feature_columns(test_raw)
    attack_rows = test_raw[test_raw[config.TARGET] != "Normal"].reset_index(drop=True)
    X_attack = attack_rows[feature_cols]

    ctx = generators.build_context(train_raw, test_raw)
    identity, _ = generators.generate_variant(X_attack, "identity", 0, ctx)

    # Predictiile pe test curat, restranse la aceleasi randuri de atac.
    attack_mask = (test[config.TARGET] != "Normal").to_numpy()
    expected = np.asarray(clean_predictions)[attack_mask]
    actual = predictor.predict(identity)

    mismatches = int((actual != expected).sum())
    if mismatches:
        raise AssertionError(
            f"[{MODEL_NAME}] varianta-identitate O2 NU reproduce predictiile pe test curat: "
            f"{mismatches} nepotriviri din {len(expected)} randuri")
    logger.info(f"  [identitate O2] {MODEL_NAME}: {len(expected)} randuri, 0 nepotriviri")
    return {"rows": int(len(expected)), "mismatches": 0}


def verify_schema_compatibility(predictor: TransformerNIDS, logger, n_rows: int = 64) -> dict:
    """Confirma ca predictorul accepta un frame perturbat de O2 fara eroare."""
    from src.perturbation import generators, schema as perturbation_schema

    train_raw, test_raw = data_loader.load_data()
    feature_cols = perturbation_schema.feature_columns(test_raw)
    attack_rows = test_raw[test_raw[config.TARGET] != "Normal"].reset_index(drop=True)

    probe_type, probe_level = "combined_mimic", len(config.COMBINED_LEVELS)
    ctx = generators.build_context(train_raw, test_raw)
    probe, _ = generators.generate_variant(
        attack_rows[feature_cols].head(n_rows), probe_type, probe_level, ctx)

    predictions = predictor.predict(probe)
    probabilities = predictor.predict_proba(probe)
    logger.info(f"  [schema] {len(predictions)} predictii pe {probe_type}_L{probe_level} OK "
                f"(probabilitati {probabilities.shape})")
    return {"variant": f"{probe_type}_L{probe_level}", "rows": int(len(predictions))}


def export_predictions(ids: np.ndarray, y_true: np.ndarray, predictions: np.ndarray) -> None:
    """Scrie predictiile pe test curat, in fisier propriu (schema ca la O1)."""
    pd.DataFrame({
        "sample_id": ids,
        "true_label": y_true,
        f"{MODEL_NAME.lower()}_prediction": predictions,
    }).to_csv(config.TRANSFORMER_PREDICTIONS_PATH, index=False)


def export_eligible_flags(test: pd.DataFrame, predictions: np.ndarray) -> int:
    """Scrie flagurile de eligibilitate pe randurile de atac (conventia din O2).

    Eligibil = modelul a ridicat o alarma pe fluxul CURAT (a prezis orice altceva
    decat "Normal"). O3 calculeaza evaziunea doar peste aceste fluxuri, altfel
    atacurile pe care modelul nu le-a prins niciodata ar fi numarate ca evaziuni.

    Returns:
        int: numarul de fluxuri eligibile.
    """
    attack_mask = (test[config.TARGET] != "Normal").to_numpy()
    attack_predictions = np.asarray(predictions)[attack_mask]
    flags = pd.DataFrame({
        "sample_id": test.loc[attack_mask, config.ID_COL].to_numpy(),
        "true_label": test.loc[attack_mask, config.TARGET].to_numpy(),
        f"{MODEL_NAME.lower()}_eligible": attack_predictions != "Normal",
    })
    flags.to_csv(config.TRANSFORMER_ELIGIBLE_FLAGS_PATH, index=False)
    return int(flags[f"{MODEL_NAME.lower()}_eligible"].sum())


def save_metadata(model: FTTransformer, split_info: dict, training_summary: dict,
                  history: list, metrics: dict, checks: dict) -> None:
    """Salveaza metadata completa a experimentului ca JSON."""
    utils.save_json({
        "model": MODEL_NAME,
        "architecture": "FT-Transformer (Gorishniy et al., NeurIPS 2021)",
        "dataset": "UNSW-NB15",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.RANDOM_SEED,
        "git_commit": _git_commit(),
        "device": split_info["device"],
        "protocol": {
            "train_split": "oficial UNSW_NB15_training-set.csv",
            "test_split": "oficial UNSW_NB15_testing-set.csv",
            "validation": f"{config.TRANSFORMER_VAL_FRACTION:.0%} stratificat din TRAIN, "
                          f"niciodata din test",
            "cross_validation": False,
            "preprocessing_fit_on": "sub-setul de train (fara validare), "
                                    "ca early stopping-ul sa ramana nepartinitor",
        },
        "hyperparameters": {
            **model.hyperparameters(),
            "lr": config.TRANSFORMER_LR,
            "weight_decay": config.TRANSFORMER_WEIGHT_DECAY,
            "batch_size": config.TRANSFORMER_BATCH_SIZE,
            "max_epochs": config.TRANSFORMER_MAX_EPOCHS,
            "early_stopping_patience": config.TRANSFORMER_EARLY_STOPPING_PATIENCE,
            "grad_clip_norm": config.TRANSFORMER_GRAD_CLIP_NORM,
            "class_weighting": "balanced (invers frecventei, ca la arbori)",
        },
        "split_sizes": {k: v for k, v in split_info.items() if k != "device"},
        "training": training_summary,
        "history": history,
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        },
        "verification": checks,
        "library_versions": {**utils.library_versions(), "torch": torch.__version__},
    }, config.TRANSFORMER_METADATA_PATH)


def run(logger, max_epochs: int = config.TRANSFORMER_MAX_EPOCHS,
        resume: bool = True, eval_only: bool = False) -> dict:
    """Antreneaza, evalueaza si verifica FT-Transformer-ul.

    Parameters:
        logger (logging.Logger): logger.
        max_epochs (int): plafonul de epoci (suprascrie config-ul, util la teste).
        resume (bool): continua dintr-un checkpoint existent, daca exista.
        eval_only (bool): sare peste antrenare si foloseste modelul deja salvat.
            Util daca antrenarea a reusit dar un pas de integrare a esuat —
            nu se pierde nimic si nu se reantreneaza.

    Returns:
        dict: rezumatul rularii.
    """
    training.seed_everything(config.RANDOM_SEED)
    utils.ensure_output_dirs()
    device = resolve_device()

    logger.info("=" * 70)
    logger.info("FT-TRANSFORMER — AL TREILEA CLASIFICATOR")
    logger.info("=" * 70)
    if device.type == "cuda":
        logger.info(f"Dispozitiv: {device} ({torch.cuda.get_device_name(0)}, "
                    f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        logger.info(f"Dispozitiv: {device}")
    logger.info(f"Algoritmi deterministi: {config.TRANSFORMER_DETERMINISTIC}")

    train_df, test_df = data_loader.load_data()
    feature_exclude = [config.ID_COL, config.TARGET, config.LABEL_COL]
    X_train_full = train_df.drop(columns=feature_exclude)
    y_train_full = train_df[config.TARGET]
    X_test = test_df.drop(columns=feature_exclude)
    y_test = test_df[config.TARGET]

    encoder = LabelEncoder().fit(y_train_full)
    _assert_label_mapping_matches(encoder, logger)
    classes = encoder.classes_
    labels = sorted(y_train_full.unique())

    # Validarea se decupeaza STRATIFICAT din TRAIN, niciodata din test.
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config.TRANSFORMER_VAL_FRACTION,
        stratify=y_train_full, random_state=config.RANDOM_SEED)
    logger.info(f"Split: train={len(X_sub)}, validare={len(X_val)} "
                f"({config.TRANSFORMER_VAL_FRACTION:.0%} din train), test={len(X_test)}")

    # Preprocesarea se ajusteaza doar pe sub-setul de train, ca statisticile
    # setului de validare sa nu se scurga in decizia de early stopping.
    preprocessor = TabularPreprocessor().fit(X_sub)
    logger.info(f"Preprocesare: {preprocessor.n_numeric} numerice standardizate, "
                f"{preprocessor.n_categorical} categorice "
                f"(cardinalitati {preprocessor.cardinalities_})")

    num_sub, cat_sub = preprocessor.transform(X_sub)
    num_val, cat_val = preprocessor.transform(X_val)
    y_sub_enc = encoder.transform(y_sub)
    y_val_enc = encoder.transform(y_val)

    train_loader = training.make_loader(num_sub, cat_sub, y_sub_enc,
                                        config.TRANSFORMER_BATCH_SIZE, shuffle=True)
    val_loader = training.make_loader(num_val, cat_val, y_val_enc,
                                      config.TRANSFORMER_EVAL_BATCH_SIZE, shuffle=False)

    model = FTTransformer(n_numeric=preprocessor.n_numeric,
                          cardinalities=preprocessor.cardinalities_,
                          n_classes=len(classes))
    logger.info(f"Model: {model.hyperparameters()['n_parameters']:,} parametri, "
                f"{config.TRANSFORMER_N_BLOCKS} blocuri x {config.TRANSFORMER_N_HEADS} capete, "
                f"d_token={config.TRANSFORMER_D_TOKEN}")

    weights = training.class_weights(y_sub_enc, len(classes))
    logger.info("Ponderi de clasa (balanced): "
                + ", ".join(f"{c}={w:.1f}" for c, w in zip(classes, weights.numpy())))

    if eval_only:
        logger.info("\n[eval-only] se sare peste antrenare; se foloseste modelul salvat")
        predictor = TransformerNIDS.load(config.TRANSFORMER_MODEL_PATH, device=device)
        model, preprocessor, classes = predictor.model, predictor.preprocessor, predictor.classes_
        training_summary, history = {"eval_only": True}, []
    else:
        logger.info("\nAntrenare:")
        training_summary, history = training.train(
            model, train_loader, val_loader, weights, device, logger,
            max_epochs=max_epochs, resume=resume)
        predictor = TransformerNIDS(model, preprocessor, classes, device=device)

    # ---- Evaluare pe setul de test oficial ----
    predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test)

    metrics = evaluation.evaluate(MODEL_NAME, y_test, predictions, labels)
    evaluation.log_evaluation(metrics, logger)
    evaluation.save_metrics(metrics)
    evaluation.save_confusion_matrix(metrics)

    predictor.save(config.TRANSFORMER_MODEL_PATH)
    logger.info(f"\n[salvat] {config.TRANSFORMER_MODEL_PATH}")

    evaluation.export_probabilities(MODEL_NAME, test_df[config.ID_COL].values,
                                     probabilities, classes)
    export_predictions(test_df[config.ID_COL].to_numpy(), y_test.to_numpy(), predictions)
    n_eligible = export_eligible_flags(test_df, predictions)
    logger.info(f"[salvat] {config.TRANSFORMER_PREDICTIONS_PATH.name}, "
                f"{config.TRANSFORMER_ELIGIBLE_FLAGS_PATH.name} "
                f"({n_eligible} fluxuri eligibile)")

    # ---- Verificari ----
    logger.info("\nVerificari:")
    checks = {
        "reload_self_check": validate_saved_model(predictor, X_test, predictions, logger),
        "o2_identity": verify_o2_identity(predictor, test_df, predictions, logger),
        "schema_compatibility": verify_schema_compatibility(predictor, logger),
    }

    split_info = {"device": str(device), "train": len(X_sub),
                  "validation": len(X_val), "test": len(X_test)}
    save_metadata(model, split_info, training_summary, history, metrics, checks)

    logger.info("\n" + "=" * 70)
    logger.info("COMPARATIE (test oficial, aceleasi features/split la toate modelele)")
    logger.info("=" * 70)
    logger.info(f"  {'model':<16}{'macro F1':>10}{'weighted F1':>14}")
    for name, macro, weighted in [("RandomForest", 0.5010, 0.7416), ("XGBoost", 0.5098, 0.7816),
                                   (MODEL_NAME, metrics["macro_f1"], metrics["weighted_f1"])]:
        logger.info(f"  {name:<16}{macro:>10.4f}{weighted:>14.4f}")
    logger.info("=" * 70)

    return {
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "eligible": n_eligible,
        "training": training_summary,
        "checks": checks,
    }
