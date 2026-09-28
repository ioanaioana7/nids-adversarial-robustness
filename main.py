"""
Baseline NIDS multi-clasa pe UNSW-NB15 (Random Forest + XGBoost)
================================================================
Pasii 1-3 din lucrare: incarcare -> preprocesare -> antrenare -> evaluare.

Orchestreaza modulele din src/ (config, data_loader, eda, preprocessing,
models, evaluation, experiment, shap_analysis). Metodologia (split, encodare,
hiperparametri) e neschimbata fata de baseline-ul original — vezi
src/config.py pentru toti parametrii.

Cum se ruleaza:
    pip install pandas scikit-learn matplotlib joblib xgboost shap
    python main.py

Rezultate: models/*.joblib, results/ (figures, metrics, feature_importance,
eda, predictions, training.log, experiment_metadata.json, dataset_summary.json,
label_mapping.json, feature_names.csv).
"""

import joblib
from sklearn.preprocessing import LabelEncoder

from src import config, data_loader, eda, evaluation, experiment, models, shap_analysis, utils


def main():
    utils.set_global_seed(config.RANDOM_SEED)
    logger = utils.setup_logging()

    train, test = data_loader.load_data()
    eda.run_eda(train, logger)

    feature_exclude = [config.ID_COL, config.TARGET, config.LABEL_COL]
    X_train, y_train = train.drop(columns=feature_exclude), train[config.TARGET]
    X_test, y_test = test.drop(columns=feature_exclude), test[config.TARGET]
    numeric_cols = eda.numeric_feature_columns(train)
    labels = sorted(y_train.unique())

    # ---- Model 1: Random Forest (baseline principal) ----
    logger.info("\nAntrenare Random Forest...")
    rf_pipeline = models.build_rf_pipeline(numeric_cols)
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)

    rf_metrics = evaluation.evaluate("RandomForest", y_test, rf_pred, labels)
    evaluation.log_evaluation(rf_metrics, logger)
    evaluation.save_metrics(rf_metrics)
    evaluation.save_confusion_matrix(rf_metrics)

    top_rf = evaluation.save_feature_importance(
        "RandomForest", rf_pipeline.named_steps["prep"], rf_pipeline.named_steps["clf"].feature_importances_)
    logger.info(f"\n{'=' * 60}\nTOP 15 FEATURES (Random Forest)\n{'=' * 60}")
    for feat, val in top_rf.items():
        logger.info(f"  {feat:<35} {val:.4f}")

    rf_model_path = config.MODELS_DIR / "rf_baseline.joblib"
    joblib.dump(rf_pipeline, rf_model_path)
    logger.info(f"[salvat] {rf_model_path}")
    evaluation.validate_saved_model("RandomForest", rf_model_path, X_test, rf_pred, logger)

    rf_proba = rf_pipeline.predict_proba(X_test)
    evaluation.export_probabilities("RandomForest", test[config.ID_COL].values, rf_proba,
                                     rf_pipeline.named_steps["clf"].classes_)
    shap_analysis.compute_and_save_shap(rf_pipeline, X_test, logger)

    # ---- Model 2: XGBoost (comparatie) ----
    # XGBoost cere etichete numerice -> codificam tinta cu LabelEncoder
    logger.info("\nAntrenare XGBoost...")
    le = LabelEncoder().fit(y_train)
    xgb_pipeline = models.build_xgb_pipeline(numeric_cols)
    xgb_pipeline.fit(X_train, le.transform(y_train))
    xgb_pred = le.inverse_transform(xgb_pipeline.predict(X_test))

    xgb_metrics = evaluation.evaluate("XGBoost", y_test, xgb_pred, labels)
    evaluation.log_evaluation(xgb_metrics, logger)
    evaluation.save_metrics(xgb_metrics)
    evaluation.save_confusion_matrix(xgb_metrics)

    xgb_model_path = config.MODELS_DIR / "xgb_baseline.joblib"
    joblib.dump({"pipeline": xgb_pipeline, "label_encoder": le}, xgb_model_path)
    logger.info(f"[salvat] {xgb_model_path}")
    evaluation.validate_saved_model("XGBoost", xgb_model_path, X_test, xgb_pred, logger)

    xgb_proba = xgb_pipeline.predict_proba(X_test)
    evaluation.export_probabilities("XGBoost", test[config.ID_COL].values, xgb_proba, le.classes_)

    # ---- Metadata de experiment, pentru reproducibilitate si pentru O2 ----
    experiment.save_label_mapping(le)0
    experiment.save_feature_names(rf_pipeline.named_steps["prep"])
    experiment.save_dataset_summary(train, test, numeric_cols)
    experiment.save_experiment_metadata(
        numeric_cols, labels,
        rf_params=rf_pipeline.named_steps["clf"].get_params(),
        xgb_params=xgb_pipeline.named_steps["clf"].get_params(),
    )
    evaluation.export_predictions(
        test[config.ID_COL].values, y_test.values,
        {"RandomForest": rf_pred, "XGBoost": xgb_pred},
    )

    logger.info("\nGata. Modelele sunt in models/, toate rezultatele in results/.")


if __name__ == "__main__":
    main()
