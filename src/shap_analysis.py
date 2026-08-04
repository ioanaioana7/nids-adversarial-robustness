"""Analiza SHAP (optionala) pentru Random Forest — pregateste terenul pentru O4.

Daca librăria shap nu este instalata sau calculul esueaza, se salveaza un
fisier placeholder si se logheaza motivul, fara sa opreasca pipeline-ul
principal (asa cum cere checklist-ul pre-O2: pasul e "optional but recommended").
"""

import numpy as np
import pandas as pd

from src import config


def compute_and_save_shap(pipeline, X_test: pd.DataFrame, logger,
                           sample_size: int = config.SHAP_SAMPLE_SIZE) -> None:
    """Calculeaza valorile SHAP pentru clasificatorul RF, pe un esantion din test, si le salveaza.

    Parameters:
        pipeline (sklearn.pipeline.Pipeline): pipeline-ul RF antrenat (rare -> prep -> clf).
        X_test (pd.DataFrame): setul de test brut (neprocesat).
        logger (logging.Logger): logger pentru mesaje.
        sample_size (int): numarul de mostre de test folosite pentru SHAP (calcul costisitor
            pe tot setul de test; un esantion e suficient pentru analiza de sensibilitate).
    """
    placeholder_path = config.FEATURE_IMPORTANCE_DIR / "shap_mean_abs.csv"
    try:
        import shap
    except ImportError:
        logger.info("[SHAP] pachetul 'shap' nu este instalat — se omite (placeholder salvat).")
        pd.DataFrame(columns=["feature", "mean_abs_shap"]).to_csv(placeholder_path, index=False)
        return

    try:
        import matplotlib.pyplot as plt

        sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=config.RANDOM_SEED)
        transformed = pipeline[:-1].transform(sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()  # TreeExplainer nu accepta matrici sparse
        feature_names = pipeline.named_steps["prep"].get_feature_names_out()

        explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
        # additivity check esueaza cu class_weight="balanced" (ponderare pe clasa
        # in interiorul arborilor), fara sa indice o problema reala de corectitudine
        shap_values = explainer.shap_values(transformed, check_additivity=False)

        classes = pipeline.named_steps["clf"].classes_
        if isinstance(shap_values, list):
            # lista de matrici (n_samples, n_features), una per clasa
            per_class = shap_values
        elif shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) -> lista per clasa, ceruta de summary_plot
            per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        else:
            per_class = [shap_values]
        mean_abs = np.mean([np.abs(sv) for sv in per_class], axis=(0, 1))

        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}) \
            .sort_values("mean_abs_shap", ascending=False) \
            .to_csv(placeholder_path, index=False)

        plt.figure()
        shap.summary_plot(per_class, transformed, feature_names=feature_names,
                           class_names=list(classes), plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"[salvat] shap_mean_abs.csv, shap_summary.png (esantion={len(sample)})")
    except Exception as exc:  # calculul SHAP nu trebuie sa opreasca antrenarea baseline-ului
        logger.info(f"[SHAP] calculul a esuat ({exc!r}) — se salveaza placeholder.")
        pd.DataFrame(columns=["feature", "mean_abs_shap"]).to_csv(placeholder_path, index=False)
