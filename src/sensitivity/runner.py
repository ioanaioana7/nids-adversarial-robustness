"""Orchestrarea O4: importanta, corelatii, analiza rezultatelor, artefacte."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config, data_loader, inference, utils
from src.sensitivity import figures, importance as importance_utils, linkage, outcomes_analysis

# Varianta pe care se face analiza detaliata: cea mai puternica perturbare TTL
# realizabila, adica exact cazul care produce rezultatul principal din O3.
FOCUS_VARIANT = "ttl_mimic_L3"


def _perturbable_features() -> list[str]:
    """Caracteristicile pe care O2 le poate modifica efectiv."""
    return sorted(set(config.SOURCE_CONTROLLED + config.CONNECTION_RATE_FEATURES
                      + config.DERIVED_FEATURES + config.CAUSALLY_FORCED_BY_DUR))


def _load_per_row(model_name: str) -> pd.DataFrame | None:
    """Rezultatele per flux produse de O3, daca exista."""
    path = config.EVASION_PER_ROW_DIR / f"outcomes_{model_name.lower()}.csv.gz"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_baseline_proba(model_name: str) -> tuple:
    """Probabilitatile pe traficul curat, indexate pe sample_id."""
    path = config.PREDICTIONS_DIR / f"baseline_probabilities_{model_name.lower()}.csv"
    frame = pd.read_csv(path).set_index("sample_id")
    classes = np.array([c[2:-1] for c in frame.columns])
    return frame, classes


def run(logger) -> dict:
    """Ruleaza analiza de sensibilitate completa.

    Returns:
        dict: rezumatul rularii.
    """
    utils.set_global_seed(config.RANDOM_SEED)
    utils.ensure_output_dirs()

    logger.info("=" * 70)
    logger.info("O4 — ANALIZA DE SENSIBILITATE")
    logger.info("=" * 70)

    train, test = data_loader.load_data()
    models = inference.load_frozen_models(include_transformer=True)
    logger.info(f"Modele (predictie determinista): {list(models)}")

    feature_exclude = [config.ID_COL, config.TARGET, config.LABEL_COL]
    sample = importance_utils.stratified_sample(test)
    X_sample = sample.drop(columns=feature_exclude)
    y_sample = sample[config.TARGET].to_numpy()
    logger.info(f"Esantion stratificat: {len(sample):,} randuri "
                f"({100 * (y_sample != 'Normal').mean():.1f}% atacuri)")

    # ---- 1. Importanta prin permutare, comparabila intre modele ----
    logger.info("\n1. Importanta prin permutare (model-agnostica):")
    importance = pd.concat(
        [importance_utils.permutation_importance(model, X_sample, y_sample,
                                                 logger=logger, model_name=name)
         for name, model in models.items()],
        ignore_index=True)
    importance.to_csv(config.SENSITIVITY_DIR / "permutation_importance.csv", index=False)

    for name in models:
        top = (importance[importance["model"] == name]
               .nlargest(5, "detection_drop")[["feature", "detection_drop"]])
        listed = ", ".join(f"{r.feature}={r.detection_drop:.2f}" for r in top.itertuples())
        logger.info(f"  {name:<14} top5: {listed}")

    # ---- 2. Cat din importanta sta in ce poate atinge atacatorul ----
    attribution = linkage.single_feature_attribution(importance, _perturbable_features())
    attribution.to_csv(config.SENSITIVITY_DIR / "importance_attribution.csv", index=False)
    logger.info("\n2. Concentrarea importantei:")
    for _, row in attribution.iterrows():
        logger.info(f"  {row['model']:<14} caracteristica dominanta: {row['top_feature']} "
                    f"({row['top_feature_share_pct']:.1f}% din total)  |  "
                    f"perturbabile: {row['perturbable_share_pct']:.1f}%")

    # ---- 3. Corelatia importanta atinsa <-> evaziune produsa ----
    logger.info("\n3. Corelatia dintre importanta atinsa si evaziunea produsa:")
    deltas = linkage.load_feature_deltas()
    evasion = pd.read_csv(config.EVASION_DIR / "evasion_summary.csv")
    mass = linkage.importance_mass(deltas, importance)
    merged, correlations = linkage.correlate_with_evasion(mass, evasion)
    merged.to_csv(config.SENSITIVITY_DIR / "importance_vs_evasion.csv", index=False)
    correlations.to_csv(config.SENSITIVITY_DIR / "importance_evasion_correlation.csv", index=False)
    for _, row in correlations.iterrows():
        logger.info(f"  {row['model']:<14} Spearman rho = "
                    f"{row['spearman_importance_mass_weighted']:.3f} "
                    f"(p = {row['pvalue_importance_mass_weighted']:.4f}, "
                    f"n = {row['n_variants']} variante)")

    # ---- 4. Interpretarea rezultatului (c) si proximitatea de granita ----
    logger.info(f"\n4. Analiza rezultatelor pe varianta {FOCUS_VARIANT}:")
    proximity_rows, destination_tables = [], {}
    for name in models:
        per_row = _load_per_row(name)
        if per_row is None:
            logger.info(f"  {name}: lipsesc rezultatele per flux — ruleaza `python run_evasion.py`")
            continue
        baseline_proba, classes = _load_baseline_proba(name)

        destinations = outcomes_analysis.outcome_destinations(per_row, FOCUS_VARIANT)
        destination_tables[name] = destinations
        if not destinations.empty:
            destinations.to_csv(
                config.SENSITIVITY_DIR / f"confusion_destinations_{name.lower()}.csv")

        proximity = outcomes_analysis.boundary_proximity(per_row, baseline_proba,
                                                          classes, FOCUS_VARIANT)
        if proximity:
            proximity["model"] = name
            proximity_rows.append(proximity)
            if "confidence_gap" in proximity:
                logger.info(f"  {name:<14} P(clasa reala) pe trafic curat — "
                            f"evadate: {proximity['mean_p_true_clean_evaded']:.3f}  "
                            f"ne-evadate: {proximity['mean_p_true_clean_not_evaded']:.3f}  "
                            f"(diferenta {proximity['confidence_gap']:+.3f})")

    if proximity_rows:
        pd.DataFrame(proximity_rows).to_csv(
            config.SENSITIVITY_DIR / "boundary_proximity.csv", index=False)

    # ---- 5. De ce diverg modelele pe clasa Generic ----
    target = config.SENSITIVITY_DIVERGENCE_CLASS
    logger.info(f"\n5. Divergenta pe clasa {target} (arborii rezista, Transformer-ul cedeaza):")
    class_rows = test[test[config.TARGET] == target]
    class_sample = class_rows.sample(min(len(class_rows), config.SENSITIVITY_SAMPLE_SIZE),
                                     random_state=config.RANDOM_SEED)
    X_class = class_sample.drop(columns=feature_exclude)
    y_class = class_sample[config.TARGET].to_numpy()

    # Donor extern OBLIGATORIU aici: in clasa Generic, sttl e 254 pentru 98,8%
    # dintre randuri si dttl e 0 pentru 97,1%. O permutare interna nu ar schimba
    # practic nimic si ar raporta importanta ~0 tocmai pentru caracteristicile de
    # care modelul depinde cel mai mult. Valorile de inlocuire se trag deci din
    # distributia intregului set de test.
    donor = test.drop(columns=feature_exclude)
    class_importance = pd.concat(
        [importance_utils.permutation_importance(model, X_class, y_class,
                                                 model_name=name, donor=donor)
         for name, model in models.items()],
        ignore_index=True)
    class_importance.to_csv(
        config.SENSITIVITY_DIR / f"permutation_importance_{target.lower()}.csv", index=False)
    divergence = outcomes_analysis.divergence_analysis(class_importance, target)
    divergence.to_csv(config.SENSITIVITY_DIR / "divergence_analysis.csv", index=False)
    for name in models:
        top = divergence[divergence["model"] == name].head(3)
        listed = ", ".join(f"{r.feature}({r.share_pct:.0f}%)" for r in top.itertuples())
        logger.info(f"  {name:<14} {listed}")

    # ---- Figuri ----
    figures.plot_importance(importance, config.FIGURES_DIR / "sensitivity_importance.png")
    figures.plot_importance_vs_evasion(merged, correlations,
                                        config.FIGURES_DIR / "sensitivity_importance_vs_evasion.png")
    for name, destinations in destination_tables.items():
        figures.plot_outcome_destinations(
            destinations, config.FIGURES_DIR / f"sensitivity_destinations_{name.lower()}.png",
            FOCUS_VARIANT, name)
    logger.info(f"\n[salvat] figuri in {config.FIGURES_DIR}")

    payload = {
        "analysed_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": int(len(sample)),
        "n_repeats": config.SENSITIVITY_N_REPEATS,
        "focus_variant": FOCUS_VARIANT,
        "divergence_class": target,
        "correlations": correlations.to_dict(orient="records"),
        "attribution": attribution.to_dict(orient="records"),
        "random_seed": config.RANDOM_SEED,
        "library_versions": utils.library_versions(),
    }
    utils.save_json(payload, config.SENSITIVITY_DIR / "sensitivity_summary.json")
    logger.info(f"\nArtefacte in {config.SENSITIVITY_DIR}")
    return payload
