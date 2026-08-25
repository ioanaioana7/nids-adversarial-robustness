"""Orchestrarea O3: masurare, agregare, artefacte."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config, data_loader, inference, utils
from src.evasion import aggregation, figures, measurement
from src.perturbation import generators, schema


def _load_attack_frame() -> tuple:
    """Fluxurile de atac din setul de test oficial, in ordinea folosita de O2.

    Returns:
        tuple: (X features, etichete reale, identificatori, train, test).
    """
    train, test = data_loader.load_data()
    attack_rows = test[test[config.TARGET] != "Normal"].reset_index(drop=True)
    return (attack_rows[schema.feature_columns(test)],
            attack_rows[config.TARGET].to_numpy(),
            attack_rows[config.ID_COL].to_numpy(),
            train, test)


def _write_per_row(per_row_frames: list, logger) -> dict:
    """Salveaza rezultatele per rand, cate un fisier comprimat per model.

    Necesare pentru O4 (corelarea evaziunii cu importanta caracteristicilor) si
    pentru analiza intensitatii minime de evaziune. Se scriu comprimat si sunt
    excluse din git, ca celelalte artefacte voluminoase regenerabile.
    """
    combined = pd.concat(per_row_frames, ignore_index=True)
    written = {}
    for model, group in combined.groupby("model"):
        path = config.EVASION_PER_ROW_DIR / f"outcomes_{model.lower()}.csv.gz"
        group.drop(columns="model").to_csv(path, index=False, compression="gzip")
        written[model] = {"rows": int(len(group)), "path": str(path.name)}
        logger.info(f"  [salvat] {path.name} ({len(group):,} randuri)")
    return combined, written


def run(logger) -> dict:
    """Masoara rata de evaziune pe toata grila de variante, pentru toate modelele.

    Returns:
        dict: rezumatul rularii.
    """
    utils.set_global_seed(config.RANDOM_SEED)
    utils.ensure_output_dirs()

    logger.info("=" * 70)
    logger.info("O3 — MASURAREA RATEI DE EVAZIUNE")
    logger.info("=" * 70)

    X, true_labels, ids, train, test = _load_attack_frame()
    logger.info(f"Fluxuri de atac in setul de test: {len(X):,}")

    # Calea determinista comuna: aceleasi predictii ca la verificarea din O2, deci
    # o egalitate pe muchie de cutit nu poate fi numarata drept evaziune.
    models = inference.load_frozen_models(include_transformer=True)
    logger.info(f"Modele (predictie determinista): {list(models)}")

    references = {name: measurement.load_reference(name, ids) for name in models}
    for name, reference in references.items():
        n_eligible = int(reference["eligible"].sum())
        logger.info(f"  {name:<14} eligibile: {n_eligible:,} "
                    f"({100 * n_eligible / len(ids):.2f}%)")

    ctx = generators.build_context(train, test)
    summary, per_class, per_row_frames = measurement.run_sweep(
        models, references, X, true_labels, ids, ctx, logger)

    # ---- Invariantul: la nivelul 0 evaziunea trebuie sa fie exact 0 ----
    logger.info("\nVerificare invariant (varianta-identitate):")
    identity_check = aggregation.sanity_check_identity(summary, logger)

    # ---- Tabele agregate ----
    headline = aggregation.headline_table(summary)
    bounds = aggregation.bounds_table(summary)
    composition = aggregation.outcome_composition(summary)

    summary.drop(columns=[c for c in summary.columns if c.startswith("_")], errors="ignore") \
        .to_csv(config.EVASION_DIR / "evasion_summary.csv", index=False)
    per_class.to_csv(config.EVASION_DIR / "evasion_per_class.csv", index=False)
    bounds.to_csv(config.EVASION_DIR / "evasion_bounds.csv", index=False)
    composition.to_csv(config.EVASION_DIR / "outcome_composition.csv", index=False)
    headline.to_csv(config.EVASION_DIR / "evasion_headline.csv")

    per_row, per_row_written = (None, {})
    if config.EVASION_PERSIST_PER_ROW and per_row_frames:
        logger.info("")
        per_row, per_row_written = _write_per_row(per_row_frames, logger)
        minimum = aggregation.minimum_evasion_intensity(per_row)
        minimum.to_csv(config.EVASION_DIR / "minimum_evasion_intensity.csv", index=False)

    # ---- Figuri ----
    figures.plot_evasion_curves(summary, config.FIGURES_DIR / "evasion_curves.png")
    for model in sorted(summary["model"].unique()):
        figures.plot_outcome_composition(
            composition, config.FIGURES_DIR / f"evasion_outcomes_{model.lower()}.png", model)
        figures.plot_per_class_heatmap(
            per_class, config.FIGURES_DIR / f"evasion_per_class_{model.lower()}.png", model)
    logger.info(f"\n[salvat] figuri in {config.FIGURES_DIR}")

    # ---- Rezultatul principal, ca interval ----
    logger.info("\n" + "=" * 70)
    logger.info("REZULTAT PRINCIPAL — perturbarea TTL (interval hold..mimic)")
    logger.info("=" * 70)
    ttl = bounds[bounds["family"] == "ttl"]
    for model in sorted(ttl["model"].unique()):
        rows = ttl[ttl["model"] == model].sort_values("level")
        parts = "  ".join(
            f"L{int(r.level)}: {r.evasion_lower_hold:5.1f}–{r.evasion_upper_mimic:5.1f}%"
            for r in rows.itertuples())
        logger.info(f"  {model:<14} {parts}")
    logger.info("=" * 70)

    summary_payload = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "attack_rows": int(len(X)),
        "variants": int(summary["variant"].nunique()),
        "models": list(models),
        "eligible": {name: int(ref["eligible"].sum()) for name, ref in references.items()},
        "identity_invariant": identity_check,
        "per_row_files": per_row_written,
        "random_seed": config.RANDOM_SEED,
        "library_versions": utils.library_versions(),
    }
    utils.save_json(summary_payload, config.EVASION_DIR / "evasion_run_summary.json")
    logger.info(f"\nArtefacte in {config.EVASION_DIR}")
    return summary_payload
