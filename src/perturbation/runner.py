"""Orchestrarea generarii variantelor si scrierea artefactelor O2.

Nu masoara evaziunea. Produce variante validate + metadatele de care O3 are
nevoie ca sa le regenereze identic si sa le masoare.
"""

import json

import numpy as np
import pandas as pd

from src import config, data_loader, inference, utils
from src.perturbation import dependencies, generators, schema, validators

# Coloane pe care variante anume au voie sa le modifice, prin exceptie.
EXTRA_MUTABLE = {"ttl_both": ("dttl",)}


def load_frozen_models(include_transformer: bool = False) -> dict:
    """Incarca modelele inghetate prin calea DETERMINISTA comuna (src.inference).

    Deleaga la src.inference.load_frozen_models ca verificarea de aici si
    masurarea din O3 sa foloseasca exact acelasi drum de predictie. Altfel, un
    rand pe muchie de cutit (vezi src/inference.py) s-ar putea rezolva diferit
    intre verificare si masurare si ar fi numarat drept evaziune provocata de
    perturbare, cand de fapt e doar ordinea firelor de executie.

    Returns:
        dict: nume model -> obiect cu .predict / .predict_proba / .classes_.
    """
    return inference.load_frozen_models(include_transformer=include_transformer)


def predict(models: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Prezice cu modelele inghetate, returnand etichete de clasa.

    Parameters:
        models (dict): rezultatul lui load_frozen_models().
        X (pd.DataFrame): cadrul de features.

    Returns:
        dict[str, np.ndarray]: nume model -> etichete prezise.
    """
    return inference.predict(models, X)


def verify_identity_reproduces_baseline(models: dict, X_attack: pd.DataFrame,
                                         ids: np.ndarray, logger) -> dict:
    """Verificarea cea mai importanta din O2: nivelul 0 reproduce exact baseline-ul O1.

    Daca predictiile pe varianta-identitate difera de
    results/predictions/baseline_predictions.csv, harnessul insusi introduce
    drift si orice rata de evaziune masurata ulterior ar fi un artefact.

    Parameters:
        models (dict): modelele inghetate.
        X_attack (pd.DataFrame): fluxurile de atac (features).
        ids (np.ndarray): identificatorii corespunzatori.
        logger (logging.Logger): logger.

    Returns:
        dict: rezultatul verificarii, per model.

    Raises:
        validators.ValidityError: la orice nepotrivire.
    """
    baseline = pd.read_csv(config.PREDICTIONS_DIR / "baseline_predictions.csv")
    baseline = baseline.set_index("sample_id").loc[ids]

    preds = predict(models, X_attack)
    result = {}
    for model_name, predicted in preds.items():
        expected = baseline[f"{model_name.lower()}_prediction"].to_numpy()
        mismatches = int((np.asarray(predicted) != expected).sum())
        result[model_name] = {"rows": int(len(expected)), "mismatches": mismatches}
        if mismatches:
            raise validators.ValidityError(
                f"varianta-identitate NU reproduce baseline-ul pentru {model_name}: "
                f"{mismatches} nepotriviri din {len(expected)} randuri")
        logger.info(f"  [identitate] {model_name}: {len(expected)} randuri, 0 nepotriviri")
    return result


def feature_deltas(original: pd.DataFrame, perturbed: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Cuantifica ce s-a schimbat efectiv intre original si varianta.

    Returns:
        pd.DataFrame: o linie per coloana modificata, cu procentul de randuri
            afectate si magnitudinea schimbarii.
    """
    rows = []
    for col in original.columns:
        o, p = original[col].to_numpy(), perturbed[col].to_numpy()
        changed = o != p
        n_changed = int(changed.sum())
        if n_changed == 0:
            continue
        entry = {"variant": variant, "feature": col, "rows_changed": n_changed,
                 "pct_rows_changed": round(100.0 * n_changed / len(original), 3)}
        if pd.api.types.is_numeric_dtype(original[col]):
            delta = np.abs(p[changed].astype(float) - o[changed].astype(float))
            entry.update({"mean_abs_delta": float(delta.mean()),
                          "median_abs_delta": float(np.median(delta)),
                          "max_abs_delta": float(delta.max())})
        rows.append(entry)
    return pd.DataFrame(rows)


def save_ct_state_ttl_analysis(ct_lookup: pd.Series, normal_lookup: pd.Series,
                               train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Salveaza maparea empirica ct_state_ttl si dovada ca NU e o regula pe intervale TTL.

    Specificatia presupunea o regula recuperabila de binning peste intervale de
    TTL. Masuratoarea o infirma: valori TTL adiacente dau ieșiri diferite, deci
    ct_state_ttl e un contor pe fereastra glisanta, nu o functie de intervale.
    """
    train_combos = set(train.groupby(["state", "sttl", "dttl"]).groups.keys())
    test_combos = set(test.groupby(["state", "sttl", "dttl"]).groups.keys())

    int_dttl0 = {
        int(sttl): int(value)
        for (state, sttl, dttl), value in ct_lookup.items()
        if state == "INT" and dttl == 0
    }
    counterexamples = [
        {"pair": "sttl=62 vs sttl=63", "state": "INT", "dttl": 0,
         "values": [int_dttl0.get(62), int_dttl0.get(63)]},
        {"pair": "sttl=254 vs sttl=252", "state": "INT", "dttl": 0,
         "values": [int_dttl0.get(254), int_dttl0.get(252)]},
    ]

    payload = {
        "rule_type": "empirical_lookup",
        "why_not_a_binning_rule": (
            "Valori TTL adiacente produc ct_state_ttl diferit (vezi counterexamples), "
            "deci nicio partitionare pe intervale de TTL nu poate reproduce maparea. "
            "ct_state_ttl este un contor pe fereastra de 100 de conexiuni, imposibil "
            "de recalculat din inregistrari de flux izolate."
        ),
        "counterexamples": counterexamples,
        "state_INT_dttl0_sttl_to_ct_state_ttl": int_dttl0,
        "coverage": {
            "combos_train": len(train_combos),
            "combos_test": len(test_combos),
            "combos_test_unseen_in_train": len(test_combos - train_combos),
            "combos_in_lookup": int(len(ct_lookup)),
            "observed_ct_state_ttl_values": sorted(int(v) for v in set(ct_lookup.values)),
        },
        "policies": {
            "hold": "ct_state_ttl pastrat neschimbat (margine conservatoare)",
            "mimic": "lookup exact -> mod al traficului normal la acel sttl -> "
                     f"constanta {config.CT_STATE_TTL_FALLBACK}",
        },
        "normal_traffic_sttl_to_ct_state_ttl": {
            int(k): int(v) for k, v in normal_lookup.items()
        },
        "lookup": [
            {"state": state, "sttl": int(sttl), "dttl": int(dttl), "ct_state_ttl": int(value)}
            for (state, sttl, dttl), value in ct_lookup.items()
        ],
    }
    utils.save_json(payload, config.PERTURBATION_DIR / "ct_state_ttl_rule.json")


def run(logger) -> dict:
    """Genereaza, valideaza si salveaza intreaga grila de variante O2.

    Parameters:
        logger (logging.Logger): logger pentru consola si perturbation.log.

    Returns:
        dict: rezumatul rularii (numar variante, verificari, cai artefacte).
    """
    utils.set_global_seed(config.RANDOM_SEED)
    utils.ensure_output_dirs()

    train, test = data_loader.load_data()
    feature_cols = schema.feature_columns(test)

    attack_rows = test[test[config.TARGET] != "Normal"].reset_index(drop=True)
    X_attack = attack_rows[feature_cols]
    ids = attack_rows[config.ID_COL].to_numpy()
    logger.info("=" * 70)
    logger.info("O2 — GENERARE VARIANTE PERTURBATE")
    logger.info("=" * 70)
    logger.info(f"Fluxuri de atac in setul de test: {len(X_attack)}")

    models = load_frozen_models()
    logger.info(f"Modele inghetate incarcate: {list(models)}")

    # ---- Verificarea critica: identitatea reproduce baseline-ul O1 ----
    logger.info("\nVerificare identitate (nivel 0) vs baseline O1:")
    identity_check = verify_identity_reproduces_baseline(models, X_attack, ids, logger)

    # ---- Eligibilitate per model, ca O3 sa poata subselecta ----
    baseline = pd.read_csv(config.PREDICTIONS_DIR / "baseline_predictions.csv") \
        .set_index("sample_id").loc[ids]
    eligibility = pd.DataFrame({
        "sample_id": ids,
        "true_label": attack_rows[config.TARGET].to_numpy(),
        "rf_eligible": (baseline["randomforest_prediction"] != "Normal").to_numpy(),
        "xgb_eligible": (baseline["xgboost_prediction"] != "Normal").to_numpy(),
    })
    eligibility.to_csv(config.PERTURBATION_DIR / "eligible_flags.csv", index=False)
    logger.info(f"\nEligibile (alarma pe traficul curat): "
                f"RF={int(eligibility.rf_eligible.sum())}, "
                f"XGB={int(eligibility.xgb_eligible.sum())} din {len(eligibility)}")

    # ---- Lookup-urile ct_state_ttl ----
    ctx = generators.build_context(train, test)
    save_ct_state_ttl_analysis(ctx["ct_lookup"], ctx["normal_lookup"], train, test)
    logger.info(f"ct_state_ttl: {len(ctx['ct_lookup'])} combinatii in lookup")

    # ---- Grila de variante ----
    manifest_rows, validity_frames, delta_frames = [], [], []
    grid = generators.variant_grid()
    logger.info(f"\nGenerare {len(grid)} variante:")

    for ptype, level in grid:
        variant = f"{ptype}_L{level}"
        perturbed, meta = generators.generate_variant(X_attack, ptype, level, ctx)

        report = validators.validate(X_attack, perturbed, variant,
                                     extra_mutable=EXTRA_MUTABLE.get(ptype, ()))
        validity_frames.append(report)

        deltas = feature_deltas(X_attack, perturbed, variant)
        delta_frames.append(deltas)

        ct = meta["ct_state_ttl"]
        manifest_rows.append({
            "variant": variant,
            "type": ptype,
            "level": level,
            "description": generators.PERTURBATION_TYPES[ptype][2],
            "params": json.dumps(meta["params"], sort_keys=True),
            "n_rows": len(perturbed),
            "n_features_changed": len(deltas),
            "features_changed": ",".join(deltas["feature"]) if not deltas.empty else "",
            "ct_policy": ct["policy"],
            "ct_resolved_exact": ct["resolved_exact"],
            "ct_resolved_normal_fallback": ct["resolved_normal_fallback"],
            "ct_resolved_constant_fallback": ct["resolved_constant_fallback"],
            "rows_zero_dur": meta["propagation"].get("rows_zero_dur", 0),
            "rows_single_packet": meta["propagation"].get("rows_single_packet", 0),
            "rows_needing_extra_packets": meta["propagation"].get("rows_needing_extra_packets", 0),
            "seed": config.RANDOM_SEED,
            "checks_run": len(report),
            "checks_failed": int((report["violations"] > 0).sum()),
        })

        sample = perturbed.head(config.PERTURBATION_SAMPLE_ROWS).copy()
        sample.insert(0, config.ID_COL, ids[:len(sample)])
        sample.to_csv(config.PERTURBATION_SAMPLES_DIR / f"{variant}.csv", index=False)

        if config.PERTURBATION_PERSIST_VARIANTS:
            config.PERTURBATION_VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
            full = perturbed.copy()
            full.insert(0, config.ID_COL, ids)
            full.to_csv(config.PERTURBATION_VARIANTS_DIR / f"{variant}.csv.gz",
                        index=False, compression="gzip")

        logger.info(f"  {variant:<22} {len(deltas):>2} features modificate, "
                    f"{len(report)} verificari OK")

    # ---- Compatibilitate de schema: modelul accepta variantele ----
    logger.info("\nVerificare compatibilitate schema (modelul accepta variantele):")
    probe_type, probe_level = "combined_mimic", len(config.COMBINED_LEVELS)
    probe, _ = generators.generate_variant(X_attack.head(64), probe_type, probe_level, ctx)
    probe_preds = predict(models, probe)
    for model_name, values in probe_preds.items():
        logger.info(f"  {model_name}: {len(values)} predictii pe {probe_type}_L{probe_level} OK")

    # ---- Artefacte ----
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(config.PERTURBATION_DIR / "perturbation_manifest.csv", index=False)
    validity = pd.concat(validity_frames, ignore_index=True)
    validity.to_csv(config.PERTURBATION_DIR / "validity_report.csv", index=False)
    pd.concat(delta_frames, ignore_index=True).to_csv(
        config.PERTURBATION_DIR / "feature_deltas.csv", index=False)

    summary = {
        "attack_rows": int(len(X_attack)),
        "variants_generated": int(len(manifest)),
        "checks_run": int(len(validity)),
        "checks_failed": int((validity["violations"] > 0).sum()),
        "identity_check": identity_check,
        "eligible": {"RandomForest": int(eligibility.rf_eligible.sum()),
                     "XGBoost": int(eligibility.xgb_eligible.sum())},
        "persisted_full_variants": bool(config.PERTURBATION_PERSIST_VARIANTS),
        "random_seed": config.RANDOM_SEED,
        "library_versions": utils.library_versions(),
    }
    utils.save_json(summary, config.PERTURBATION_DIR / "perturbation_summary.json")

    logger.info("\n" + "=" * 70)
    logger.info(f"Variante generate: {summary['variants_generated']}")
    logger.info(f"Verificari rulate: {summary['checks_run']}, "
                f"picate: {summary['checks_failed']}")
    logger.info(f"Artefacte in {config.PERTURBATION_DIR}")
    logger.info("=" * 70)
    return summary
