"""Orchestrarea O6: augmentare -> antrenare -> evaluare -> verdict.

Etapele sunt separate si reluabile: fiecare model deja salvat este sarit, deci o
intrerupere costa cel mult bratul in curs, nu toata rularea.
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config, data_loader, utils
from src.adversarial import augment, evaluation, trees, weights as weights_module
from src.perturbation import generators, schema

# brat -> (mod de augmentare, tipuri excluse)
ARM_MODES = {
    config.O6_ARM_BASELINE: (augment.ORIGIN_ORIGINAL, ()),
    config.O6_ARM_CONTROL: (augment.ORIGIN_DUPLICATE, ()),
    config.O6_ARM_DEFENDED: (augment.ORIGIN_PERTURBED, ()),
    config.O6_ARM_HOLDOUT: (augment.ORIGIN_PERTURBED, tuple(config.O6_HOLDOUT_TYPES)),
}


# ==========================================================================
# Etapa 1 — datele
# ==========================================================================

def build_tree_set(arm: str, train: pd.DataFrame, ctx_train: dict) -> pd.DataFrame:
    """Setul de antrenare al arborilor pentru un brat.

    Constructia e determinista (seed fix, extragere vectorizata), deci apelul se
    poate repeta ori de cate ori e nevoie in loc sa se tina cadrul in memorie.
    Amprenta din manifest confirma ca reconstructia da acelasi continut.
    """
    mode, holdout = ARM_MODES[arm]
    k = 0 if mode == augment.ORIGIN_ORIGINAL else config.O6_K
    return augment.build_augmented(train, ctx_train, mode, config.RANDOM_SEED,
                                   k=k, holdout=holdout)


def prepare(logger) -> dict:
    """Construieste contextele, seturile augmentate si manifestul.

    Returns:
        dict: tot ce au nevoie etapele urmatoare (cadre, contexte, ponderi).
    """
    train, test = data_loader.load_data()
    labels = sorted(train[config.TARGET].unique())

    logger.info("=" * 74)
    logger.info("O6 — ANTRENARE ADVERSARIALA")
    logger.info("=" * 74)
    logger.info(f"Train: {len(train):,} randuri  |  Test: {len(test):,} randuri")

    # Contextul de ANTRENARE: doar train. In O6 perturbarile devin date de
    # antrenare, deci un lookup construit din test ar fi scurgere train/test.
    ctx_train = generators.build_context(train)
    context_check = augment.verify_context_is_train_only(ctx_train, len(train))

    # Contextul de EVALUARE: train+test, identic cu O3, ca variantele masurate sa
    # fie aceleasi. Modelele sunt deja inghetate cand se foloseste, deci nu poate
    # contamina antrenarea.
    ctx_eval = generators.build_context(train, test)
    logger.info(f"Context antrenare (doar train): {len(ctx_train['ct_lookup'])} combinatii "
                f"ct_state_ttl  |  context evaluare (ca O3): {len(ctx_eval['ct_lookup'])}")

    # ---- Split-ul retelei, INAINTE de augmentare ----
    train_sub, val = augment.stratified_split(train)
    split_check = augment.verify_split_matches_baseline(train_sub, val, train)
    logger.info(f"Split retea: train={len(train_sub):,}, validare={len(val):,} "
                f"(identic cu O1: {split_check['identical_to_baseline_split']})")

    # ---- Ponderi inghetate, per familie, pe multimea pe care O1 le-a calculat ----
    frozen_trees = weights_module.balanced_weights(train[config.TARGET])
    frozen_transformer = weights_module.balanced_weights(train_sub[config.TARGET])
    logger.info("Ponderi inghetate (arbori, pe train intreg): "
                + ", ".join(f"{c}={w:.2f}" for c, w in sorted(frozen_trees.items())))

    # ---- Seturile de antrenare ale arborilor ----
    # Se construiesc, se amprenteaza si se ELIBEREAZA: sunt deterministe, deci
    # etapa de antrenare le reconstruieste in 0,6 s, iar patru cadre de 294.682
    # de randuri tinute simultan in memorie ar concura inutil cu padurea.
    provenance = {}
    for arm in (config.O6_ARM_BASELINE, config.O6_ARM_CONTROL,
                config.O6_ARM_DEFENDED, config.O6_ARM_HOLDOUT):
        started = time.time()
        frame = build_tree_set(arm, train, ctx_train)
        provenance[arm] = augment.composition(frame)
        provenance[arm]["fingerprint"] = augment.frame_fingerprint(augment.training_frame(frame))
        provenance[arm]["seconds"] = round(time.time() - started, 2)
        columns = ["source_id", config.TARGET, "variant_type", "variant_level", "origin"]
        frame[columns].to_csv(config.O6_PROVENANCE_DIR / f"provenance_{arm}.csv.gz",
                              index=False, compression="gzip")
        ratio = provenance[arm]["attack_to_normal_ratio"]
        logger.info(f"  set {arm:<7} {len(frame):,} randuri "
                    f"({provenance[arm]['rows_added']:,} adaugate), "
                    f"atac:Normal = {ratio:.3f}")
        del frame

    # ---- Verificarea "aceeasi parte a split-ului" pentru bratul retelei ----
    sub_ids, val_ids = set(train_sub[config.ID_COL]), set(val[config.ID_COL])
    sub_augmented = augment.build_augmented(train_sub, ctx_train, augment.ORIGIN_PERTURBED,
                                            config.RANDOM_SEED, k=config.O6_K)
    same_side = augment.verify_same_side(sub_augmented, sub_ids, val_ids)
    logger.info(f"  copiile retelei raman de partea sursei: {same_side}")

    manifest = build_manifest(train, test, ctx_train, ctx_eval, context_check, split_check,
                              same_side, provenance, frozen_trees, frozen_transformer)
    utils.save_json(manifest, config.O6_MANIFEST_PATH)
    logger.info(f"\n[salvat] {config.O6_MANIFEST_PATH.name}")

    return {"train": train, "test": test, "labels": labels,
            "ctx_train": ctx_train, "ctx_eval": ctx_eval,
            "train_sub": train_sub, "val": val,
            "frozen_trees": frozen_trees, "frozen_transformer": frozen_transformer,
            "manifest": manifest}


def build_manifest(train, test, ctx_train, ctx_eval, context_check, split_check,
                   same_side, provenance, frozen_trees, frozen_transformer) -> dict:
    """Manifestul de reproducibilitate, cu criteriile PRE-INREGISTRATE.

    Criteriile sunt scrise inainte de a se cunoaste rezultatele, tocmai ca
    verdictul sa nu poata fi ajustat dupa ele.
    """
    sampling = augment.sampling_probabilities()
    holdout_sampling = augment.sampling_probabilities(tuple(config.O6_HOLDOUT_TYPES))
    return {
        "objective": "O6 — antrenare adversariala prin augmentare",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.RANDOM_SEED,
        "k": config.O6_K,
        "arms": {arm: {"mode": mode, "holdout": list(holdout)}
                 for arm, (mode, holdout) in ARM_MODES.items()},
        "seeds": {"trees": config.O6_TREE_SEEDS,
                  "transformer": config.O6_TRANSFORMER_SEEDS,
                  "holdout": config.O6_HOLDOUT_SEEDS},
        "generator_context": {
            "training_context": {
                "built_from": "train",
                "check": context_check,
                "ct_state_ttl_combinations": int(len(ctx_train["ct_lookup"])),
                "why": "in O6 perturbarile devin date de antrenare, deci un lookup "
                       "construit din test ar fi scurgere train/test",
            },
            "evaluation_context": {
                "built_from": "train+test",
                "ct_state_ttl_combinations": int(len(ctx_eval["ct_lookup"])),
                "why": "identic cu O3, ca variantele masurate sa fie aceleasi; "
                       "modelele sunt inghetate cand se foloseste",
            },
        },
        "split": split_check,
        "same_side_check": same_side,
        "sampling": sampling,
        "sampling_holdout_arm": holdout_sampling,
        "frozen_class_weights": {
            "trees": {"computed_on": "etichetele train ORIGINALE (175.341 randuri)",
                      "weights": frozen_trees},
            "transformer": {"computed_on": "sub-setul de train ORIGINAL (85%, 149.039 randuri)",
                            "weights": frozen_transformer},
            "xgboost": "fara ponderi de clasa nici in O1; nimic de inghetat",
        },
        "datasets": provenance,
        "dataset_sizes": {"train": int(len(train)), "test": int(len(test))},
        "preregistered_criteria": {
            "primary_robustness": {
                "class": config.O6_CRITERION_CLASS,
                "min_improvement_pp_over_c1": config.O6_CRITERION_MIN_EVASION_DROP_PP,
                "measured_on": "cohorta comuna, medie peste seed-uri",
            },
            "primary_clean_cost": {
                "max_macro_f1_drop": config.O6_CRITERION_MAX_MACRO_F1_DROP,
                "measured_on": "setul de test oficial curat",
            },
            "safety": {
                "max_per_class_f1_drop": config.O6_CRITERION_MAX_PER_CLASS_F1_DROP,
            },
            "stretch_target": {
                "max_absolute_evasion_pct": config.O6_STRETCH_EVASION_ABS,
                "note": "tinta aspirationala, NU criteriu de trecere",
            },
        },
        "non_goals": [
            "atacuri pe gradient (FGSM/PGD/ZOO) — in afara modelului de amenintare",
            "atacatori adaptivi care se reoptimizeaza impotriva modelului aparat",
            "robustete dincolo de familia de perturbari O2",
        ],
        "library_versions": utils.library_versions(),
    }


# ==========================================================================
# Etapa 2 — arborii
# ==========================================================================

def train_trees(state: dict, logger) -> dict:
    """Antreneaza RF si XGBoost pe fiecare (brat, seed), sarind ce exista deja."""
    from src import eda

    numeric_cols = eda.numeric_feature_columns(state["train"])
    feature_exclude = [config.ID_COL, config.TARGET, config.LABEL_COL]
    results = {}

    logger.info("\n" + "=" * 74)
    logger.info("ARBORI — reantrenare pe bratele O6")
    logger.info("=" * 74)

    arms = list(config.O6_ARMS) + ([config.O6_ARM_HOLDOUT] if config.O6_RUN_HOLDOUT else [])
    for arm in arms:
        seeds = config.O6_HOLDOUT_SEEDS if arm == config.O6_ARM_HOLDOUT else config.O6_TREE_SEEDS
        todo = [(seed, name) for seed in seeds for name in trees.TREE_MODELS
                if not trees.model_path(name, arm, seed).exists()]
        if not todo:
            logger.info(f"  [{arm}] toate modelele exista deja, se sare")
            continue

        frame = augment.training_frame(build_tree_set(arm, state["train"], state["ctx_train"]))
        X = frame.drop(columns=feature_exclude)
        y = frame[config.TARGET]
        del frame
        for seed, model_name in todo:
            results[f"{model_name}|{arm}|{seed}"] = trees.train_tree_arm(
                model_name, arm, seed, X, y, numeric_cols, state["frozen_trees"], logger)
            utils.save_json(results, config.ADVERSARIAL_DIR / "tree_training_runs.json")
        del X, y
    return results


# ==========================================================================
# Etapa 3 — reteaua
# ==========================================================================

def train_transformers(state: dict, logger) -> dict:
    """Antreneaza reteaua pe fiecare (brat, seed), sarind ce exista deja."""
    from src.adversarial import transformer_arm

    logger.info("\n" + "=" * 74)
    logger.info("FT-TRANSFORMER — reantrenare pe bratele O6")
    logger.info("=" * 74)

    results = {}
    arms = list(config.O6_ARMS) + ([config.O6_ARM_HOLDOUT] if config.O6_RUN_HOLDOUT else [])
    for arm in arms:
        mode, holdout = ARM_MODES[arm]
        seeds = (config.O6_HOLDOUT_SEEDS if arm == config.O6_ARM_HOLDOUT
                 else config.O6_TRANSFORMER_SEEDS)
        for seed in seeds:
            path = transformer_arm.model_path(arm, seed)
            if path.exists():
                logger.info(f"  [{arm}/s{seed}] exista deja, se sare")
                continue
            results[f"Transformer|{arm}|{seed}"] = transformer_arm.train_arm(
                arm, seed, state["train_sub"], state["val"], state["ctx_train"],
                state["frozen_transformer"], logger, mode=mode, holdout=holdout)
            utils.save_json(results, config.ADVERSARIAL_DIR / "transformer_training_runs.json")
    return results


# ==========================================================================
# Etapa 4 — evaluarea
# ==========================================================================

def _attack_frame(test: pd.DataFrame) -> tuple:
    """Fluxurile de atac din test, in ordinea folosita de O2/O3."""
    attack = test[test[config.TARGET] != "Normal"].reset_index(drop=True)
    return (attack[schema.feature_columns(test)],
            attack[config.TARGET].to_numpy(),
            attack[config.ID_COL].to_numpy())


def _load_model(family: str, arm: str, seed: int):
    """Incarca modelul unei rulari, indiferent de familie."""
    if family == "Transformer":
        from src.adversarial import transformer_arm
        return transformer_arm.load_transformer(arm, seed)
    return trees.load_tree_model(family, arm, seed)


def _run_list() -> list:
    """Toate rularile (familie, brat, seed) care ar trebui sa existe."""
    runs = []
    arms = list(config.O6_ARMS) + ([config.O6_ARM_HOLDOUT] if config.O6_RUN_HOLDOUT else [])
    for arm in arms:
        for family in ("RandomForest", "XGBoost"):
            seeds = (config.O6_HOLDOUT_SEEDS if arm == config.O6_ARM_HOLDOUT
                     else config.O6_TREE_SEEDS)
            runs.extend((family, arm, seed) for seed in seeds)
        seeds = (config.O6_HOLDOUT_SEEDS if arm == config.O6_ARM_HOLDOUT
                 else config.O6_TRANSFORMER_SEEDS)
        runs.extend(("Transformer", arm, seed) for seed in seeds)
    return runs


def evaluate_runs(state: dict, logger) -> dict:
    """Evalueaza fiecare rulare: metrici pe trafic curat + matricea de rezultate O3."""
    test, labels = state["test"], state["labels"]
    feature_exclude = [config.ID_COL, config.TARGET, config.LABEL_COL]
    X_test, y_test = test.drop(columns=feature_exclude), test[config.TARGET]
    X_attack, true_attack, attack_ids = _attack_frame(test)

    (config.ADVERSARIAL_DIR / "outcomes").mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 74)
    logger.info("EVALUARE — trafic curat + grila O2 pe fiecare brat")
    logger.info("=" * 74)

    clean_rows, variants = [], None
    for family, arm, seed in _run_list():
        key = evaluation.run_key(family, arm, seed)
        path = evaluation.outcomes_path(family, arm, seed)
        clean_path = config.ADVERSARIAL_DIR / "outcomes" / f"clean_{family.lower()}_{arm}_s{seed}.json"

        if path.exists() and clean_path.exists():
            stored = json.loads(clean_path.read_text(encoding="utf-8"))
            clean_rows.append(stored)
            if variants is None:
                variants = json.loads(
                    (config.ADVERSARIAL_DIR / "outcomes" / "variants.json").read_text("utf-8"))
            logger.info(f"  [{key}] deja evaluat, se sare")
            continue

        started = time.time()
        model = _load_model(family, arm, seed)
        clean = clean_metrics_row(model, X_test, y_test, labels, family, arm, seed, test)
        if "baseline_reproduction" in clean:
            check = clean["baseline_reproduction"]
            logger.info(f"  [{key}] vs modelul inghetat O1: "
                        + (f"{check['mismatches']} nepotriviri din {check['rows']:,}"
                           if check["checked"] else check["reason"]))
        clean_rows.append(clean)
        utils.save_json(clean, clean_path)

        variants, matrix = evaluation.sweep_outcomes(
            model, X_attack, true_attack, state["ctx_eval"], logger, key)
        np.savez_compressed(path, matrix=matrix, ids=attack_ids)
        utils.save_json(variants, config.ADVERSARIAL_DIR / "outcomes" / "variants.json")

        logger.info(f"  [{key}] macro-F1 curat {clean['macro_f1']:.4f}  "
                    f"({time.time() - started:.0f}s)")
        del model

    frame = pd.DataFrame(clean_rows)
    frame.drop(columns=[c for c in ("per_class_f1", "baseline_reproduction")
                        if c in frame.columns]).to_csv(
        config.ADVERSARIAL_DIR / "clean_metrics.csv", index=False)
    utils.save_json({r["key"]: r["baseline_reproduction"] for r in clean_rows
                     if "baseline_reproduction" in r},
                    config.ADVERSARIAL_DIR / "baseline_reproduction.json")
    utils.save_json({r["key"]: r["per_class_f1"] for r in clean_rows},
                    config.ADVERSARIAL_DIR / "clean_per_class_f1.json")

    return {"clean": clean_rows, "variants": variants,
            "true_attack": true_attack, "attack_ids": attack_ids}


BASELINE_PREDICTION_SOURCES = {
    "RandomForest": (config.PREDICTIONS_DIR / "baseline_predictions.csv",
                     "randomforest_prediction"),
    "XGBoost": (config.PREDICTIONS_DIR / "baseline_predictions.csv", "xgboost_prediction"),
    "Transformer": (config.TRANSFORMER_PREDICTIONS_PATH, "transformer_prediction"),
}


def reproduces_baseline(family: str, predictions, test: pd.DataFrame) -> dict:
    """Compara bratul de referinta la seed-ul 42 cu modelul INGHETAT din O1.

    Bratul o1 primeste exact datele originale, exact hiperparametrii originali si
    ponderile de clasa recalculate ca dictionar explicit din aceleasi numaratori.
    La seed-ul 42 ar trebui prin urmare sa reproduca modelul inghetat rand cu
    rand. Daca nu o face, undeva in lantul O6 s-a strecurat o diferenta care ar
    contamina si celelalte brate, deci verificarea se raporteaza, nu se ascunde.
    """
    path, column = BASELINE_PREDICTION_SOURCES[family]
    if not path.exists():
        return {"checked": False, "reason": f"lipseste {path.name}"}
    stored = pd.read_csv(path).set_index("sample_id").reindex(test[config.ID_COL])
    expected = stored[column].to_numpy()
    mismatches = int((np.asarray(predictions) != expected).sum())
    return {"checked": True, "rows": int(len(expected)), "mismatches": mismatches,
            "identical": mismatches == 0}


def clean_metrics_row(model, X_test, y_test, labels, family, arm, seed,
                      test: pd.DataFrame = None) -> dict:
    """Metricile pe trafic curat ale unei rulari, ca rand de tabel."""
    metrics = evaluation.clean_metrics(model, X_test, y_test, labels)
    predictions = metrics.pop("predictions")
    metrics.update({"key": evaluation.run_key(family, arm, seed),
                    "model": family, "arm": arm, "seed": seed})
    if arm == config.O6_ARM_BASELINE and seed == config.RANDOM_SEED and test is not None:
        metrics["baseline_reproduction"] = reproduces_baseline(family, predictions, test)
    return metrics


def build_cohorts(state: dict, evaluated: dict, logger) -> dict:
    """Cohorta comuna per familie de model si ratele masurate pe ea."""
    true_attack = evaluated["true_attack"]
    variants = evaluated["variants"]

    logger.info("\n" + "=" * 74)
    logger.info("COHORTA COMUNA — intersectia fluxurilor semnalate de toate bratele")
    logger.info("=" * 74)

    matrices, masks_by_family = {}, {}
    for family, arm, seed in _run_list():
        data = np.load(evaluation.outcomes_path(family, arm, seed))
        matrix = data["matrix"]
        matrices[(family, arm, seed)] = matrix
        masks_by_family.setdefault(family, []).append(
            evaluation.eligible_from_matrix(variants, matrix))

    cohorts, per_run = {}, {}
    for family, masks in masks_by_family.items():
        cohort = evaluation.common_cohort(masks)
        cohorts[family] = cohort
        sizes = [int(m.sum()) for m in masks]
        logger.info(f"  {family:<14} eligibili per rulare: {min(sizes):,}–{max(sizes):,}  "
                    f"-> cohorta comuna n={int(cohort.sum()):,} "
                    f"({100 * cohort.sum() / len(cohort):.2f}% din fluxurile de atac)")

    own_rows = []
    for (family, arm, seed), matrix in matrices.items():
        key = evaluation.run_key(family, arm, seed)
        cohort = cohorts[family]
        evaluation.check_identity_invariant(variants, matrix, cohort, key)

        per_run[(family, arm, seed)] = evaluation.rates_on_cohort(
            variants, matrix, cohort, true_attack)

        own_mask = evaluation.eligible_from_matrix(variants, matrix)
        own = evaluation.rates_on_cohort(variants, matrix, own_mask, true_attack)
        own.insert(0, "model", family)
        own.insert(1, "arm", arm)
        own.insert(2, "seed", seed)
        own_rows.append(own)

    combined, aggregated = evaluation.aggregate_over_seeds(
        per_run, ["evasion_rate", "pct_correct", "pct_misclassified_attack",
                  f"evasion_{config.O6_CRITERION_CLASS}"])

    combined.to_csv(config.ADVERSARIAL_DIR / "cohort_evasion_per_seed.csv", index=False)
    aggregated.to_csv(config.ADVERSARIAL_DIR / "cohort_evasion_aggregated.csv", index=False)
    pd.concat(own_rows, ignore_index=True).to_csv(
        config.ADVERSARIAL_DIR / "own_eligible_evasion.csv", index=False)

    cohort_sizes = {family: {"n": int(mask.sum()),
                             "pct_of_attacks": round(100.0 * mask.sum() / len(mask), 3),
                             "per_class": {str(cls): int(((true_attack == cls) & mask).sum())
                                           for cls in sorted(set(true_attack))}}
                    for family, mask in cohorts.items()}
    utils.save_json(cohort_sizes, config.ADVERSARIAL_DIR / "cohort_sizes.json")

    return {"cohorts": cohorts, "per_seed": combined, "aggregated": aggregated,
            "cohort_sizes": cohort_sizes}
