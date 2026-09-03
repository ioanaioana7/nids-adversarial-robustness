"""Importanta prin permutare aplicata modelelor O6, nu doar celor de referinta.

DE CE EXISTA ACEST MODUL
    O3 arata CA apararea reduce evaziunea, dar nu si PRIN CE. Un model poate
    deveni invariant in doua feluri complet diferite: fie a incetat sa se
    sprijine pe caracteristica pe care atacatorul o modifica, fie a invatat
    valorile ei perturbate, pastrandu-si dependenta. Cele doua situatii dau
    aceleasi rate de evaziune pe familia cunoscuta, dar se comporta diferit in
    afara ei.

    Distinctia se decide aplicand modelelor reantrenate exact procedura din O4.

DE CE SE COMPARA CU C1, NU CU O1
    Adaugarea de exemple schimba simultan volumul, proportia claselor si
    (potential) importanta caracteristicilor. Comparand o6 direct cu o1, o
    eventuala deplasare a importantei ar putea proveni din simplul volum, nu din
    perturbare — exact confuzia pentru care grupul de control exista. Se ruleaza
    prin urmare ambele brate, iar marimea raportata este diferenta o6 - c1.

COMPARABILITATE
    Se reutilizeaza `importance.stratified_sample` si `permutation_importance`
    cu parametrii impliciti, deci acelasi esantion (seed fix -> exact aceleasi
    randuri), acelasi numar de permutari si aceeasi definitie a fractiunii
    accesibile ca la rularea publicata pentru O1. Singura diferenta este modelul
    pe care se aplica.

    Incarcarea trece prin `trees.load_tree_model` si
    `transformer_arm.load_transformer`, aceleasi cai folosite la evaluarea O6,
    deci predictia e la fel de determinista.
"""

import time

import pandas as pd

from src import config, data_loader
from src.adversarial import trees, transformer_arm
from src.sensitivity import importance as importance_utils, linkage

ARMS = (config.O6_ARM_DEFENDED, config.O6_ARM_CONTROL)
FAMILIES = ("RandomForest", "XGBoost", "Transformer")
OUTPUT_DIR = config.ADVERSARIAL_DIR / "sensitivity_arms"


def _load(family: str, arm: str, seed: int):
    """Modelul unui brat, pe aceeasi cale de incarcare ca la evaluarea O6."""
    if family == "Transformer":
        return transformer_arm.load_transformer(arm, seed)
    return trees.load_tree_model(family, arm, seed)


def _perturbable_features() -> list[str]:
    """Caracteristicile pe care atacatorul le poate modifica efectiv.

    Aceeasi definitie ca in src.sensitivity.runner, ca fractiunea accesibila sa
    fie comparabila intre rulari.
    """
    return sorted(set(config.SOURCE_CONTROLLED + config.CONNECTION_RATE_FEATURES
                      + config.DERIVED_FEATURES + config.CAUSALLY_FORCED_BY_DUR))


def run(logger, seed: int = config.RANDOM_SEED) -> dict:
    """Ruleaza importanta prin permutare pentru fiecare (brat, familie).

    Salveaza dupa fiecare model si sare peste ce e deja calculat, ca o
    intrerupere sa nu piarda rularile incheiate.

    Parameters:
        logger: logger pentru progres.
        seed (int): seed-ul esantionului si al permutarilor.

    Returns:
        dict: fractiunea accesibila per brat, ca rezumat al rularii.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, test = data_loader.load_data()
    sample = importance_utils.stratified_sample(test, seed=seed)
    X = sample.drop(columns=[config.ID_COL, config.TARGET, config.LABEL_COL])
    y = sample[config.TARGET].to_numpy()
    logger.info(f"Esantion: {len(X):,} randuri, {X.shape[1]} caracteristici "
                f"x {config.SENSITIVITY_N_REPEATS} permutari")

    for arm in ARMS:
        for family in FAMILIES:
            partial = OUTPUT_DIR / f"importance_{arm}_{family.lower()}.csv"
            if partial.exists():
                logger.info(f"  [{arm}/{family}] deja calculat")
                continue
            started = time.time()
            frame = importance_utils.permutation_importance(
                _load(family, arm, seed), X, y, model_name=family, seed=seed)
            frame.to_csv(partial, index=False)
            top = frame.nlargest(1, "detection_drop").iloc[0]
            logger.info(f"  [{arm}/{family}] {(time.time() - started) / 60:.1f} min, "
                        f"dominanta: {top['feature']}={top['detection_drop']:.2f}")

    summary = {}
    for arm in ARMS:
        importance = pd.concat(
            [pd.read_csv(OUTPUT_DIR / f"importance_{arm}_{f.lower()}.csv")
             for f in FAMILIES], ignore_index=True)
        attribution = linkage.single_feature_attribution(importance, _perturbable_features())
        importance.to_csv(OUTPUT_DIR / f"permutation_importance_{arm}.csv", index=False)
        attribution.to_csv(OUTPUT_DIR / f"importance_attribution_{arm}.csv", index=False)
        summary[arm] = attribution.set_index("model")["perturbable_share_pct"].to_dict()
        logger.info(f"\nFractiunea accesibila, bratul {arm}:")
        for model, value in summary[arm].items():
            logger.info(f"  {model:<14} {value:.2f}%")

    return summary
