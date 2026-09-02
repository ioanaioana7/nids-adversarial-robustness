"""Construirea seturilor de antrenare augmentate pentru O6 si pentru controlul C1.

TREI DECIZII CARE FAC REZULTATUL APARABIL

1. CONTEXT CONSTRUIT DOAR PE TRAIN. Generatoarele din O2 isi construiesc
   lookup-urile ct_state_ttl din train+test, ceea ce e corect atata timp cat
   modelele sunt inghetate: statisticile descriu senzorul, nu modelul, si nu
   intra niciodata in antrenare. In O6 insa variantele perturbate DEVIN date de
   antrenare, deci orice statistica derivata din test ar fi scurgere. Aici se
   apeleaza build_context(train) si se verifica explicit provenienta.

2. COPIILE SE ADAUGA, NU INLOCUIESC. Fluxurile de atac originale raman in setul
   de antrenare. Cu N randuri in total si A randuri de atac, adaugarea a k copii
   per flux de atac da N + k*A randuri — nu 2N.

3. C1 ESTE CONSTRUIT DIN ACELEASI RANDURI. Bratul de control adauga exact
   aceleasi randuri sursa, in acelasi numar, dar NEPERTURBATE. Astfel O6 si C1
   au acelasi numar de randuri, aceeasi distributie de clase si acelasi efort de
   antrenare; difera intr-un singur lucru — daca randurile adaugate sunt sau nu
   perturbate.

ESANTIONAREA VARIANTELOR
   Uniforma peste tipurile realizabile, apoi uniforma peste nivelurile tipului
   ales. Deliberat NU ponderata dupa ratele de evaziune masurate in O3: a pondera
   dupa exact marimea pe care o masuram ulterior ar inclina evaluarea in favoarea
   apararii. Probabilitatile efective se scriu in manifest.
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.perturbation import generators, schema

# Coloanele de evidenta atasate randurilor generate. Nu ajung niciodata in
# matricea de features: sunt eliminate inainte de antrenare (vezi training_frame).
PROVENANCE_COLUMNS = ["source_id", "variant_type", "variant_level", "origin", "split_side"]

ORIGIN_ORIGINAL = "original"
ORIGIN_PERTURBED = "perturbed"
ORIGIN_DUPLICATE = "duplicate"


def realizable_types(holdout: tuple = ()) -> list[str]:
    """Tipurile de perturbare admise in augmentare.

    Se exclud: identitatea (nu modifica nimic), tipurile marcate nerealizabile in
    O3 (ttl_both atinge dttl, care apartine victimei) si eventualele tipuri
    scoase deliberat pentru bratul de generalizare.

    Parameters:
        holdout (tuple[str]): tipuri excluse suplimentar (bratul o6_loo).

    Returns:
        list[str]: numele tipurilor, in ordinea din PERTURBATION_TYPES.
    """
    excluded = {"identity", *config.EVASION_NON_REALIZABLE_TYPES, *holdout}
    return [name for name in generators.PERTURBATION_TYPES if name not in excluded]


def variant_catalog(holdout: tuple = ()) -> list[tuple[str, int]]:
    """Perechile (tip, nivel) din care se esantioneaza.

    Returns:
        list[tuple[str, int]]: toate nivelurile fiecarui tip admis.
    """
    catalog = []
    for name in realizable_types(holdout):
        _, n_levels, _ = generators.PERTURBATION_TYPES[name]
        catalog.extend((name, level) for level in range(1, n_levels + 1))
    return catalog


def sampling_probabilities(holdout: tuple = ()) -> dict:
    """Probabilitatea fiecarei variante sub schema "uniform tip, apoi uniform nivel".

    Un tip cu 3 niveluri primeste aceeasi masa totala ca unul cu 4, deci
    variantele lui individuale sunt mai probabile. Se raporteaza explicit si masa
    pe familie (cate variante ating TTL-ul), ca alegerea sa fie auditabila.

    Returns:
        dict: probabilitati per varianta, per tip si pe familia TTL.
    """
    types = realizable_types(holdout)
    p_type = 1.0 / len(types)
    per_variant, per_type = {}, {}
    for name in types:
        _, n_levels, _ = generators.PERTURBATION_TYPES[name]
        per_type[name] = p_type
        for level in range(1, n_levels + 1):
            per_variant[f"{name}_L{level}"] = p_type / n_levels

    ttl_family = [n for n in types if n.startswith(("ttl", "combined"))]
    return {
        "scheme": config.O6_SAMPLING,
        "n_types": len(types),
        "n_variants": len(per_variant),
        "p_per_type": per_type,
        "p_per_variant": per_variant,
        "ttl_touching_types": ttl_family,
        "ttl_touching_mass": round(p_type * len(ttl_family), 6),
    }


def assign_variants(n_rows: int, seed: int, holdout: tuple = ()) -> pd.DataFrame:
    """Alege cate o varianta pentru fiecare rand, determinist.

    Determinismul e dat de (seed, numarul de randuri, ordinea randurilor). Cum
    setul de plecare e fixat si continutul lui e amprentat in manifest, alegerea
    facuta pentru un flux dat se poate reproduce exact. Extragerea e vectorizata,
    deci nu depinde de planificarea firelor sau a workerilor DataLoader-ului.

    Parameters:
        n_rows (int): cate atribuiri se genereaza.
        seed (int): seed-ul extragerii; pentru augmentarea dinamica se compune
            din seed-ul rularii si numarul epocii.
        holdout (tuple[str]): tipuri excluse.

    Returns:
        pd.DataFrame: coloanele variant_type si variant_level, cate un rand.
    """
    types = realizable_types(holdout)
    n_levels = np.array([generators.PERTURBATION_TYPES[t][1] for t in types])

    rng = np.random.default_rng(seed)
    type_idx = rng.integers(0, len(types), size=n_rows)
    # Un singur apel uniform, scalat la numarul de niveluri al tipului ales:
    # echivalent cu o extragere uniforma per tip, dar vectorizat.
    level = np.floor(rng.random(n_rows) * n_levels[type_idx]).astype(int) + 1
    level = np.minimum(level, n_levels[type_idx])

    return pd.DataFrame({
        "variant_type": np.asarray(types, dtype=object)[type_idx],
        "variant_level": level,
    })


def perturb_attack_rows(attack: pd.DataFrame, assignment: pd.DataFrame,
                        ctx: dict) -> pd.DataFrame:
    """Aplica variantei atribuite fiecarui rand, grupat pe (tip, nivel).

    Generatoarele din O2 sunt vectorizate peste un cadru intreg, deci se apeleaza
    o data per grup, nu o data per rand: cateva zecimi de secunda pentru intreg
    setul de atac, ceea ce face augmentarea dinamica per epoca practicabila.

    Parameters:
        attack (pd.DataFrame): randurile de atac brute (cu id/attack_cat/label).
        assignment (pd.DataFrame): variant_type / variant_level, aliniate pozitional.
        ctx (dict): contextul generatoarelor, construit DOAR pe train.

    Returns:
        pd.DataFrame: randurile perturbate, in ordinea originala, cu coloanele de
            evidenta atasate.
    """
    feature_cols = schema.feature_columns(attack)
    X = attack[feature_cols].reset_index(drop=True)
    meta = attack.drop(columns=feature_cols).reset_index(drop=True)
    assignment = assignment.reset_index(drop=True)

    pieces = []
    for (ptype, level), idx in assignment.groupby(["variant_type", "variant_level"]).groups.items():
        chunk = X.loc[idx]
        perturbed, _ = generators.generate_variant(chunk, ptype, int(level), ctx)
        # generate_variant pastreaza indexul, deci concatenarea + sortarea readuc
        # randurile exact in ordinea de plecare.
        pieces.append(perturbed)

    out = schema.restore_dtypes(pd.concat(pieces).sort_index(), X)
    if not out.index.equals(X.index):
        raise AssertionError("perturbarea a pierdut sau a reordonat randuri")

    result = pd.concat([meta, out], axis=1)
    result["source_id"] = attack[config.ID_COL].to_numpy()
    result["variant_type"] = assignment["variant_type"].to_numpy()
    result["variant_level"] = assignment["variant_level"].to_numpy()
    result["origin"] = ORIGIN_PERTURBED
    return result


def duplicate_attack_rows(attack: pd.DataFrame) -> pd.DataFrame:
    """Copiile NEPERTURBATE ale bratului de control C1.

    Aceleasi randuri, acelasi numar, nicio modificare. Diferenta fata de O6 e
    exact una singura: daca randurile adaugate sunt sau nu perturbate.
    """
    result = attack.reset_index(drop=True).copy()
    result["source_id"] = attack[config.ID_COL].to_numpy()
    result["variant_type"] = "none"
    result["variant_level"] = 0
    result["origin"] = ORIGIN_DUPLICATE
    return result


def build_augmented(base: pd.DataFrame, ctx: dict, mode: str, seed: int,
                    k: int = config.O6_K, holdout: tuple = (),
                    split_side: str = "train") -> pd.DataFrame:
    """Construieste un set de antrenare augmentat: original + k copii de atac.

    Parameters:
        base (pd.DataFrame): cadrul original (id/attack_cat/label + features).
        ctx (dict): contextul generatoarelor (train-only pentru O6).
        mode (str): ORIGIN_PERTURBED pentru O6, ORIGIN_DUPLICATE pentru C1.
        seed (int): seed-ul esantionarii variantelor.
        k (int): cate copii per flux de atac.
        holdout (tuple[str]): tipuri excluse din esantionare.
        split_side (str): eticheta partii ("train"/"val"), pentru verificarea de
            apartenenta la aceeasi parte a split-ului.

    Returns:
        pd.DataFrame: originalul + copiile, cu coloanele de evidenta.
    """
    base = base.reset_index(drop=True)
    attack = base[base[config.TARGET] != "Normal"].reset_index(drop=True)

    original = base.copy()
    original["source_id"] = base[config.ID_COL].to_numpy()
    original["variant_type"] = "none"
    original["variant_level"] = 0
    original["origin"] = ORIGIN_ORIGINAL

    copies = []
    for copy_index in range(k):
        if mode == ORIGIN_PERTURBED:
            # Seed distinct per copie, ca a doua copie a unui flux sa nu repete
            # varianta primeia.
            assignment = assign_variants(len(attack), seed * 1_000 + copy_index, holdout)
            copies.append(perturb_attack_rows(attack, assignment, ctx))
        elif mode == ORIGIN_DUPLICATE:
            copies.append(duplicate_attack_rows(attack))
        else:
            raise ValueError(f"mod de augmentare necunoscut: {mode!r}")

    result = pd.concat([original, *copies], ignore_index=True)
    result["split_side"] = split_side
    return result


def training_frame(augmented: pd.DataFrame) -> pd.DataFrame:
    """Elimina coloanele de evidenta, lasand exact schema pe care o astepta modelul."""
    return augmented.drop(columns=PROVENANCE_COLUMNS, errors="ignore")


def stratified_split(train: pd.DataFrame, seed: int = config.RANDOM_SEED,
                     val_fraction: float = config.TRANSFORMER_VAL_FRACTION) -> tuple:
    """Reproduce EXACT split-ul train/validare folosit de O1 pentru retea.

    Indicii produsi de train_test_split depind doar de numarul de randuri, de
    etichetele de stratificare, de fractiune si de random_state — nu de faptul ca
    O1 imparte X si y separat, iar aici se imparte cadrul intreg. Egalitatea se
    verifica explicit in verify_split_matches_baseline.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (sub-setul de train, setul de validare).
    """
    return train_test_split(train, test_size=val_fraction,
                            stratify=train[config.TARGET], random_state=seed)


def verify_split_matches_baseline(sub: pd.DataFrame, val: pd.DataFrame,
                                  train: pd.DataFrame) -> dict:
    """Confirma ca split-ul de aici coincide, rand cu rand, cu cel din O1.

    Raises:
        AssertionError: daca vreun rand a schimbat partea.
    """
    feature_exclude = [config.ID_COL, config.TARGET, config.LABEL_COL]
    X_ref, y_ref = train.drop(columns=feature_exclude), train[config.TARGET]
    _, X_val_ref, _, _ = train_test_split(
        X_ref, y_ref, test_size=config.TRANSFORMER_VAL_FRACTION,
        stratify=y_ref, random_state=config.RANDOM_SEED)

    expected = set(train.loc[X_val_ref.index, config.ID_COL])
    actual = set(val[config.ID_COL])
    if expected != actual:
        raise AssertionError(
            f"split-ul de validare difera de cel din O1: {len(expected ^ actual)} randuri diferite")
    return {"train_rows": int(len(sub)), "val_rows": int(len(val)),
            "identical_to_baseline_split": True}


def verify_same_side(augmented: pd.DataFrame, allowed_ids: set, forbidden_ids: set) -> dict:
    """Confirma ca fiecare copie generata a ramas de aceeasi parte cu fluxul sursa.

    Daca o copie perturbata a unui flux de antrenare ar ajunge in validare,
    macro-F1-ul de validare — deci si early stopping-ul — ar fi contaminat.

    Raises:
        AssertionError: la orice sursa in afara partii permise.
    """
    sources = set(augmented["source_id"].unique())
    leaked = sources & forbidden_ids
    outside = sources - allowed_ids
    if leaked or outside:
        raise AssertionError(
            f"copii generate din randuri de pe cealalta parte a split-ului: "
            f"{len(leaked)} scurgeri, {len(outside)} surse necunoscute")
    return {"source_ids": int(len(sources)), "leaked": 0, "outside_allowed_set": 0}


def verify_context_is_train_only(ctx: dict, train_rows: int) -> dict:
    """Confirma ca lookup-urile generatoarelor NU au vazut setul de test.

    Verificarea de regresie ceruta explicit de planul O6: in O6 perturbarile
    devin date de antrenare, deci un lookup construit din test ar fi scurgere
    train/test.

    Raises:
        AssertionError: daca contextul a fost construit din mai mult de un cadru
            sau dintr-un cadru care nu e setul de train.
    """
    frames = ctx.get("source_frames")
    if frames != [train_rows]:
        raise AssertionError(
            f"contextul generatoarelor NU e construit doar pe train: "
            f"cadre sursa {frames}, asteptat [{train_rows}]")
    return {"source_frames": frames, "train_only": True}


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Amprenta SHA-256 a continutului unui cadru, pentru manifest.

    Se hash-uieste reprezentarea CSV a cadrului sortat pe coloane, ca doua rulari
    care produc acelasi set sa dea aceeasi amprenta indiferent de ordinea in care
    au fost construite coloanele.
    """
    payload = frame.reindex(columns=sorted(frame.columns)).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def composition(augmented: pd.DataFrame) -> dict:
    """Numaratori pe care manifestul trebuie sa le contina: marimi, prior de clasa."""
    counts = augmented[config.TARGET].value_counts()
    n = int(len(augmented))
    n_normal = int(counts.get("Normal", 0))
    return {
        "rows": n,
        "rows_original": int((augmented["origin"] == ORIGIN_ORIGINAL).sum()),
        "rows_added": int((augmented["origin"] != ORIGIN_ORIGINAL).sum()),
        "rows_normal": n_normal,
        "rows_attack": n - n_normal,
        "attack_to_normal_ratio": round((n - n_normal) / n_normal, 6) if n_normal else None,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_prior": {str(k): round(v / n, 6) for k, v in counts.items()},
        "variant_counts": {str(k): int(v) for k, v in
                           augmented["variant_type"].value_counts().items()},
    }
