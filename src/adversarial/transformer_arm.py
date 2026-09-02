"""Antrenarea FT-Transformer-ului pe bratele O6 / C1 / O1, cu augmentare dinamica.

DE CE DINAMICA. Arborii primesc un set fix: fiecare flux de atac apare cu exact
k variante, alese o singura data. Reteaua parcurge insa zeci de epoci, deci
poate vedea o varianta NOUA a aceluiasi flux la fiecare trecere. Diversitatea de
perturbari pe care o vede reteaua este prin urmare mai mare decat a arborilor;
planul O6 cere explicit sa NU se incerce egalarea celor doua (ar insemna k=4-8
la arbori, adica o distorsiune masiva a raportului atac/Normal), ci sa se
raporteze diferenta si sa se trateze comparatia intre familii ca fiind
orientativa.

CE RAMANE NESCHIMBAT FATA DE O1
  - split-ul 85/15 stratificat, cu acelasi seed;
  - preprocesarea (scaler + vocabulare) ajustata pe sub-setul de train CURAT.
    Refit-ul pe date augmentate ar introduce un al doilea canal de diferenta
    intre C1 si O6 (statistici de normalizare diferite), deci reprezentarea de
    intrare se pastreaza identica pentru toate bratele;
  - criteriul de selectie: macro-F1 NETEZIT pe validarea CURATA. Daca early
    stopping-ul ar fi condus de o validare adversariala, O6 ar schimba si datele,
    si criteriul, iar comparatia cu O1 nu ar mai fi controlata. Robustetea se
    masoara DUPA antrenare, cu O3, nu se foloseste la alegerea checkpoint-ului.

CE SE SCHIMBA
  - continutul unei epoci: original + copii de atac (perturbate la O6,
    duplicate la C1, deloc la O1);
  - ponderile de clasa, care sunt cele inghetate din O1 (vezi weights.py).
"""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from src import config
from src.adversarial import augment, weights as weights_module
from src.perturbation import schema
from src.transformer import training
from src.transformer.model import FTTransformer
from src.transformer.predictor import TransformerNIDS, resolve_device

MODEL_NAME = "Transformer"


def model_path(arm: str, seed: int):
    """Calea fisierului salvat pentru un (brat, seed)."""
    return config.ADVERSARIAL_MODELS_DIR / f"transformer_{arm}_s{seed}.pt"


class EpochAugmenter:
    """Produce DataLoader-ul unei epoci: baza curata + copii de atac.

    Tensorii bazei curate se calculeaza o singura data. La fiecare epoca se
    reesantioneaza doar variantele randurilor de atac, se genereaza cadrul
    perturbat si se trece prin aceeasi preprocesare ajustata pe train curat.

    DETERMINISM. Extragerea variantelor foloseste un generator initializat cu
    (seed_rulare, numar_epoca) si e vectorizata peste randuri intr-o ordine fixa,
    inainte de construirea DataLoader-ului. Varianta atribuita unui flux la o
    epoca data se reproduce prin urmare exact, indiferent de planificarea
    firelor sau a workerilor — spre deosebire de o extragere facuta in
    __getitem__, care ar depinde de ordinea de acces.
    """

    def __init__(self, train_sub: pd.DataFrame, preprocessor, encoder: LabelEncoder,
                 ctx: dict, mode: str, seed: int, holdout: tuple = (),
                 k: int = config.O6_K):
        self.mode = mode
        self.seed = seed
        self.holdout = tuple(holdout)
        self.k = k
        self.ctx = ctx
        self.preprocessor = preprocessor
        self.encoder = encoder

        feature_exclude = [config.ID_COL, config.TARGET, config.LABEL_COL]
        self.base_X = train_sub.drop(columns=feature_exclude).reset_index(drop=True)
        self.base_y = encoder.transform(train_sub[config.TARGET])
        self.base_num, self.base_cat = preprocessor.transform(self.base_X)

        attack_mask = (train_sub[config.TARGET] != "Normal").to_numpy()
        self.attack = train_sub[attack_mask].reset_index(drop=True)
        self.attack_X = self.attack.drop(columns=feature_exclude).reset_index(drop=True)
        self.attack_y = encoder.transform(self.attack[config.TARGET])
        self.feature_cols = schema.feature_columns(train_sub)

        if mode == augment.ORIGIN_DUPLICATE:
            # Copiile lui C1 sunt aceleasi la fiecare epoca: nu are ce sa se
            # reesantioneze, deci tensorii se calculeaza o singura data.
            self.dup_num = np.repeat(preprocessor.transform(self.attack_X)[0], k, axis=0)
            self.dup_cat = np.repeat(preprocessor.transform(self.attack_X)[1], k, axis=0)
            self.dup_y = np.repeat(self.attack_y, k)

        self.rows_per_epoch = len(self.base_X) + (
            0 if mode == augment.ORIGIN_ORIGINAL else k * len(self.attack_X))

    def epoch_seed(self, epoch: int) -> int:
        """Seed-ul extragerii pentru o epoca, derivat din seed-ul rularii."""
        return int(self.seed) * 1_000_003 + int(epoch)

    def variants_for_epoch(self, epoch: int) -> pd.DataFrame:
        """Variantele atribuite randurilor de atac la epoca data (k blocuri)."""
        frames = []
        for copy_index in range(self.k):
            frames.append(augment.assign_variants(
                len(self.attack_X), self.epoch_seed(epoch) + copy_index * 7919, self.holdout))
        return pd.concat(frames, ignore_index=True)

    def tensors_for_epoch(self, epoch: int) -> tuple:
        """Tensorii complete ai unei epoci: (numerice, categoriale, etichete)."""
        if self.mode == augment.ORIGIN_ORIGINAL:
            return self.base_num, self.base_cat, self.base_y
        if self.mode == augment.ORIGIN_DUPLICATE:
            return (np.concatenate([self.base_num, self.dup_num]),
                    np.concatenate([self.base_cat, self.dup_cat]),
                    np.concatenate([self.base_y, self.dup_y]))

        assignment = self.variants_for_epoch(epoch)
        stacked = pd.concat([self.attack] * self.k, ignore_index=True)
        perturbed = augment.perturb_attack_rows(stacked, assignment, self.ctx)
        num, cat = self.preprocessor.transform(perturbed[self.base_X.columns])
        return (np.concatenate([self.base_num, num]),
                np.concatenate([self.base_cat, cat]),
                np.concatenate([self.base_y, np.tile(self.attack_y, self.k)]))

    def loader_for_epoch(self, epoch: int):
        """DataLoader-ul epocii, cu amestecare initializata determinist."""
        num, cat, y = self.tensors_for_epoch(epoch)
        return training.make_loader(num, cat, y, config.TRANSFORMER_BATCH_SIZE,
                                    shuffle=True, seed=self.epoch_seed(epoch))

    def expected_distinct_variants(self, epochs: int) -> int:
        """Cate variante distincte poate vedea un flux pe parcursul antrenarii.

        Marimea pe care planul O6 cere sa fie raportata alaturi de k-ul arborilor,
        pentru ca diferenta de expunere adversariala intre familii sa fie vizibila.
        """
        return int(min(epochs * self.k, len(augment.variant_catalog(self.holdout))))


def train_arm(arm: str, seed: int, train_sub: pd.DataFrame, val: pd.DataFrame,
              ctx: dict, frozen_weights: dict, logger,
              mode: str, holdout: tuple = (),
              max_epochs: int = config.TRANSFORMER_MAX_EPOCHS) -> dict:
    """Antreneaza un singur (brat, seed) si salveaza predictorul complet.

    Parameters:
        arm (str): numele bratului ("o1"/"c1"/"o6"/"o6_loo").
        seed (int): seed-ul rularii.
        train_sub (pd.DataFrame): sub-setul de train (85%), CURAT.
        val (pd.DataFrame): setul de validare (15%), CURAT — ramane curat.
        ctx (dict): contextul generatoarelor, construit doar pe train.
        frozen_weights (dict): ponderile de clasa inghetate din O1, calculate pe
            sub-setul de train ORIGINAL (85%) — exact multimea pe care O1 le-a
            calculat, nu pe setul de train intreg si nu pe cel augmentat.
        logger (logging.Logger): logger.
        mode (str): tipul de augmentare (original / perturbed / duplicate).
        holdout (tuple[str]): tipuri excluse (bratul de generalizare).
        max_epochs (int): plafonul de epoci.

    Returns:
        dict: rezumatul antrenarii si calea modelului.
    """
    path = model_path(arm, seed)
    training.seed_everything(seed)
    device = resolve_device()

    feature_exclude = [config.ID_COL, config.TARGET, config.LABEL_COL]
    X_sub = train_sub.drop(columns=feature_exclude)
    X_val = val.drop(columns=feature_exclude)

    # Encoder si preprocesare ajustate pe train CURAT — identic cu O1.
    from src.transformer.preprocessing import TabularPreprocessor
    encoder = LabelEncoder().fit(train_sub[config.TARGET])
    preprocessor = TabularPreprocessor().fit(X_sub)

    augmenter = EpochAugmenter(train_sub, preprocessor, encoder, ctx, mode, seed, holdout)

    num_val, cat_val = preprocessor.transform(X_val)
    val_loader = training.make_loader(num_val, cat_val, encoder.transform(val[config.TARGET]),
                                      config.TRANSFORMER_EVAL_BATCH_SIZE, shuffle=False)
    # Loader de rezerva: pastreaza semnatura functiei de antrenare din O1 si
    # ofera generatorul pe care il consulta checkpointing-ul.
    base_loader = training.make_loader(augmenter.base_num, augmenter.base_cat,
                                       augmenter.base_y, config.TRANSFORMER_BATCH_SIZE,
                                       shuffle=True, seed=seed)

    model = FTTransformer(n_numeric=preprocessor.n_numeric,
                          cardinalities=preprocessor.cardinalities_,
                          n_classes=len(encoder.classes_))
    weight_vector = torch.from_numpy(
        weights_module.torch_weights(frozen_weights, encoder.classes_))

    logger.info(f"\n[{arm}/s{seed}] {augmenter.rows_per_epoch:,} randuri/epoca "
                f"({len(augmenter.base_X):,} originale + "
                f"{augmenter.rows_per_epoch - len(augmenter.base_X):,} adaugate), "
                f"validare curata {len(val):,}")

    # Bratul de referinta nu augmenteaza nimic, deci foloseste EXACT calea din O1:
    # un singur DataLoader construit inainte de bucla. Asa, la seed-ul 42 el
    # trebuie sa reproduca bit-cu-bit modelul inghetat, ceea ce se si verifica.
    factory = None if mode == augment.ORIGIN_ORIGINAL else augmenter.loader_for_epoch

    started = time.time()
    summary, history = training.train(
        model, base_loader, val_loader, weight_vector, device, logger,
        max_epochs=max_epochs,
        # Checkpointing-ul pe epoca e dezactivat: starea DataLoader-ului nu mai e
        # suficienta pentru reluare cand loturile se regenereaza la fiecare epoca.
        # Reluarea se face la granularitate de BRAT (un brat deja salvat e sarit).
        checkpoint_path=None, resume=False,
        loader_factory=factory)

    predictor = TransformerNIDS(model, preprocessor, encoder.classes_, device=device)
    predictor.save(path)

    elapsed = time.time() - started
    logger.info(f"[{arm}/s{seed}] gata in {elapsed / 60:.1f} min -> {path.name}")

    return {
        "arm": arm, "seed": seed, "path": str(path),
        "mode": mode, "holdout": list(holdout),
        "rows_per_epoch": int(augmenter.rows_per_epoch),
        "epochs_run": summary["epochs_run"],
        "best_epoch": summary["best_epoch"],
        "best_val_macro_f1_smoothed": summary["best_val_macro_f1_smoothed"],
        "best_val_macro_f1_raw": summary["best_val_macro_f1_raw"],
        "expected_distinct_variants_per_flow":
            augmenter.expected_distinct_variants(summary["epochs_run"]),
        "seconds": round(elapsed, 1),
        "history": history,
    }


def load_transformer(arm: str, seed: int) -> TransformerNIDS:
    """Reincarca predictorul salvat pentru un (brat, seed)."""
    return TransformerNIDS.load(model_path(arm, seed))
