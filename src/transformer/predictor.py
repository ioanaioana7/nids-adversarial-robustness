"""TransformerNIDS — invelis cu interfata sklearn peste reteaua antrenata.

Scopul: O2 si O3 sa trateze toate cele trei modele uniform. Predictorul accepta
un DataFrame BRUT cu schema originala UNSW-NB15 (exact ce produce
generate_variant din O2) si aplica intern lantul
RareCategoryGrouper -> scaler/encodere -> tensori -> forward -> softmax -> etichete.
"""

import numpy as np
import pandas as pd
import torch

from src import config
from src.transformer.model import FTTransformer
from src.transformer.preprocessing import TabularPreprocessor


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Alege CUDA daca e disponibil, altfel CPU.

    Parameters:
        prefer_cuda (bool): daca False, forteaza CPU.

    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TransformerNIDS:
    """Predictor cu API in stil sklearn: predict / predict_proba / classes_.

    Attributes:
        classes_ (np.ndarray): etichetele de clasa, in aceeasi ordine ca la RF/XGBoost.
    """

    def __init__(self, model: FTTransformer, preprocessor: TabularPreprocessor,
                 classes: np.ndarray, device: torch.device | None = None,
                 batch_size: int = config.TRANSFORMER_EVAL_BATCH_SIZE):
        self.model = model
        self.preprocessor = preprocessor
        self.classes_ = np.asarray(classes)
        self.device = device or resolve_device()
        self.batch_size = batch_size
        self.model.to(self.device).eval()

    def _forward_logits(self, X: pd.DataFrame) -> np.ndarray:
        """Ruleaza reteaua pe loturi si returneaza logits, fara gradienti."""
        x_num, x_cat = self.preprocessor.transform(X)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(x_num), self.batch_size):
                stop = start + self.batch_size
                num = torch.from_numpy(x_num[start:stop]).to(self.device)
                cat = torch.from_numpy(x_cat[start:stop]).to(self.device)
                outputs.append(self.model(num, cat).cpu())
        if not outputs:
            return np.empty((0, len(self.classes_)), dtype=np.float32)
        return torch.cat(outputs).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilitati per clasa, cu ordinea coloanelor egala cu self.classes_.

        Parameters:
            X (pd.DataFrame): features brute UNSW-NB15.

        Returns:
            np.ndarray: [n_samples, n_classes].
        """
        logits = torch.from_numpy(self._forward_logits(X))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Eticheta de clasa prezisa (string), ca la RF/XGBoost.

        Parameters:
            X (pd.DataFrame): features brute UNSW-NB15.

        Returns:
            np.ndarray: etichete de tip string, [n_samples].
        """
        return self.classes_[self._forward_logits(X).argmax(axis=1)]

    # ------------------------------------------------------------------
    # Persistenta
    # ------------------------------------------------------------------
    def save(self, path=config.TRANSFORMER_MODEL_PATH) -> None:
        """Salveaza ponderile impreuna cu preprocesarea ajustata si clasele.

        Inferenta devine astfel auto-continuta: nu e nevoie de datele de
        antrenare ca sa reconstruiesti scalerul sau vocabularele.
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "preprocessor": self.preprocessor,
            "classes": self.classes_,
            "architecture": {
                "n_numeric": self.preprocessor.n_numeric,
                "cardinalities": self.preprocessor.cardinalities_,
                "n_classes": len(self.classes_),
            },
            "hyperparameters": self.model.hyperparameters(),
        }, path)

    @classmethod
    def load(cls, path=config.TRANSFORMER_MODEL_PATH,
             device: torch.device | None = None) -> "TransformerNIDS":
        """Reincarca un predictor complet de pe disc.

        Parameters:
            path (Path | str): fisierul .pt salvat de save().
            device (torch.device | None): dispozitivul tinta; auto-detectat daca lipseste.

        Returns:
            TransformerNIDS
        """
        device = device or resolve_device()
        # weights_only=False: bundle-ul contine si obiectele sklearn de preprocesare.
        bundle = torch.load(path, map_location=device, weights_only=False)
        arch = bundle["architecture"]
        model = FTTransformer(n_numeric=arch["n_numeric"],
                              cardinalities=arch["cardinalities"],
                              n_classes=arch["n_classes"])
        model.load_state_dict(bundle["model_state_dict"])
        return cls(model, bundle["preprocessor"], bundle["classes"], device=device)

    # ------------------------------------------------------------------
    # Hook pentru extinderi viitoare — NEIMPLEMENTAT INTENTIONAT
    # ------------------------------------------------------------------
    def logits_with_grad(self, X: pd.DataFrame):
        """PLACEHOLDER pentru atacuri bazate pe gradient (FGSM/PGD) — in afara scopului.

        Spre deosebire de RF/XGBoost, aceasta retea e diferentiabila end-to-end,
        deci gradientul pierderii se poate propaga pana la intrarile numerice.
        Asta ar permite atacuri adversariale de tip white-box, ca extindere
        viitoare a lucrarii.

        ATENTIE metodologica pentru cine implementeaza asta ulterior: un gradient
        calculat in spatiul standardizat produce perturbari care NU respecta
        automat constrangerile fizice impuse in O2 (directionalitate, limite de
        domeniu, consistenta caracteristicilor derivate). Rezultatul ar fi un
        vector de features, nu un flux de retea realizabil. De aceea sarcina
        curenta trece Transformer-ul prin ACELEASI perturbari realizabile din O2
        ca arborii, ca cele trei modele sa fie comparate in aceleasi conditii.

        Raises:
            NotImplementedError: intotdeauna; exista doar ca punct de extindere.
        """
        raise NotImplementedError(
            "Atacurile pe gradient (FGSM/PGD) sunt in afara scopului acestei etape. "
            "Vezi docstring-ul pentru constrangerile de realizabilitate care trebuie "
            "respectate inainte de a le implementa.")
