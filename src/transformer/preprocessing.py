"""Preprocesarea proprie a FT-Transformer-ului.

DE CE E NEVOIE DE EA. Pipeline-ul de arbori trece coloanele numerice mai departe
NESCALATE ("passthrough"), pentru ca arborii folosesc doar ordinea valorilor, nu
magnitudinea lor. O retea neuronala este insa sensibila la scara: cu `sbytes` de
ordinul 1e7 alaturi de `is_ftp_login` in {0,1}, gradientii sunt dominati de
coloanele mari si antrenarea diverge sau stagneaza.

Prin urmare Transformer-ul primeste propria preprocesare, dar peste ACELEASI
features, ACELEASI randuri si ACELASI split ca arborii:
  - `proto`/`service`/`state` trec prin acelasi RareCategoryGrouper ca arborii
    (vocabular categorial identic, deci comparabil), apoi sunt codificate ca
    indici intregi pentru nn.Embedding — NU one-hot, embedding-urile fiind
    tocmai ideea din FT-Transformer;
  - coloanele numerice sunt standardizate cu StandardScaler.

Totul se ajusteaza EXCLUSIV pe train si se salveaza impreuna cu modelul, ca
inferenta sa fie reproductibila si ca frame-urile perturbate din O2 sa fie
transformate identic.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.preprocessing import RareCategoryGrouper


class TabularPreprocessor:
    """Transforma un DataFrame brut UNSW-NB15 in tensori numerici + categoriali.

    Attributes:
        numeric_columns_ (list[str]): coloanele numerice, in ordinea de intrare.
        categorical_columns_ (list[str]): coloanele categorice.
        categories_ (dict[str, list]): vocabularul per coloana categorica.
        cardinalities_ (list[int]): dimensiunea vocabularului per coloana.
    """

    def __init__(self, categorical_columns=None, min_freq=config.MIN_CATEGORY_FREQUENCY,
                 unknown_label=config.TRANSFORMER_UNKNOWN_CATEGORY):
        self.categorical_columns = list(categorical_columns or config.CATEGORICAL_COLUMNS)
        self.min_freq = min_freq
        self.unknown_label = unknown_label

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        """Ajusteaza gruparea categoriilor rare, vocabularele si scalerul, pe TRAIN.

        Parameters:
            X (pd.DataFrame): features de antrenare (fara id/attack_cat/label).

        Returns:
            TabularPreprocessor: self.
        """
        self.categorical_columns_ = [c for c in self.categorical_columns if c in X.columns]
        self.numeric_columns_ = [c for c in X.columns if c not in self.categorical_columns_]

        # Acelasi grouper ca la arbori: categoriile sub prag devin "other".
        self.grouper_ = RareCategoryGrouper(
            columns=self.categorical_columns_, min_freq=self.min_freq,
            other_label=self.unknown_label).fit(X)
        grouped = self.grouper_.transform(X)

        # Vocabular per coloana. Indexul 0 e rezervat bucket-ului necunoscut, ca
        # orice valoare nevazuta la inferenta sa aiba unde sa cada.
        self.categories_ = {}
        for col in self.categorical_columns_:
            seen = [v for v in sorted(grouped[col].astype(str).unique())
                    if v != self.unknown_label]
            self.categories_[col] = [self.unknown_label] + seen
        self.category_index_ = {
            col: {value: i for i, value in enumerate(values)}
            for col, values in self.categories_.items()
        }
        self.cardinalities_ = [len(self.categories_[c]) for c in self.categorical_columns_]

        self.scaler_ = StandardScaler().fit(X[self.numeric_columns_].to_numpy(dtype=np.float64))
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transforma un DataFrame brut in (numerice standardizate, indici categoriali).

        Parameters:
            X (pd.DataFrame): features brute, cu schema originala UNSW-NB15.

        Returns:
            tuple[np.ndarray, np.ndarray]: (float32 [n, n_num], int64 [n, n_cat]).

        Raises:
            KeyError: daca lipsesc coloane fata de cele vazute la fit.
        """
        missing = [c for c in self.numeric_columns_ + self.categorical_columns_
                   if c not in X.columns]
        if missing:
            raise KeyError(f"coloane lipsa fata de fit(): {missing}")

        grouped = self.grouper_.transform(X)
        x_num = self.scaler_.transform(
            grouped[self.numeric_columns_].to_numpy(dtype=np.float64)).astype(np.float32)

        codes = np.empty((len(X), len(self.categorical_columns_)), dtype=np.int64)
        for j, col in enumerate(self.categorical_columns_):
            mapping = self.category_index_[col]
            # Valorile nevazute cad pe indexul 0 (bucket-ul necunoscut).
            codes[:, j] = grouped[col].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
        return x_num, codes

    def fit_transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ajusteaza pe X si returneaza imediat transformarea lui X."""
        return self.fit(X).transform(X)

    @property
    def n_numeric(self) -> int:
        """Numarul de coloane numerice."""
        return len(self.numeric_columns_)

    @property
    def n_categorical(self) -> int:
        """Numarul de coloane categorice."""
        return len(self.categorical_columns_)
