"""Transformari de preprocesare: gruparea categoriilor rare si encoderul principal."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src import config


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Inlocuieste categoriile rare (sub min_freq aparitii in train) sau
    necunoscute (aparute doar la inferenta) cu other_label, per coloana.

    Util mai ales pentru 'proto', care are 133 de valori distincte, majoritatea
    aproape unice.
    """

    def __init__(self, columns, min_freq=config.MIN_CATEGORY_FREQUENCY,
                 other_label=config.OTHER_CATEGORY_LABEL):
        self.columns = columns
        self.min_freq = min_freq
        self.other_label = other_label

    def fit(self, X, y=None):
        """Retine, per coloana, multimea categoriilor cu >= min_freq aparitii in X.

        Parameters:
            X (pd.DataFrame): date de antrenare.
            y: neutilizat, prezent pentru compatibilitate cu API-ul sklearn.

        Returns:
            RareCategoryGrouper: self.
        """
        self.keep_ = {
            col: set(X[col].value_counts()[lambda s: s >= self.min_freq].index)
            for col in self.columns
        }
        return self

    def transform(self, X):
        """Inlocuieste valorile din afara multimii invatate cu other_label.

        Parameters:
            X (pd.DataFrame): date de transformat (train sau test/inferenta).

        Returns:
            pd.DataFrame: copie a X cu categoriile rare/necunoscute inlocuite.
        """
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].where(X[col].isin(self.keep_[col]), self.other_label)
        return X


def build_preprocessor(numeric_cols):
    """Construieste ColumnTransformer-ul: one-hot pe coloanele categorice, passthrough numeric.

    Parameters:
        numeric_cols (list[str]): numele coloanelor numerice.

    Returns:
        sklearn.compose.ColumnTransformer
    """
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLUMNS),
        ("num", "passthrough", numeric_cols),
    ])
