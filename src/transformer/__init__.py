"""FT-Transformer — al treilea clasificator (PyTorch), aditiv fata de O1 si O2.

Aceleasi features, aceleasi randuri, acelasi split oficial ca RF/XGBoost;
difera doar preprocesarea numerica (standardizare, obligatorie pentru o retea
neuronala) si encodarea categoricelor (embedding-uri in loc de one-hot).
"""

from src.transformer.predictor import TransformerNIDS

__all__ = ["TransformerNIDS"]
