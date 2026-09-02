"""Ponderile de clasa INGHETATE, comune tuturor bratelor O6.

DE CE INGHETATE. In O1, ambele familii de modele isi calculeaza ponderile la
momentul antrenarii, din numaratorile setului primit: Random Forest prin
class_weight="balanced", reteaua prin compute_class_weight("balanced", ...).
Daca O6 si C1 ar pastra acest comportament, ponderile s-ar recalcula pe
numaratorile augmentate si bratele ar diferi de O1 in DOUA lucruri deodata —
datele si functia de cost. Comparatia nu ar mai fi controlata.

Solutia: se calculeaza o singura data, pe etichetele ORIGINALE, si se transmit
explicit tuturor bratelor. Augmentarea schimba atunci doar datele, nu obiectivul.
Vectorul exact ajunge in manifest.

XGBoost nu foloseste ponderi de clasa in O1 (nici sample_weight, nici
scale_pos_weight), deci pentru el nu exista nimic de inghetat: obiectivul e deja
identic intre brate. Se consemneaza explicit, ca absenta sa nu fie citita ca o
omisiune.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def balanced_weights(y_original) -> dict:
    """Ponderile "balanced" calculate pe etichetele ORIGINALE, ca dictionar.

    Parameters:
        y_original (array-like): etichetele setului de antrenare neaugmentat.

    Returns:
        dict: eticheta de clasa -> pondere, in forma acceptata de
            RandomForestClassifier(class_weight=...).
    """
    classes = np.unique(np.asarray(y_original))
    values = compute_class_weight("balanced", classes=classes, y=np.asarray(y_original))
    return {cls: float(w) for cls, w in zip(classes, values)}


def torch_weights(weight_map: dict, class_order) -> "np.ndarray":
    """Reordoneaza ponderile in ordinea codurilor folosite de retea.

    Parameters:
        weight_map (dict): rezultatul lui balanced_weights.
        class_order (array-like): clasele in ordinea codificarii (LabelEncoder).

    Returns:
        np.ndarray: vector float32 de lungimea numarului de clase.
    """
    return np.array([weight_map[cls] for cls in class_order], dtype=np.float32)
