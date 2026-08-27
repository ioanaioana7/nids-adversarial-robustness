"""O4 — analiza de sensibilitate.

Leaga rata de evaziune masurata in O3 de importanta caracteristicilor pe care
perturbarile le ating, si interpreteaza cele trei rezultate posibile ale unei
perturbari. Nu reantreneaza nimic: lucreaza pe modelele inghetate, prin calea
determinista din src.inference.
"""

from src.sensitivity.importance import permutation_importance

__all__ = ["permutation_importance"]
