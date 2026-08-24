"""O2 — modul de generare a variantelor perturbate ale traficului malitios.

Nu masoara evaziunea (aceea e O3): se opreste la "exista variante perturbate
valide si demonstrabil bine-formate".
"""

from src.perturbation.generators import PERTURBATION_TYPES, generate_variant, variant_grid

__all__ = ["generate_variant", "variant_grid", "PERTURBATION_TYPES"]
