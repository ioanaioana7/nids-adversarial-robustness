"""O3 — masurarea ratei de evaziune pe variantele perturbate generate de O2.

Nu genereaza variante (aceea e O2) si nu reantreneaza nimic (O1 ramane inghetat):
incarca modelele prin calea determinista din src.inference, regenereaza variantele
determinist si numara ce se intampla cu fiecare flux.
"""

from src.evasion.outcomes import classify_outcomes, evasion_summary

__all__ = ["classify_outcomes", "evasion_summary"]
