"""
Registre des executeurs de cible.

Ajouter une cible (carte physique, conteneur pilote a distance...) revient
a ecrire une sous-classe de TargetRunner et a l'enregistrer ici ;
benchmark.py n'a pas a changer.
"""

from runners.base import DEVELOPMENT, LEVELS, TargetRunner, parse_level
from runners.local import LocalRunner

RUNNERS = {
    LocalRunner.name: LocalRunner,
}


def get_runner(name, level):
    if name not in RUNNERS:
        raise ValueError(f"Executeur inconnu : {name!r} (disponibles : {sorted(RUNNERS)}).")
    return RUNNERS[name](level)
