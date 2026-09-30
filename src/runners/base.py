"""
Abstraction d'executeur de cible.

Un executeur sait ou il mesure (describe) et comment il mesure (_measure).
Les regles de la strategie de validation ne sont PAS laissees aux
sous-classes : elles sont appliquees ici, dans run_benchmark, pour que
brancher un nouvel executeur (carte physique, conteneur, emulation) ne
puisse pas les contourner.

  - Le niveau 3 (emulation) ne produit pas de latence : run_benchmark
    remplace les latences par null, quelle que soit la sous-classe.
  - Le niveau 1 designe du silicium ARM64 reel : le declarer sur une autre
    architecture est une erreur, pas un avertissement.
"""

from abc import ABC, abstractmethod

DEVELOPMENT = "développement"
LEVELS = (1, 2, 3, DEVELOPMENT)

# Portee des latences par niveau, reprise du tableau de CLAUDE.md.
LATENCY_SCOPE = {
    1: "absolue : silicium ARM64 reel",
    2: "relative entre variantes uniquement : ressources plafonnees",
    3: "aucune : emulation, seule la reussite fonctionnelle compte",
    DEVELOPMENT: "relative : poste de developpement, pas une cible",
}

ARM64_MACHINES = ("aarch64", "arm64")


def parse_level(value):
    """Convertit la valeur de --level (texte) en niveau valide."""
    text = str(value).strip().lower()
    if text in ("1", "2", "3"):
        return int(text)
    if text in ("développement", "developpement", "dev"):
        return DEVELOPMENT
    raise ValueError(f"Niveau inconnu : {value!r} (attendu : 1, 2, 3 ou developpement).")


class TargetRunner(ABC):
    """Executeur de cible. Les sous-classes implementent _describe_target et _measure."""

    name = None

    def __init__(self, level):
        if level not in LEVELS:
            raise ValueError(f"Niveau invalide : {level!r} (attendu : {LEVELS}).")
        self.level = level

    @property
    def latency_valid(self):
        # Decide par le niveau, jamais par la sous-classe (regle 3).
        return self.level != 3

    @abstractmethod
    def _describe_target(self):
        """Machine mesuree : dict avec architecture, cpu_model, int8_dot_product."""

    @abstractmethod
    def _measure(self, model, runs, warmup, threads):
        """Mesure brute : dict avec latency_ms (median, mean, p95, min, max) et peak_memory_mb."""

    def describe(self):
        target = self._describe_target()
        return {
            "runner": self.name,
            "level": self.level,
            "architecture": target["architecture"],
            "cpu_model": target["cpu_model"],
            "int8_dot_product": target["int8_dot_product"],
            "latency_valid": self.latency_valid,
            "latency_scope": LATENCY_SCOPE[self.level],
        }

    def run_benchmark(self, model, runs, warmup, threads):
        description = self.describe()
        if self.level == 1 and description["architecture"].lower() not in ARM64_MACHINES:
            raise ValueError(
                f"Niveau 1 declare sur l'architecture {description['architecture']} : "
                "le niveau 1 est reserve au silicium ARM64 reel."
            )

        result = self._measure(model, runs, warmup, threads)

        if not self.latency_valid:
            result["latency_ms"] = None
            result["latency_note"] = "Latence non rapportee : " + LATENCY_SCOPE[self.level]
        return result
