# CLAUDE.md

Contexte permanent du projet. Lis ce fichier avant toute tâche.

## Le projet

Preuve de concept d'une **chaîne MLOps automatisée** pour l'optimisation et la
validation de modèles d'IA destinés à des cibles ARM contraintes. Réalisée dans
le cadre d'un stage de 6 semaines chez TELNET Holding (pôle R&D Ingénierie
Logicielle & Produit), en télétravail.

Le problème traité : le passage d'un modèle entraîné à un modèle validé pour
cible embarquée est aujourd'hui manuel, non tracé et non mesuré. On construit la
chaîne qui automatise et fiabilise ce passage.

## Contrainte structurante

**Aucun accès à du matériel physique.** Ni carte de développement, ni ressource
cloud de l'entreprise, ni budget. Tout doit tourner sur un poste de
développement et sur les runners gratuits de GitHub Actions.

La stratégie de validation est donc à trois niveaux, et cette distinction doit
apparaître partout où un chiffre est produit :

| Niveau | Moyen | Ce qui est exploitable |
|---|---|---|
| 1 | Runner `ubuntu-24.04-arm` (GitHub Actions) | Silicium ARM64 réel : latence, mémoire, précision |
| 2 | Conteneur à ressources plafonnées (`--cpus`, `--memory`) | Comparaison relative entre variantes |
| 3 | Émulation multi-architecture (buildx / QEMU) | Portabilité fonctionnelle uniquement — **jamais** les temps |

## Règles non négociables

1. **Ne jamais inventer un résultat de mesure.** Si un benchmark n'a pas été
   exécuté, ne produis pas de chiffre. Exécute la commande et rapporte la sortie
   réelle, ou dis que la mesure manque.
2. **Chaque chiffre porte ses conditions.** Architecture, nombre de threads,
   limites de ressources, nombre d'itérations. Un résultat sans contexte
   d'exécution est un résultat invalide.
3. **Le niveau 3 ne produit pas de latence.** Sous émulation, on ne rapporte que
   la réussite ou l'échec fonctionnel.
4. **Pas de dépendance payante ni de service nécessitant une carte bancaire.**
5. **Pas de données d'entreprise.** Uniquement des jeux de données et des modèles
   publics, afin que tout reste reproductible et communicable.

## Architecture du code

```
src/export_onnx.py     Export d'un modèle PyTorch pré-entraîné vers ONNX
src/quantize.py        Quantification INT8 (dynamique et statique)
src/benchmark.py       Banc de mesure : taille, latence, pic mémoire
src/check_budget.py    Barrière de rejet : compare un rapport à un budget
budget.json            Seuils de latence, taille et mémoire
Dockerfile.bench       Image multi-architecture du banc
.github/workflows/     Pipeline CI : benchmark x86_64 + ARM64
```

**Principe de conception central** : le banc de mesure est développé derrière une
abstraction d'« exécuteur de cible ». Les trois niveaux de validation en sont des
implémentations interchangeables. Brancher une carte physique plus tard ne doit
demander qu'une implémentation supplémentaire, sans toucher au reste. Respecte
cette abstraction dans tout ajout.

## Conventions

- Python 3.11, pas de framework superflu, bibliothèque standard privilégiée
- Chaque script est exécutable seul en ligne de commande avec `argparse`
- Sorties de mesure en JSON dans `results/`, jamais commitées
- Commentaires et messages utilisateur en français, code et identifiants en anglais
- Messages de commit descriptifs, un commit par étape logique

## Ce que je dois pouvoir défendre

Ce travail est présenté devant un jury. Je dois pouvoir expliquer chaque ligne.
Quand tu produis du code non trivial, explique le raisonnement — pas seulement
le résultat. Si tu choisis entre plusieurs approches, dis pourquoi.
