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

## État d'avancement

Dernière mise à jour : 2026-09-28. Un seul commit existe dans l'historique
(`552de7f`, structure initiale du 11 août) et aucun remote Git n'est
configuré (`git remote -v` vide) — le dépôt n'a donc jamais été poussé.
Le constat ci-dessous porte sur l'état réel du disque, pas seulement sur ce
qui est commité.

### Fait

- Export ONNX (`src/export_onnx.py`) : MobileNetV2 pré-entraîné, 3 504 872
  paramètres, produit `models/model.onnx` (13.34 Mo) — exécuté et vérifié.
- Quantification dynamique **et** statique (`src/quantize.py`) : les deux
  modes tournent, `model_int8.onnx` et `model_int8_static.onnx` générés.
- Banc de mesure (`src/benchmark.py`) : taille, latence (médiane/p95), pic
  mémoire, avec un bloc `environment` (machine, OS, threads, version ORT) —
  chaque chiffre est accompagné de ses conditions d'exécution, conformément
  à la règle 2.
- Barrière de budget (`src/check_budget.py`) : sort en code 1 si un seuil
  est dépassé ; testée dans les deux sens (OK et REJET) sur des rapports
  réels.
- Niveau 2 de la stratégie de validation exercé : benchmarks obtenus dans un
  conteneur `--cpus=1.0 --memory=512m` (`results/fp32_docker.json`,
  `results/int8_docker.json`, non commités — conforme à la convention).
- Pipeline CI (`.github/workflows/bench.yml`) écrit, avec matrice x86_64 +
  `ubuntu-24.04-arm` (niveau 1).

### Partiel

- **Niveau 1 (ARM64 réel) jamais exécuté** : le workflow existe mais,
  faute de remote configuré, n'a jamais tourné sur GitHub Actions — aucune
  mesure ARM64 réelle n'existe encore dans le dépôt.
- **`budget.json` non calibré** : les seuils actuels (5.0 Mo / 50.0 ms /
  300.0 Mo) sont des valeurs de placeholder. Les deux variantes mesurées à
  ce jour (FP32 et INT8 dynamique) échouent la barrière.
- **Export fragile face aux versions récentes de torch** : `torch` et
  `torchvision` sont commentés (non épinglés) dans `requirements.txt`. Avec
  une version récente (exporteur "dynamo" par défaut), l'export produit deux
  fichiers (`model.onnx` + `.data`) et `os.path.getsize()` ne mesure alors
  que le squelette (~0.25 Mo au lieu de la taille réelle) — un modèle hors
  budget peut passer la barrière sans être détecté. Constat détaillé dans
  l'audit du 11 août (non commité à ce jour).
- **Quantification statique jamais testée en conditions réelles** :
  calibration faite sur `data/calib.npy`, un jeu synthétique, pas de données
  de calibration réelles documentées dans le dépôt.
- **Historique Git incomplet** : un seul commit ; le travail réalisé depuis
  (modèles générés, benchmarks Docker, audit de la chaîne) n'est pas
  versionné, ce qui va à l'encontre de la convention « un commit par étape
  logique ».

### À faire

- **Mesure de précision top-1 FP32 vs INT8** : aucun script ne l'implémente
  (pas de `src/evaluate.py`). C'est le chiffre central attendu dans toute
  comparaison de variantes, absent de la chaîne actuelle — alors que le
  tableau de la stratégie de validation annonce la précision comme
  exploitable au niveau 1.
- **Abstraction d'« exécuteur de cible »** : annoncée ci-dessus comme
  principe de conception central, mais `benchmark.py` appelle directement
  `onnxruntime.InferenceSession`, sans classe `TargetRunner` /
  `LocalRunner` / `ContainerRunner` / `EmulatedRunner`. Rien dans le code
  ne force aujourd'hui un exécuteur « émulé » à marquer sa latence comme
  non exploitable (règle 3) — c'est une discipline humaine, pas une
  garantie du code.
- `onnx.checker.check_model()` après export et après quantification — non
  appelé ; aurait pu détecter plus tôt le problème d'export décrit
  ci-dessus.
- Format de quantification statique : `quant_format=None` fixé en dur dans
  `quantize.py`, alors qu'ONNX Runtime recommande lui-même `QDQ` pour de
  bonnes performances x64 (avertissement émis à l'exécution, actuellement
  ignoré).
- Note de cadrage écrite (livrable prévu pour la fin de la première
  semaine).

## Ce que je dois pouvoir défendre

Ce travail est présenté devant un jury. Je dois pouvoir expliquer chaque ligne.
Quand tu produis du code non trivial, explique le raisonnement — pas seulement
le résultat. Si tu choisis entre plusieurs approches, dis pourquoi.
