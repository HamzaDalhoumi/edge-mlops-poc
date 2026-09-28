# Audit de la chaîne — exécution réelle et constats

Date : 2026-08-11
Contexte : vérification de bout en bout de la chaîne existante (`src/export_onnx.py`,
`src/quantize.py`, `src/benchmark.py`, `src/check_budget.py`), sans aucune
réécriture de code. Tous les chiffres ci-dessous proviennent de commandes
réellement exécutées, pas d'estimations.

---

## 1. Rôle de chaque script

**`src/export_onnx.py`**
Charge MobileNetV2 pré-entraîné (torchvision, poids ImageNet), le passe en mode
`eval()`, et l'exporte via `torch.onnx.export` avec un axe batch dynamique
(`dynamic_axes`). Point clé à défendre : c'est le pivot de toute la chaîne — une
fois en ONNX, le modèle n'est plus lié à PyTorch, ce qui permet de le quantifier
et de le benchmarker avec un runtime indépendant (`onnxruntime`).

**`src/quantize.py`**
Convertit les poids FP32 en INT8 via `onnxruntime.quantization`. Deux modes :
`dynamic` (pas de données, quantifie seulement les poids, calibre les
activations à l'exécution) et `static` (nécessite un `CalibrationDataReader`
sur un `.npy`, quantifie aussi les activations — gain plus fort mais risque de
perte de précision plus élevé). Point clé : le `NpyCalibrationReader`
implémente l'interface `CalibrationDataReader` attendue par `quantize_static`,
en présentant les échantillons un par un.

**`src/benchmark.py`**
Charge le modèle dans une `InferenceSession` ONNX Runtime CPU, génère des
entrées aléatoires conformes à la signature du modèle, fait un warmup puis
mesure la latence sur N itérations (médiane/p95/etc.) et le pic de mémoire
résidente via `resource.getrusage`. Point clé à défendre : le bloc
`environment` (machine, OS, version d'ORT, nb de CPU) est ce qui rend un
chiffre de latence interprétable — sans lui, un nombre seul ne vaut rien,
exactement l'esprit de CLAUDE.md.

**`src/check_budget.py`**
Compare un rapport JSON de benchmark à des seuils déclarés dans `budget.json`,
affiche OK/REJET ligne par ligne, sort en code 1 si un seuil est dépassé.
C'est la brique qui transforme une mesure en garde-fou CI : c'est elle qui
ferait échouer un job GitHub Actions si un modèle dépasse son budget.

---

## 2. Exécution réelle de bout en bout

Environnement : venv local (Python 3.13 au moment du test, packages
fraîchement installés), Docker Desktop pour le benchmark. Le module `resource`
utilisé par `benchmark.py` **n'existe pas sous Windows** — confirmé par
`import resource` → `ModuleNotFoundError`. Le niveau 1/2 de la stratégie de
validation impose donc de passer par Linux (conteneur ou CI), ce qui a été
fait ici via `Dockerfile.bench`.

### Export

```
Architecture  : mobilenet_v2
Parametres    : 3,504,872
Taille ONNX   : 13.34 Mo  (vraie taille, single-file — voir §3.A)
```

### Quantification dynamique

```
Mode          : dynamic
Avant         : 13.34 Mo
Apres         : 3.52 Mo
Reduction     : 73.6 %
```

### Benchmark

Exécuté dans le conteneur `Dockerfile.bench` (`--cpus=1.0 --memory=512m`,
50 itérations, 1 thread intra-op — niveau 2 de la stratégie de validation) :

| Variante | Taille | Latence p95 | Pic mémoire | Machine |
|---|---|---|---|---|
| FP32 | 13.344 Mo | 11.68 ms | 89.1 Mo | x86_64 / Linux, ORT 1.28.0 |
| INT8 dynamique | 3.521 Mo | 70.50 ms | 85.6 Mo | x86_64 / Linux, ORT 1.28.0 |

Résultat mesuré deux fois de suite, cohérent (~66-70 ms p95 pour l'INT8 à
chaque run).

### Barrière de budget

Avec le `budget.json` actuel :

```
FP32 : REJET  Taille du modele  13.344 Mo (max 5.0 Mo)     → exit 1
INT8 : REJET  Latence p95        70.495 ms (max 50.0 ms)   → exit 1
```

Les deux sens (OK/REJET) sont bien exercés et le code de sortie est correct
dans les deux cas — le mécanisme de garde-fou fonctionne tel quel.

---

## 3. Faiblesses et manques constatés

### A. L'export casse silencieusement avec les versions actuelles de torch

`requirements.txt` laisse torch/torchvision non épinglés (et commentés). En
installant les dernières versions disponibles (torch 2.13, seule branche
compatible avec Python 3.13), `torch.onnx.export` bascule par défaut sur le
nouvel exporteur "dynamo" :

- Il produit deux fichiers (`model.onnx` + `model.onnx.data`, poids
  externalisés). `os.path.getsize(args.output)` dans `export_onnx.py` — et le
  même calcul dans `benchmark.py` pour `size_mb` — ne mesure alors que le
  petit fichier squelette (0.25 Mo au lieu de 13.3 Mo réels).
  **Ce chiffre remonte tel quel dans `check_budget.py` et ferait passer un
  modèle hors budget comme "OK".** C'est le point le plus grave : un
  garde-fou censé bloquer peut être trompé silencieusement par un simple
  changement de version de dépendance.
- Le graphe produit fait aussi échouer `onnxruntime.quantization`
  (`InferenceError: shape inference`), reproductible même avec `--opset 18`
  et même après le pré-traitement recommandé par ORT lui-même
  (`onnxruntime.quantization.preprocess`). Il a fallu forcer `dynamo=False`
  (en diagnostic, hors script) pour obtenir un modèle exploitable.
- Cause racine : venv en Python 3.13 alors que CLAUDE.md fixe la convention à
  3.11, combiné à l'absence de version épinglée pour torch.
- Fix possible : épingler torch (`<2.6` ou passer explicitement
  `dynamo=False` dans `export_onnx.py`) et aligner le venv sur Python 3.11.

### B. `budget.json` n'est pas calibré

Aucune des deux variantes ne passe actuellement la barrière (FP32 rejeté sur
la taille, INT8 rejeté sur la latence) — alors que GUIDE-SEMAINE-1.md prescrit
de calibrer les seuils à partir de vraies mesures avec ~20 % de marge. Ce sont
des valeurs de placeholder, pas encore des seuils réels.

### C. La quantification dynamique ralentit l'inférence au lieu de l'accélérer

70 ms p95 contre 11.7 ms en FP32, mesuré deux fois, cohérent. Ce n'est pas un
bug de script — c'est un comportement connu d'ONNX Runtime CPU avec le format
`QOperator` sur des architectures à convolutions depthwise (MobileNetV2). Le
mode statique de `quantize.py` émet d'ailleurs lui-même l'avertissement
`"Please use QuantFormat.QDQ ... Or it will lead to bad performance on x64"`
— mais `quantize.py` fixe `quant_format=None` en dur, sans option CLI pour
passer en QDQ. À signaler explicitement dans le rapport : la quantification
réduit la taille mais dégrade ici la latence, résultat contre-intuitif mais
réel et mesuré.

### D. Aucune mesure de précision

Il n'existe pas de `src/evaluate.py` — la perte de précision top-1 entre FP32
et INT8 n'est mesurée nulle part dans la chaîne actuelle. C'est exactement
l'étape que PROMPTS-CLAUDE-CODE.md désigne comme "celle que la plupart des
étudiants sautent" et "le cœur du rapport final". Sans elle, la réduction de
taille/latence ne prouve rien devant un jury.

### E. L'abstraction "exécuteur de cible" n'existe pas encore dans le code

CLAUDE.md la présente comme le "principe de conception central", mais
`benchmark.py` appelle directement `onnxruntime.InferenceSession` sans classe
`TargetRunner` / `LocalRunner` / `ContainerRunner` / `EmulatedRunner`. Rien
dans le code ne force par exemple un `EmulatedRunner` à marquer sa latence
comme non exploitable — c'est aujourd'hui une discipline humaine, pas une
garantie du code.

### F. Pas de `onnx.checker.check_model()`

Ni `export_onnx.py` ni `quantize.py` ne valident le modèle produit avant de le
faire suivre à l'étape suivante — alors que c'était explicitement demandé dans
le prompt d'origine (PROMPTS-CLAUDE-CODE.md, prompt 1). Aurait pu détecter le
problème d'export (§A) plus tôt.

### G. Mode statique jamais testé en conditions réelles

Il a fallu générer des données de calibration synthétiques (bruit aléatoire)
pour le faire tourner en diagnostic — il n'y a ni jeu de calibration réel, ni
test, documenté dans le dépôt.

### H. `peak_memory_mb` mesure tout le processus Python

Pas seulement l'inférence (RSS du process entier, y compris chargement d'ORT).
Utilisable en comparaison relative entre variantes, mais pas comme chiffre
absolu isolé — à préciser si la question tombe en soutenance.

---

## Note méthodologique

Aucun fichier du dépôt n'a été modifié pour produire ces résultats.
Uniquement des scripts de diagnostic exécutés hors dépôt et des arguments CLI
déjà supportés par les scripts existants (`--opset`).
