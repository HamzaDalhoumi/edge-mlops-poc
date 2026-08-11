# Prompts Claude Code — Semaine 1

Un prompt par session. Ne les enchaîne pas dans un même contexte : chaque étape
produit un résultat à vérifier avant de passer à la suivante.

Avant tout : place `CLAUDE.md` à la racine du dépôt. Claude Code le lit
automatiquement et n'aura pas besoin qu'on lui réexplique le contexte à chaque
fois.

---

## Prompt 0 — Initialisation

```
Initialise le dépôt de ce projet selon la structure décrite dans CLAUDE.md.

Crée :
- l'arborescence src/ models/ results/ docs/ .github/workflows/
- un .gitignore excluant .venv/, __pycache__/, results/, data/, *.onnx
- un requirements.txt avec onnx, onnxruntime, numpy (versions minimales, pas épinglées)
- un README.md court : objectif du projet, commandes de démarrage, tableau des
  trois niveaux de validation

Ne crée pas encore les scripts Python. Termine par un commit initial.
```

---

## Prompt 1 — Export ONNX

```
Écris src/export_onnx.py : export d'un modèle torchvision pré-entraîné vers ONNX.

Exigences :
- argparse : --arch (mobilenet_v2 par défaut, plus mobilenet_v3_small, resnet18,
  squeezenet1_1), --output, --opset (13 par défaut), --size (224 par défaut)
- axe batch dynamique
- affiche en fin d'exécution : architecture, nombre de paramètres, taille du
  fichier ONNX en Mo

Ensuite : exécute le script avec les valeurs par défaut et montre-moi la sortie
réelle. Vérifie que le fichier produit passe onnx.checker.check_model.
```

---

## Prompt 2 — Quantification

```
Écris src/quantize.py : quantification INT8 d'un modèle ONNX via
onnxruntime.quantization.

Exigences :
- deux modes : dynamique (par défaut, sans données) et statique (avec un
  CalibrationDataReader lisant un fichier .npy)
- en mode statique, détecte automatiquement le nom du tenseur d'entrée si
  --input-name n'est pas fourni
- affiche taille avant, taille après, pourcentage de réduction

Ensuite : exécute-le en mode dynamique sur models/model.onnx et montre-moi la
sortie réelle. Vérifie que le modèle quantifié se charge bien dans une
InferenceSession et produit une sortie de forme correcte.
```

---

## Prompt 3 — Banc de mesure

```
Écris src/benchmark.py : banc de mesure d'un modèle ONNX.

Exigences :
- argparse : --model, --runs (50), --warmup (5), --threads (1), --out
- génère automatiquement des entrées aléatoires conformes à la signature du
  modèle (lis session.get_inputs(), gère les dimensions dynamiques)
- phase de préchauffage avant mesure
- rapporte : taille du fichier, latence médiane / moyenne / p95 / min / max,
  pic de mémoire résident
- rapporte aussi l'environnement d'exécution : platform.machine(), système,
  version d'onnxruntime, nombre de CPU visibles
- sortie JSON sur stdout, et dans --out si fourni

Le bloc environnement n'est pas décoratif : sans lui un chiffre de latence n'a
aucune valeur. Traite-le comme obligatoire.

Ensuite : exécute-le sur le modèle FP32 puis sur le modèle INT8, et présente-moi
les deux résultats réels côte à côte.
```

---

## Prompt 4 — Mesure de la précision

```
Écris src/evaluate.py : mesure de la précision top-1 d'un modèle ONNX sur un
jeu de validation public.

Exigences :
- utilise Imagenette (sous-ensemble public d'ImageNet) ou un équivalent
  téléchargeable sans authentification
- prétraitement conforme à celui attendu par les modèles torchvision
  (redimensionnement, recadrage centré, normalisation ImageNet)
- --limit pour restreindre le nombre d'images évaluées
- sortie JSON : précision top-1, nombre d'images, nom du modèle

Ensuite : évalue le modèle FP32 et le modèle INT8 sur le même sous-ensemble
(500 images suffisent), et construis-moi un tableau markdown comparant taille,
latence p95 et précision top-1, avec l'écart de précision en points.

C'est ce tableau qui justifie tout le projet : une réduction de taille sans
mesure de la précision perdue ne prouve rien. Ne remplis aucune case avec une
valeur estimée.
```

---

## Prompt 5 — Barrière de rejet

```
Écris src/check_budget.py : comparaison d'un rapport de benchmark à un budget
déclaré.

Exigences :
- lit un rapport JSON et un budget JSON
- vérifie les seuils présents dans le budget : max_size_mb,
  max_latency_p95_ms, max_peak_memory_mb (tous facultatifs)
- affiche un tableau lisible ligne par ligne avec OK ou REJET
- sort en code 1 si au moins un seuil est dépassé, 0 sinon

Crée aussi budget.json avec des seuils calibrés à partir des mesures déjà
obtenues, avec environ 20 % de marge.

Ensuite : démontre-moi que la barrière fonctionne dans les deux sens — un cas
qui passe et un cas qui échoue avec le bon code de sortie.
```

---

## Prompt 6 — Profils de ressources contraints

```
Mets en place le niveau 2 de la stratégie de validation : exécution du banc
sous ressources plafonnées.

Exigences :
- Dockerfile.bench : image multi-architecture basée sur python:3.11-slim,
  entrypoint sur src/benchmark.py
- profiles.json définissant trois profils avec un quota CPU et un plafond
  mémoire : low_end, raspberry_class, jetson_class
- scripts/run_profiles.sh : construit l'image, exécute le banc sous chaque
  profil, écrit un rapport par profil dans results/

Ensuite : exécute les trois profils et présente-moi un tableau comparatif des
latences réelles obtenues.

Ajoute dans le README une note méthodologique : ces profils contraignent les
ressources mais ne reproduisent ni le jeu d'instructions ni la hiérarchie de
cache d'un vrai SoC embarqué. Ils servent à comparer des variantes entre elles,
pas à prédire une latence absolue sur cible réelle.
```

---

## Prompt 7 — Pipeline d'intégration continue

```
Écris .github/workflows/bench.yml : pipeline exécutant la chaîne complète sur
deux architectures.

Exigences :
- matrice de deux runners : ubuntu-24.04 (x86_64) et ubuntu-24.04-arm (ARM64)
- fail-fast désactivé, pour que l'échec d'une architecture n'annule pas l'autre
- étapes : checkout, setup-python 3.11, installation, quantification,
  benchmark FP32, benchmark INT8, vérification du budget
- publication des rapports JSON en artefacts, y compris en cas d'échec
- déclenchement sur push vers main, pull request, et manuellement

Le runner ARM64 est du silicium ARM réel et reste gratuit sur les dépôts
publics : c'est notre niveau 1 de validation, il ne doit pas être remplacé par
de l'émulation.

Ensuite : explique-moi ce que fait chaque étape et ce que je dois vérifier dans
l'onglet Actions après le premier push.
```

---

## Prompt 8 — Abstraction d'exécuteur de cible

```
Refactorise le banc de mesure derrière une abstraction d'exécuteur de cible,
comme décrit dans CLAUDE.md.

Exigences :
- une classe de base TargetRunner définissant l'interface d'exécution d'un
  benchmark et la description de la cible
- trois implémentations : LocalRunner (processus local), ContainerRunner
  (conteneur à ressources plafonnées), EmulatedRunner (plateforme émulée)
- chaque exécuteur rapporte son niveau de validation et la portée de ses
  mesures, de sorte que EmulatedRunner marque explicitement ses latences comme
  non exploitables
- src/benchmark.py conserve son interface en ligne de commande actuelle

Objectif : ajouter plus tard un DeviceRunner pour une carte physique ne doit
demander qu'une classe supplémentaire, sans modifier le reste.

Vérifie qu'aucune commande existante ne casse. Explique-moi les choix de
conception avant de coder.
```

---

## Prompt 9 — Note de cadrage

```
À partir de l'état actuel du dépôt et des résultats réels présents dans
results/, rédige docs/note-de-cadrage.md.

Plan :
1. Rappel du sujet et du périmètre
2. État de l'art : MLOps appliqué à l'Edge AI, techniques de compression,
   formats et runtimes
3. Stratégie de validation à trois niveaux, avec leurs limites explicites
4. Premiers résultats — uniquement les chiffres présents dans results/
5. Risques identifiés et arbitrages à demander à l'encadrant

Contraintes :
- 4 à 6 pages
- aucun chiffre qui ne provienne d'un fichier de results/
- pour la section 2, indique clairement quelles références tu affirmes et
  lesquelles je dois vérifier moi-même avant de les citer
- termine par trois questions précises à poser à l'encadrant
```

---

## Bonnes pratiques d'usage

**Une tâche par session.** Un contexte encombré dégrade la qualité et t'empêche
de vérifier ce qui a été fait.

**Exige l'exécution, pas la promesse.** Chaque prompt ci-dessus se termine par
« exécute et montre-moi la sortie réelle ». C'est délibéré : un modèle de
langage produit très volontiers un tableau de benchmarks plausible sans avoir
rien exécuté. Sur ce projet précis, un chiffre inventé qui se retrouve dans ton
rapport est une faute qui te suivra jusqu'à la soutenance.

**Commite après chaque étape validée.** Tu pourras revenir en arrière, et
l'historique Git est ta preuve d'avancement auprès de ton encadrant.

**Demande les explications avant le code sur les parties non triviales.** Les
prompts 8 et 9 le font explicitement. Tu dois pouvoir défendre chaque choix
devant un jury — du code que tu ne peux pas expliquer est un risque, pas un
gain de temps.

**Relis le diff.** Toujours. C'est ton projet, ton rapport, ta soutenance.
