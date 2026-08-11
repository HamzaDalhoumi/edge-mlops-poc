# Guide de démarrage — Semaine 1

**Objectif** : produire, en autonomie totale et sans aucune ressource de l'entreprise, les livrables de la semaine 1 — environnement opérationnel, modèle de référence, première chaîne d'optimisation mesurée, et note de cadrage.

## Principe directeur

Rien dans la semaine 1 ne dépend de TELNET. Tu as besoin de trois choses seulement : ton ordinateur, un compte GitHub, et une connexion. La cible ARM passe par les runners ARM64 gratuits de GitHub Actions — du silicium ARM réel, sans matériel ni budget cloud.

Ce point mérite d'être assumé explicitement dans ta note de cadrage : le socle est construit sur des données et des modèles publics, donc entièrement reproductible et communicable. Si un accès aux données internes s'ouvre plus tard, il se branche sans refonte. Si jamais il ne s'ouvre pas, ton travail reste complet et démontrable.

---

## Jour 1 — Environnement et dépôt

**Durée : 2 à 3 h**

### 1.1 Créer le dépôt

Crée un dépôt **public** sur GitHub (nom suggéré : `edge-mlops-poc`). Le caractère public est ici fonctionnel, pas cosmétique : les runners ARM64 ne sont gratuits que sur les dépôts publics. C'est aussi ce qui rend ton travail montrable à un recruteur.

### 1.2 Installer l'environnement local

```bash
git clone https://github.com/<ton-compte>/edge-mlops-poc.git
cd edge-mlops-poc

python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

pip install onnx onnxruntime numpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

La version CPU de PyTorch suffit et pèse bien moins lourd que la version CUDA. Tu n'entraînes rien de lourd : tu pars d'un modèle pré-entraîné.

### 1.3 Poser la structure

```
edge-mlops-poc/
├── src/
│   ├── export_onnx.py
│   ├── quantize.py
│   ├── benchmark.py
│   └── check_budget.py
├── models/
├── results/
├── docs/
├── .github/workflows/bench.yml
├── budget.json
├── requirements.txt
└── README.md
```

Copie les fichiers du kit fourni dans cette arborescence, puis fais un premier commit. Commite chaque jour : l'historique Git est une preuve d'avancement que tu pourras montrer à ton encadrant.

---

## Jour 2 — Modèle de référence et export ONNX

**Durée : 3 à 4 h**

### 2.1 Choisir le modèle

Prends **MobileNetV2** (poids ImageNet, torchvision). Trois raisons : l'architecture a été conçue pour les cibles contraintes, donc elle est pertinente pour le sujet ; les poids pré-entraînés évitent toute phase d'entraînement ; et c'est une référence connue, ce qui rend tes chiffres comparables à la littérature.

```bash
python src/export_onnx.py --arch mobilenet_v2 --output models/model.onnx
```

Attends-toi à environ 14 Mo et 3,5 millions de paramètres.

### 2.2 Comprendre ce que fait ONNX

ONNX est un format de représentation intermédiaire : il décrit le graphe de calcul du modèle indépendamment du framework qui l'a produit. C'est le pivot de toute la chaîne — c'est parce que le modèle est en ONNX qu'on peut ensuite le quantifier, le compiler et l'exécuter sur des architectures différentes sans retoucher le code d'entraînement.

### 2.3 Première mesure de référence

```bash
python src/benchmark.py --model models/model.onnx --runs 50 --threads 1 --out results/x86_fp32.json
```

Cette mesure sur ton x86 est ta **baseline**. Toutes les optimisations suivantes se compareront à elle.

---

## Jour 3 — Chaîne de quantification

**Durée : 4 à 5 h**

### 3.1 Quantification dynamique

```bash
python src/quantize.py --input models/model.onnx --output models/model_int8.onnx
python src/benchmark.py --model models/model_int8.onnx --runs 50 --out results/x86_int8.json
```

Tu devrais observer une réduction de taille d'environ 70 à 75 %. La quantification convertit les poids de flottants 32 bits en entiers 8 bits : quatre fois moins de mémoire, et des opérations entières souvent plus rapides sur les cœurs embarqués.

### 3.2 Mesurer la perte de précision

C'est l'étape que la plupart des étudiants sautent, et c'est précisément celle qui fait la valeur du travail. Une réduction de taille sans mesure de la précision conservée ne prouve rien.

Télécharge quelques centaines d'images d'un jeu public (Imagenette, sous-ensemble d'ImageNet, ou CIFAR-10 redimensionné), évalue le modèle FP32 puis le modèle INT8 sur le même sous-ensemble, et consigne l'écart de précision top-1.

### 3.3 Consigner le compromis

Construis ton premier tableau de résultats :

| Variante | Taille | Latence p95 | Précision top-1 | Écart |
|----------|--------|-------------|-----------------|-------|
| FP32     | …      | …           | …               | réf.  |
| INT8 dyn.| …      | …           | …               | …     |

Ce tableau est le cœur de ton rapport final. Commence-le dès maintenant.

---

## Jour 4 — Profils de ressources contraints

**Durée : 3 h**

Un modèle mesuré sur ton portable à pleine puissance ne dit rien de son comportement sur un équipement contraint. Docker permet de plafonner CPU et mémoire, ce qui te donne des classes d'équipement reproductibles.

```bash
docker build -f Dockerfile.bench -t bench:local .

# Profil « équipement bas de gamme »
docker run --rm --cpus=0.5 --memory=256m bench:local --model models/model_int8.onnx --runs 50

# Profil « carte type Raspberry Pi »
docker run --rm --cpus=1.0 --memory=512m bench:local --model models/model_int8.onnx --runs 50

# Profil « carte type Jetson »
docker run --rm --cpus=2.0 --memory=2g bench:local --model models/model_int8.onnx --runs 50
```

Définis ces trois profils dans un fichier de configuration versionné. C'est ce qui constitue le **niveau 2** de la stratégie de validation décrite dans ta proposition de sujet.

Précision méthodologique à écrire noir sur blanc : ces profils contraignent les ressources, ils ne reproduisent pas le jeu d'instructions ni la hiérarchie de cache d'un vrai SoC embarqué. Ils servent à **comparer des variantes entre elles**, pas à prédire une latence absolue sur cible réelle.

---

## Jour 5 — Cible ARM réelle et intégration continue

**Durée : 4 h**

### 5.1 Activer le pipeline

Le fichier `.github/workflows/bench.yml` du kit exécute la chaîne complète sur deux cibles en parallèle : `ubuntu-24.04` (x86_64) et `ubuntu-24.04-arm` (ARM64). Pousse-le et observe le résultat dans l'onglet Actions.

```bash
git add .github/workflows/bench.yml
git commit -m "CI: benchmark multi-architecture x86_64 et ARM64"
git push
```

Tu obtiens alors des mesures sur du **vrai ARM**, sans posséder aucune carte. C'est le niveau 1 de ta stratégie de validation.

### 5.2 Émulation locale en complément

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx build --platform linux/arm64 -f Dockerfile.bench -t bench:arm64 --load .
docker run --rm --platform linux/arm64 bench:arm64 --model models/model_int8.onnx --runs 20
```

Utile pour valider la portabilité des artefacts hors ligne. **Les temps mesurés sous émulation ne sont pas exploitables** — dis-le explicitement partout où ces chiffres apparaissent.

### 5.3 Activer la barrière

```bash
python src/check_budget.py --report results/arm64_int8.json --budget budget.json
```

Le script sort en code 1 si un seuil est dépassé, ce qui fait échouer le pipeline. C'est ce mécanisme qui transforme une simple mesure en garde-fou qualité — et c'est l'argument central de ton sujet auprès de l'encadrant.

Calibre les seuils de `budget.json` à partir de tes premières mesures ARM, avec une marge d'environ 20 %.

---

## Jour 6 — Note de cadrage et état de l'art

**Durée : 4 à 5 h**

Rédige un document de 4 à 6 pages, à envoyer à ton encadrant en fin de semaine. C'est ton premier livrable formel et il vaut autant que le code.

Plan suggéré :

1. **Rappel du sujet et du périmètre** — reprends la proposition validée.
2. **État de l'art** — MLOps appliqué à l'Edge AI, techniques de compression (quantification, élagage, distillation), formats et runtimes (ONNX Runtime, TensorFlow Lite, TensorRT). Vise une dizaine de références sérieuses, pas quarante liens de blog.
3. **Stratégie de validation retenue** — les trois niveaux, avec leurs limites explicitement énoncées.
4. **Cas d'usage retenu et justification.**
5. **Premiers résultats** — le tableau du jour 3 et les mesures ARM du jour 5. Avoir des chiffres réels dès la première semaine change complètement la perception de ton travail.
6. **Risques identifiés et arbitrages demandés.**

---

## Ce que tu auras à la fin de la semaine

- Un dépôt public versionné, avec un historique de commits quotidiens
- Un modèle de référence exporté en ONNX
- Une chaîne de quantification automatisée et mesurée
- Trois profils de ressources contraints reproductibles
- Un pipeline d'intégration continue exécutant le banc sur x86_64 **et** sur ARM64 réel
- Une barrière de rejet automatique fonctionnelle
- Une note de cadrage avec des résultats chiffrés

C'est un livrable de semaine 1 solide — et obtenu sans dépendre d'une seule ressource de l'entreprise.

---

## Deux conseils

**Envoie la note de cadrage à ton encadrant en fin de semaine, même s'il ne l'a pas demandée.** Un stagiaire à distance qui produit un livrable écrit dès la première semaine s'installe immédiatement dans une autre catégorie. Termine par deux ou trois questions précises : c'est ce qui ouvre le dialogue et te donne de la matière pour la suite.

**Documente les limites au fur et à mesure, pas à la fin.** Chaque fois que tu écris un chiffre, écris à côté dans quelles conditions il a été obtenu. Le jour de la soutenance, la question qui tombera sera « sur quoi avez-vous mesuré ? » — et tu auras la réponse écrite depuis six semaines.
