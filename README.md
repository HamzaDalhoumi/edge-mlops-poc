# Edge MLOps — Preuve de concept

Chaîne automatisée d'optimisation et de validation de modèles d'IA
pour cibles ARM contraintes.

## Démarrage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/export_onnx.py --arch mobilenet_v2 --output models/model.onnx
python src/repair_onnx.py --input models/model.onnx --output models/model_clean.onnx
python src/quantize.py  --input models/model_clean.onnx --output models/model_int8.onnx
python src/benchmark.py --model models/model_int8.onnx --out results/int8.json
python src/check_budget.py --report results/int8.json --budget budget.json
```

## Stratégie de validation

| Niveau | Moyen | Portée |
|---|---|---|
| 1 | Runner ARM64 (GitHub Actions) | Silicium ARM réel — latence et mémoire exploitables |
| 2 | Conteneur à ressources plafonnées | Comparaison de variantes sous budget contraint |
| 3 | Émulation multi-architecture | Portabilité fonctionnelle — temps non exploitables |

Le banc de mesure est conçu derrière une abstraction d'exécuteur de cible :
brancher une carte physique ne demande qu'une implémentation supplémentaire.
