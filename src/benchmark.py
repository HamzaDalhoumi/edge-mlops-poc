"""
Banc de mesure d'un modele ONNX.

Mesure, pour un modele donne :
  - la taille du fichier sur disque
  - la latence d'inference (mediane, p95) sur N iterations
  - le pic de memoire resident du processus

La mesure est deleguee a un executeur de cible (src/runners/) : le banc ne
sait pas ou il mesure, l'executeur si. Le rapport contient la description
complete de l'executeur (bloc "runner"), dont le niveau de validation et
l'indicateur latency_valid.

Le resultat est ecrit en JSON, ce qui permet de le comparer
automatiquement a un budget dans le pipeline d'integration continue.

Usage :
    python src/benchmark.py --model models/model.onnx --runs 50 --out results/bench.json
    python src/benchmark.py --model models/model.onnx --level 1 --out results/arm64.json
"""

import argparse
import json
import os

import onnxruntime as ort

from hostinfo import environment
from modelutils import model_size_mb
from runners import DEVELOPMENT, RUNNERS, get_runner, parse_level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1,
                        help="Nombre de threads intra-op. 1 = comportement le plus proche d'une cible contrainte.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--runner", choices=sorted(RUNNERS), default="local",
                        help="Executeur de cible qui realise la mesure.")
    parser.add_argument("--level", type=parse_level, default=DEVELOPMENT,
                        help="Niveau de validation : 1, 2, 3 ou developpement (defaut, le plus prudent).")
    args = parser.parse_args()

    runner = get_runner(args.runner, args.level)
    measure = runner.run_benchmark(args.model, args.runs, args.warmup, args.threads)

    report = {
        "model": os.path.basename(args.model),
        "size_mb": round(model_size_mb(args.model), 3),
        "runs": args.runs,
        "threads": args.threads,
        **measure,
        "environment": environment(ort.__version__),
        "runner": runner.describe(),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n-> rapport ecrit dans {args.out}")


if __name__ == "__main__":
    main()
