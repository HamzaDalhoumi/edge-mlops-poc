"""
Critere de rejet automatique.

Compare un rapport de benchmark a un budget declare et sort en code 1
si un seuil est depasse. C'est cette brique qui transforme une mesure
en garde-fou : integree au pipeline d'integration continue, elle empeche
la promotion d'un modele hors budget.

Usage :
    python src/check_budget.py --report results/int8.json --budget budget.json
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True,
                        help="Rapport de benchmark (JSON produit par benchmark.py).")
    parser.add_argument("--budget", required=True)
    parser.add_argument("--accuracy", default=None,
                        help="Rapport de precision du modele evalue (JSON produit par evaluate.py).")
    parser.add_argument("--accuracy-reference", default=None,
                        help="Rapport de precision du modele FP32 de reference, requis avec "
                             "--accuracy pour calculer l'ecart de precision en points.")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)
    with open(args.budget) as f:
        budget = json.load(f)

    checks = []

    if "max_size_mb" in budget:
        checks.append((
            "Taille du modele",
            report["size_mb"],
            budget["max_size_mb"],
            "Mo",
        ))
    if "max_latency_p95_ms" in budget:
        checks.append((
            "Latence p95",
            report["latency_ms"]["p95"],
            budget["max_latency_p95_ms"],
            "ms",
        ))
    if "max_peak_memory_mb" in budget:
        checks.append((
            "Pic memoire",
            report["peak_memory_mb"],
            budget["max_peak_memory_mb"],
            "Mo",
        ))
    if "max_accuracy_drop_pts" in budget:
        if not args.accuracy:
            raise SystemExit("budget.json fixe max_accuracy_drop_pts : --accuracy est requis.")
        if not args.accuracy_reference:
            raise SystemExit("--accuracy requiert --accuracy-reference pour calculer l'ecart en points.")
        with open(args.accuracy) as f:
            accuracy = json.load(f)
        with open(args.accuracy_reference) as f:
            accuracy_reference = json.load(f)
        drop = accuracy_reference["top1_accuracy"] - accuracy["top1_accuracy"]
        checks.append((
            "Perte de precision",
            drop,
            budget["max_accuracy_drop_pts"],
            "pts",
        ))

    failures = 0
    print(f"Verification du budget pour : {report['model']}")
    print(f"Cible mesuree              : {report['environment']['machine']}")
    print("-" * 62)

    for label, measured, limit, unit in checks:
        ok = measured <= limit
        status = "OK    " if ok else "REJET "
        print(f"{status} {label:<22} {measured:>9.3f} {unit:<3} (max {limit} {unit})")
        if not ok:
            failures += 1

    print("-" * 62)

    if failures:
        print(f"{failures} critere(s) non respecte(s) : le modele est rejete.")
        sys.exit(1)

    print("Tous les criteres sont respectes : le modele peut etre promu.")


if __name__ == "__main__":
    main()
