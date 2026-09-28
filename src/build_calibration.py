"""
Genere un jeu de calibration pour la quantification statique, a partir
d'images reelles du dossier TRAIN d'Imagenette.

Jamais VAL : ce sous-ensemble sert a evaluate.py pour mesurer la precision.
Le reutiliser pour la calibration fausserait cette mesure (la calibration
"verrait" les memes images que l'evaluation).

Reutilise exactement le pretraitement de evaluate.py (redimensionnement 256,
recadrage central 224, normalisation ImageNet), pour que les statistiques
d'activation observees a la calibration correspondent a celles vues a
l'inference.

Usage :
    python src/build_calibration.py --data data/imagenette2-320 --limit 200 --out data/calib.npy
"""

import argparse
import os

import numpy as np

from evaluate import preprocess


def collect_train_images(root, limit):
    train = os.path.join(root, "train")
    items = []
    for wnid in sorted(os.listdir(train)):
        folder = os.path.join(train, wnid)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".jpeg", ".jpg", ".png")):
                items.append(os.path.join(folder, name))
    # Echantillonnage regulier pour couvrir toutes les classes.
    if limit and limit < len(items):
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/imagenette2-320")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default="data/calib.npy")
    args = parser.parse_args()

    paths = collect_train_images(args.data, args.limit)
    if not paths:
        raise SystemExit(f"Aucune image trouvee sous {args.data}/train. Lancez d'abord --download.")

    samples = np.concatenate([preprocess(p) for p in paths], axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, samples)

    print(f"Images utilisees : {len(paths)}")
    print(f"Forme du tableau : {samples.shape}")
    print(f"Sortie           : {args.out}")


if __name__ == "__main__":
    main()
