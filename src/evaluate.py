"""
Mesure de la qualite d'un modele ONNX apres quantification.

Deux metriques complementaires :

  1. Precision top-1 sur Imagenette (sous-ensemble public d'ImageNet, 10 classes).
     La correspondance identifiant WordNet -> indice ImageNet est DERIVEE a
     l'execution depuis le fichier de reference de torchvision, et non codee
     en dur : aucune constante ne peut donc etre silencieusement fausse.

  2. Taux d'accord avec le modele de reference FP32 : sur les memes images,
     fraction des cas ou le modele quantifie predit la meme classe.
     Cette metrique ne depend d'aucune correspondance de libelles et reste
     valable meme si le jeu d'evaluation n'est pas etiquete.

Usage :
    python src/evaluate.py --download
    python src/evaluate.py --model models/model.onnx --limit 500 --out results/acc_fp32.json
    python src/evaluate.py --model models/model_int8_static.onnx --limit 500 \
        --reference models/model.onnx --out results/acc_int8.json
"""

import argparse
import json
import os
import tarfile
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
CLASS_INDEX_URL = (
    "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/"
    "imagenet_class_index.json"
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def wordnet_to_index():
    """Correspondance identifiant WordNet -> indice ImageNet-1k, derivee a l'execution."""
    with urllib.request.urlopen(CLASS_INDEX_URL, timeout=30) as response:
        table = json.load(response)
    return {value[0]: int(key) for key, value in table.items()}


def download_dataset(root="data"):
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, "imagenette2-320.tgz")
    target = os.path.join(root, "imagenette2-320")
    if os.path.isdir(target):
        print(f"Deja present : {target}")
        return target
    if not os.path.exists(archive):
        print("Telechargement d'Imagenette (environ 325 Mo)...")
        urllib.request.urlretrieve(IMAGENETTE_URL, archive)
    print("Extraction...")
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    print(f"Pret : {target}")
    return target


def preprocess(path, size=224):
    """Pretraitement conforme aux modeles torchvision : redimensionnement 256,
    recadrage central 224, normalisation ImageNet."""
    image = Image.open(path).convert("RGB")
    short = 256
    width, height = image.size
    scale = short / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
    width, height = image.size
    left, top = (width - size) // 2, (height - size) // 2
    image = image.crop((left, top, left + size, top + size))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))[np.newaxis, :].astype(np.float32)


def collect_images(root, limit):
    """Liste (chemin, identifiant WordNet) depuis le dossier de validation."""
    val = os.path.join(root, "val")
    items = []
    for wnid in sorted(os.listdir(val)):
        folder = os.path.join(val, wnid)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".jpeg", ".jpg", ".png")):
                items.append((os.path.join(folder, name), wnid))
    # Echantillonnage regulier pour couvrir toutes les classes
    if limit and limit < len(items):
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    return items


def make_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Telecharge Imagenette puis termine.")
    parser.add_argument("--model")
    parser.add_argument("--reference", default=None,
                        help="Modele FP32 de reference, pour calculer le taux d'accord.")
    parser.add_argument("--data", default="data/imagenette2-320")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.download:
        download_dataset()
        return
    if not args.model:
        raise SystemExit("--model est requis (ou utilisez --download).")

    mapping = wordnet_to_index()
    images = collect_images(args.data, args.limit)
    if not images:
        raise SystemExit(f"Aucune image trouvee sous {args.data}. Lancez --download d'abord.")

    session = make_session(args.model)
    input_name = session.get_inputs()[0].name
    reference = make_session(args.reference) if args.reference else None
    reference_input = reference.get_inputs()[0].name if reference else None

    correct = 0
    agree = 0
    for path, wnid in images:
        tensor = preprocess(path)
        predicted = int(np.argmax(session.run(None, {input_name: tensor})[0]))
        if predicted == mapping[wnid]:
            correct += 1
        if reference is not None:
            base = int(np.argmax(reference.run(None, {reference_input: tensor})[0]))
            if predicted == base:
                agree += 1

    total = len(images)
    report = {
        "model": os.path.basename(args.model),
        "images": total,
        "top1_accuracy": round(100 * correct / total, 2),
        "dataset": "imagenette2-320 (val)",
    }
    if reference is not None:
        report["reference"] = os.path.basename(args.reference)
        report["agreement_with_reference"] = round(100 * agree / total, 2)

    print(json.dumps(report, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n-> rapport ecrit dans {args.out}")


if __name__ == "__main__":
    main()
