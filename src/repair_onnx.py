"""
Reparation d'un graphe ONNX dont les annotations de forme sont incoherentes.

Symptome traite :
    InferenceError: Inferred shape and existing shape differ in dimension 0:
    (1280) vs (1000)

Cause : certains exporteurs (notamment l'exporteur "dynamo" de torch)
ecrivent dans le graphe des annotations de forme intermediaires (value_info)
qui ne correspondent pas aux formes reellement calculees. L'inference de
formes d'ONNX detecte la contradiction et echoue, ce qui bloque en amont
toute quantification.

Reparation : supprimer les annotations intermediaires, qui sont purement
informatives, puis laisser ONNX les recalculer. Les entrees, les sorties et
les poids ne sont pas touches : le modele est fonctionnellement identique.

Usage :
    python src/repair_onnx.py --input models/model.onnx --output models/model_clean.onnx
"""

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort


def repair(src, dst):
    model = onnx.load(src)
    removed = len(model.graph.value_info)
    del model.graph.value_info[:]

    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst)
    return removed, len(model.graph.value_info)


def same_outputs(a, b, tolerance=1e-4):
    """Verifie que les deux modeles produisent la meme sortie sur une entree identique."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sa = ort.InferenceSession(a, options, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(b, options, providers=["CPUExecutionProvider"])

    feeds = {}
    for spec in sa.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
        feeds[spec.name] = np.random.rand(*shape).astype(np.float32)

    out_a = sa.run(None, feeds)[0]
    out_b = sb.run(None, {sb.get_inputs()[0].name: list(feeds.values())[0]})[0]
    delta = float(np.max(np.abs(out_a - out_b)))
    return delta <= tolerance, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    removed, rebuilt = repair(args.input, args.output)

    print(f"Annotations supprimees   : {removed}")
    print(f"Annotations recalculees  : {rebuilt}")

    ok, delta = same_outputs(args.input, args.output)
    print(f"Sorties identiques       : {'oui' if ok else 'NON'} (ecart max {delta:.2e})")
    if not ok:
        raise SystemExit("La reparation a modifie le comportement du modele : a ne pas utiliser.")
    print(f"Sortie                   : {args.output}")


if __name__ == "__main__":
    main()
