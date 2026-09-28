"""
Quantification d'un modele ONNX FP32 vers INT8.

Deux modes :
  - dynamique : ne necessite aucune donnee, quantifie les poids.
    Point de depart recommande.
  - statique : necessite un jeu de calibration, quantifie poids ET activations.
    Gain plus important, mais perte de precision a mesurer.

Usage :
    python src/quantize.py --input models/model.onnx --output models/model_int8.onnx
    python src/quantize.py --input models/model.onnx --output models/model_int8_static.onnx \
        --mode static --calibration data/calib.npy
"""

import argparse
import os

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from modelutils import check_model, model_size_mb


class NpyCalibrationReader(CalibrationDataReader):
    """Lit un fichier .npy de forme (N, ...) et le presente echantillon par echantillon."""

    def __init__(self, npy_path, input_name, limit=100):
        data = np.load(npy_path)
        self.samples = [
            {input_name: np.expand_dims(x, 0).astype(np.float32)}
            for x in data[:limit]
        ]
        self.iterator = iter(self.samples)

    def get_next(self):
        return next(self.iterator, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--calibration", default=None,
                        help="Fichier .npy de donnees de calibration (mode statique uniquement).")
    parser.add_argument("--input-name", default=None,
                        help="Nom du tenseur d'entree (mode statique). Detecte automatiquement si omis.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    size_before = model_size_mb(args.input)

    if args.mode == "dynamic":
        quantize_dynamic(
            model_input=args.input,
            model_output=args.output,
            weight_type=QuantType.QInt8,
        )
    else:
        if not args.calibration:
            raise SystemExit("Le mode statique requiert --calibration.")
        import onnxruntime as ort
        input_name = args.input_name
        if input_name is None:
            input_name = ort.InferenceSession(
                args.input, providers=["CPUExecutionProvider"]
            ).get_inputs()[0].name
        reader = NpyCalibrationReader(args.calibration, input_name)
        quantize_static(
            model_input=args.input,
            model_output=args.output,
            calibration_data_reader=reader,
            quant_format=None,
        )

    check_model(args.output)

    size_after = model_size_mb(args.output)
    reduction = (1 - size_after / size_before) * 100

    print(f"Mode          : {args.mode}")
    print(f"Avant         : {size_before:.2f} Mo")
    print(f"Apres         : {size_after:.2f} Mo")
    print(f"Reduction     : {reduction:.1f} %")
    print(f"Sortie        : {args.output}")


if __name__ == "__main__":
    main()
