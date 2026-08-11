"""
Export d'un modele PyTorch pre-entraine vers ONNX.

Le modele de reference par defaut est MobileNetV2 (torchvision, poids ImageNet) :
architecture concue pour les cibles contraintes, donc representative du sujet,
et disponible sans phase d'entrainement.

Usage :
    python src/export_onnx.py --arch mobilenet_v2 --output models/model.onnx
"""

import argparse
import os

import torch
import torchvision.models as models

ARCHITECTURES = {
    "mobilenet_v2": models.mobilenet_v2,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "resnet18": models.resnet18,
    "squeezenet1_1": models.squeezenet1_1,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="mobilenet_v2")
    parser.add_argument("--output", default="models/model.onnx")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = ARCHITECTURES[args.arch](weights="DEFAULT")
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print(f"Architecture  : {args.arch}")
    print(f"Parametres    : {n_params:,}")
    print(f"Taille ONNX   : {size_mb:.2f} Mo")
    print(f"Sortie        : {args.output}")


if __name__ == "__main__":
    main()
