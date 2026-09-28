"""
Utilitaires partages de la chaine.

model_size_mb() corrige un defaut silencieux : depuis les versions recentes
de torch, l'export ONNX peut produire un fichier squelette accompagne d'un
fichier de donnees externes (model.onnx + model.onnx.data). Mesurer la seule
taille du .onnx donne alors une valeur tres inferieure a la taille reelle,
ce qui rend la barriere de budget inoperante.
"""

import os

import onnx


def external_data_files(path):
    """Noms des fichiers de donnees externes references par le modele."""
    model = onnx.load(path, load_external_data=False)
    names = set()

    def scan(tensors):
        for t in tensors:
            if t.data_location == onnx.TensorProto.EXTERNAL:
                for kv in t.external_data:
                    if kv.key == "location":
                        names.add(kv.value)

    scan(model.graph.initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                scan([attr.t])
            elif attr.type == onnx.AttributeProto.TENSORS:
                scan(attr.tensors)
    return names


def model_size_mb(path):
    """Taille reelle du modele : le .onnx plus ses fichiers de donnees externes."""
    total = os.path.getsize(path)
    directory = os.path.dirname(os.path.abspath(path))
    for name in external_data_files(path):
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            total += os.path.getsize(candidate)
    return total / (1024 * 1024)


def check_model(path):
    """Verifie la validite structurelle du modele. Leve une exception si invalide."""
    onnx.checker.check_model(path, full_check=True)
    return True
