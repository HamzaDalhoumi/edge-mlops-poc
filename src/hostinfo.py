"""
Description de la machine d'execution, jointe a chaque rapport de mesure.

Pourquoi le modele du processeur et les instructions de produit scalaire
INT8 : sur x86 sans VNNI (avx512_vnni / avx_vnni), ONNX Runtime calcule les
convolutions U8S8 avec VPMADDUBSW, qui sature. Le meme modele QDQ a ainsi
donne 64.0 % top-1 sur un Ryzen 7 5800H (AVX2 sans VNNI) contre 72.4 % en CI.
Un chiffre de precision INT8 n'est donc interpretable qu'avec cette
information. Sur ARM, l'equivalent est l'extension dotprod (drapeau Linux
"asimddp").

Detection par la bibliotheque standard uniquement :
  - Linux : /proc/cpuinfo ;
  - Windows : nom du processeur via le registre ; drapeaux non accessibles,
    signales null (inconnu) plutot que false.
"""

import os
import platform

X86_INT8_DOT_FLAGS = ("avx512_vnni", "avx_vnni")
ARM_INT8_DOT_FLAGS = {"asimddp": "dotprod"}


def _linux_cpuinfo():
    fields = {}
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                key, sep, value = line.partition(":")
                key = key.strip()
                # Premier coeur uniquement : les champs se repetent par coeur.
                if sep and key not in fields:
                    fields[key] = value.strip()
    except OSError:
        pass
    return fields


def _windows_cpu_model():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        return None


def cpu_description():
    """Modele du processeur et presence des instructions de produit scalaire INT8.

    int8_dot_product vaut un dictionnaire {drapeau: bool} si la detection est
    possible, None sinon.
    """
    system = platform.system()
    if system == "Linux":
        info = _linux_cpuinfo()
        if "flags" in info:  # x86
            flags = set(info["flags"].split())
            return {
                "cpu_model": info.get("model name"),
                "int8_dot_product": {f: f in flags for f in X86_INT8_DOT_FLAGS},
            }
        if "Features" in info:  # ARM
            features = set(info["Features"].split())
            model = info.get("model name") or (
                f"implementer {info.get('CPU implementer')} part {info.get('CPU part')}"
            )
            return {
                "cpu_model": model,
                "int8_dot_product": {name: flag in features
                                     for flag, name in ARM_INT8_DOT_FLAGS.items()},
            }
    if system == "Windows":
        return {"cpu_model": _windows_cpu_model(), "int8_dot_product": None}
    return {"cpu_model": platform.processor() or None, "int8_dot_product": None}


def environment(onnxruntime_version):
    """Bloc 'environment' commun a benchmark.py et evaluate.py."""
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or "n/a",
        **cpu_description(),
        "system": platform.system(),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime_version,
        "cpu_count_visible": os.cpu_count(),
    }
