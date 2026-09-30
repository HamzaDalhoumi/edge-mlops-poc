"""
Banc de mesure d'un modele ONNX.

Mesure, pour un modele donne :
  - la taille du fichier sur disque
  - la latence d'inference (mediane, p95) sur N iterations
  - le pic de memoire resident du processus

Le resultat est ecrit en JSON, ce qui permet de le comparer
automatiquement a un budget dans le pipeline d'integration continue.

Usage :
    python src/benchmark.py --model models/model.onnx --runs 50 --out results/bench.json
"""

import argparse
import json
import os
import platform
import resource
import statistics
import time

import numpy as np
import onnxruntime as ort

from hostinfo import environment
from modelutils import model_size_mb


def build_random_inputs(session):
    """Genere des entrees aleatoires conformes a la signature du modele."""
    feeds = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dtype = np.float32
        if "int64" in inp.type:
            dtype = np.int64
        elif "int32" in inp.type:
            dtype = np.int32
        if dtype == np.float32:
            feeds[inp.name] = np.random.rand(*shape).astype(dtype)
        else:
            feeds[inp.name] = np.zeros(shape, dtype=dtype)
    return feeds


def peak_memory_mb():
    """Pic de memoire resident du processus, en Mo."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux renvoie des kilo-octets, macOS des octets.
    return usage / 1024 if platform.system() == "Linux" else usage / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1,
                        help="Nombre de threads intra-op. 1 = comportement le plus proche d'une cible contrainte.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])

    feeds = build_random_inputs(session)

    for _ in range(args.warmup):
        session.run(None, feeds)

    latencies_ms = []
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(None, feeds)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    p95_index = min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))

    report = {
        "model": os.path.basename(args.model),
        "size_mb": round(model_size_mb(args.model), 3),
        "runs": args.runs,
        "threads": args.threads,
        "latency_ms": {
            "median": round(statistics.median(latencies_ms), 3),
            "mean": round(statistics.fmean(latencies_ms), 3),
            "p95": round(latencies_ms[p95_index], 3),
            "min": round(latencies_ms[0], 3),
            "max": round(latencies_ms[-1], 3),
        },
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "environment": environment(ort.__version__),
    }

    print(json.dumps(report, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n-> rapport ecrit dans {args.out}")


if __name__ == "__main__":
    main()
