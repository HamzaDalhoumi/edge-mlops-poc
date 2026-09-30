"""
Executeur local : ONNX Runtime dans le processus courant.

Le niveau de validation est declare par l'appelant, car le meme code sert :
  - niveau 1 sur le runner ubuntu-24.04-arm du CI ;
  - niveau 2 a l'interieur d'un conteneur a ressources plafonnees ;
  - niveau 3 dans un conteneur emule (les latences sont alors supprimees
    par TargetRunner.run_benchmark) ;
  - "développement" sur un poste de travail.
"""

import platform
import statistics
import time

import numpy as np
import onnxruntime as ort

from hostinfo import cpu_description
from runners.base import TargetRunner


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


def _windows_peak_memory_bytes():
    """PeakWorkingSetSize du processus, via GetProcessMemoryInfo (psapi)."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    # Signatures explicites : sans elles, ctypes suppose des int 32 bits et
    # tronque le handle de processus (un pointeur) sous Windows 64 bits.
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    get_memory_info.restype = wintypes.BOOL

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo a echoue.")
    return counters.PeakWorkingSetSize


def peak_memory_mb():
    """Pic de memoire resident du processus, en Mo."""
    if platform.system() == "Windows":
        return _windows_peak_memory_bytes() / (1024 * 1024)
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux renvoie des kilo-octets, macOS des octets.
    return usage / 1024 if platform.system() == "Linux" else usage / (1024 * 1024)


class LocalRunner(TargetRunner):
    name = "local"

    def _describe_target(self):
        return {"architecture": platform.machine(), **cpu_description()}

    def _measure(self, model, runs, warmup, threads):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])

        feeds = build_random_inputs(session)

        for _ in range(warmup):
            session.run(None, feeds)

        latencies_ms = []
        for _ in range(runs):
            start = time.perf_counter()
            session.run(None, feeds)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        latencies_ms.sort()
        p95_index = min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))

        return {
            "latency_ms": {
                "median": round(statistics.median(latencies_ms), 3),
                "mean": round(statistics.fmean(latencies_ms), 3),
                "p95": round(latencies_ms[p95_index], 3),
                "min": round(latencies_ms[0], 3),
                "max": round(latencies_ms[-1], 3),
            },
            "peak_memory_mb": round(peak_memory_mb(), 1),
        }
