"""Background GPU utilisation / memory sampler (NVML, falling back to ``nvidia-smi``).

Returns nothing on machines without NVIDIA GPUs — callers report "n/a" rather than
a made-up number.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class GPUSample:
    t: float
    util: list[float]
    mem_used_gb: list[float]


@dataclass
class GPUMonitor:
    interval_s: float = 0.5
    samples: list[GPUSample] = field(default_factory=list)
    backend: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _read(self):
        if self.backend == "nvml":
            import pynvml

            util, mem = [], []
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util.append(float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu))
                mem.append(pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9)
            return util, mem
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        util, mem = [], []
        for line in out.strip().splitlines():
            u, m = line.split(",")
            util.append(float(u))
            mem.append(float(m) / 1024)
        return util, mem

    def available(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self.backend = "nvml"
            return True
        except Exception:
            pass
        if shutil.which("nvidia-smi"):
            self.backend = "nvidia-smi"
            try:
                self._read()
                return True
            except Exception:
                return False
        return False

    def start(self) -> GPUMonitor:
        if not self.available():
            return self

        def loop():
            t0 = time.perf_counter()
            while not self._stop.is_set():
                try:
                    u, m = self._read()
                    self.samples.append(GPUSample(time.perf_counter() - t0, u, m))
                except Exception:
                    pass
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True, name="kiln-gpu-monitor")
        self._thread.start()
        return self

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.samples:
            return {"gpu_util_mean": None, "gpu_mem_peak_gb": None, "gpu_backend": None}
        utils = [sum(s.util) / len(s.util) for s in self.samples if s.util]
        return {
            "gpu_util_mean": sum(utils) / len(utils),
            "gpu_mem_peak_gb": max(max(s.mem_used_gb) for s in self.samples if s.mem_used_gb),
            "gpu_backend": self.backend,
            "gpu_count": len(self.samples[0].util),
        }
