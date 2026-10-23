"""Node topology discovery and NCCL environment recommendations.

Parses ``nvidia-smi topo -m`` to learn which GPU pairs share NVLink/NVSwitch
versus crossing PCIe/host bridges, and inspects InfiniBand HCAs under
``/sys/class/infiniband``. The result drives two decisions:

* which GPUs form the tensor-parallel group (must be NVLink-connected), and
* which ``NCCL_*`` variables to set for the fabric that is actually present
  (e.g. do not force ``NCCL_IB_HCA`` on a node with no HCA, do enable
  ``NCCL_NVLS_ENABLE`` on NVSwitch nodes).

Everything is best-effort: on a machine without GPUs you get an empty
topology and an empty recommendation, never an exception.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_LINK_RANK = {"NV": 0, "PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 5}


@dataclass
class Topology:
    num_gpus: int = 0
    links: dict[tuple[int, int], str] = field(default_factory=dict)
    ib_hcas: list[str] = field(default_factory=list)
    nvswitch: bool = False

    def link(self, a: int, b: int) -> str:
        return self.links.get((min(a, b), max(a, b)), "SYS")

    def nvlink_islands(self) -> list[list[int]]:
        """Connected components under NVLink adjacency."""
        seen: set[int] = set()
        islands: list[list[int]] = []
        for g in range(self.num_gpus):
            if g in seen:
                continue
            stack, comp = [g], []
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                comp.append(x)
                stack.extend(y for y in range(self.num_gpus) if y != x and self.link(x, y).startswith("NV"))
            islands.append(sorted(comp))
        return islands

    def tp_groups(self, tp: int) -> list[list[int]]:
        """Partition GPUs into TP groups that stay inside NVLink islands where possible."""
        groups: list[list[int]] = []
        for island in self.nvlink_islands():
            for i in range(0, len(island) - tp + 1, tp):
                groups.append(island[i : i + tp])
        covered = {g for grp in groups for g in grp}
        leftovers = [g for g in range(self.num_gpus) if g not in covered]
        for i in range(0, len(leftovers) - tp + 1, tp):
            groups.append(leftovers[i : i + tp])
        return groups


def _parse_topo_matrix(text: str) -> Topology:
    topo = Topology()
    rows = [ln for ln in text.splitlines() if re.match(r"^GPU\d+\s", ln)]
    topo.num_gpus = len(rows)
    for i, ln in enumerate(rows):
        cells = ln.split()[1 : 1 + topo.num_gpus]
        for j, c in enumerate(cells):
            if i < j:
                topo.links[(i, j)] = c
    nv_links = [v for v in topo.links.values() if v.startswith("NV")]
    topo.nvswitch = topo.num_gpus >= 8 and len(nv_links) == len(topo.links) and len(topo.links) > 0
    return topo


def discover(sysfs_root: str | Path = "/sys/class/infiniband") -> Topology:
    topo = Topology()
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10
            ).stdout
            topo = _parse_topo_matrix(out)
        except Exception:
            pass
    root = Path(sysfs_root)
    if root.exists():
        topo.ib_hcas = sorted(p.name for p in root.iterdir())
    return topo


def recommend_nccl_env(topo: Topology, cross_node: bool = True) -> dict[str, str]:
    env: dict[str, str] = {
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    if topo.ib_hcas and cross_node:
        env["NCCL_IB_HCA"] = ",".join(h for h in topo.ib_hcas if h.startswith("mlx"))
        env["NCCL_IB_GID_INDEX"] = "3"
        env["NCCL_IB_TIMEOUT"] = "22"
        env["NCCL_NET_GDR_LEVEL"] = "PIX"
    elif cross_node:
        env["NCCL_IB_DISABLE"] = "1"
        env["NCCL_SOCKET_NTHREADS"] = "4"
        env["NCCL_NSOCKS_PERTHREAD"] = "4"
    if topo.nvswitch:
        env["NCCL_NVLS_ENABLE"] = "1"
    if topo.num_gpus >= 8:
        env["NCCL_MIN_NCHANNELS"] = "16"
    return env


def describe(topo: Topology) -> str:
    lines = [f"GPUs: {topo.num_gpus}", f"NVSwitch: {topo.nvswitch}", f"IB HCAs: {topo.ib_hcas or 'none'}"]
    for island in topo.nvlink_islands():
        lines.append(f"NVLink island: {island}")
    return "\n".join(lines)
