#!/usr/bin/env python
"""Compare a `kiln bench --json-out` file against a stored baseline; exit 1 on regression.

Each benchmark row is keyed by its non-numeric columns; every numeric column
listed in HIGHER_IS_BETTER / LOWER_IS_BETTER is compared with a relative
tolerance. Baselines are refreshed deliberately (`--write-baseline`), never
automatically, so a slow-drift regression cannot hide.
"""

from __future__ import annotations

import argparse
import json
import sys

HIGHER_IS_BETTER = {"speedup", "eff_GB_s", "throughput_rps", "Gbps", "steps_per_s", "mfu"}
LOWER_IS_BETTER = {"fused_ms", "ttft_p99_ms", "e2e_p99_s", "sync_ms", "preemptions"}
METRICS = (
    HIGHER_IS_BETTER
    | LOWER_IS_BETTER
    | {"naive_ms", "ttft_p50_ms", "e2e_p50_s", "peak_kv_util", "bytes_MB", "wall_s", "mean_staleness"}
)


def key_of(row: dict) -> str:
    """Identity of a row = every column that is not a measured metric."""
    return "|".join(f"{k}={v}" for k, v in row.items() if k not in METRICS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("current")
    ap.add_argument("baseline")
    ap.add_argument("--tolerance", type=float, default=0.2)
    ap.add_argument("--write-baseline", action="store_true")
    a = ap.parse_args()
    cur = json.load(open(a.current, encoding="utf-8"))
    if a.write_baseline:
        json.dump(cur, open(a.baseline, "w", encoding="utf-8"), indent=2)
        print(f"wrote baseline {a.baseline}")
        return 0
    try:
        base = json.load(open(a.baseline, encoding="utf-8"))
    except FileNotFoundError:
        print(f"no baseline at {a.baseline}; nothing to compare (run with --write-baseline)")
        return 0
    failures = []
    for bench, rows in cur.items():
        base_rows = {key_of(r): r for r in base.get(bench, [])}
        for r in rows:
            b = base_rows.get(key_of(r))
            if b is None:
                continue
            for col, v in r.items():
                if col not in HIGHER_IS_BETTER | LOWER_IS_BETTER or not isinstance(v, (int, float)):
                    continue
                bv = b.get(col)
                if not isinstance(bv, (int, float)) or bv == 0:
                    continue
                ratio = v / bv
                bad = ratio < 1 - a.tolerance if col in HIGHER_IS_BETTER else ratio > 1 + a.tolerance
                if bad:
                    failures.append(f"{bench} [{key_of(r)}] {col}: {bv:.4g} -> {v:.4g} ({ratio:.2f}x)")
    for f in failures:
        print("REGRESSION", f)
    print(f"{len(failures)} regressions" if failures else "no perf regressions")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
