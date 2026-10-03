"""Human-readable summary of a results directory."""

from __future__ import annotations

import csv
import os
from typing import Sequence


def summarize(outdir: str = "results") -> None:
    for stage in ("warmup", "medium", "large"):
        path = os.path.join(outdir, stage, "results.csv")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        best = [r for r in rows if r.get("algorithm") == "best"]
        print(f"\n=== {stage} ({len(rows)} runs) ===")
        hdr = f"{'n':>6} {'algo':>12} {'loss':>10} {'C4':>6} {'C8':>6} {'C16':>7} {'girth':>6} {'indP':>5} {'mv/s':>10}"
        print(hdr)
        for r in best:
            print(f"{r['n']:>6} {r.get('best_algorithm',''):>12} "
                  f"{float(r['best_loss']):>10.5f} {r.get('c4',''):>6} {r.get('c8',''):>6} "
                  f"{r.get('c16',''):>7} {r.get('girth',''):>6} "
                  f"{r.get('longest_induced_path_lb',''):>5} "
                  f"{float(r.get('moves_per_second') or 0):>10,.0f}")
        any_ce = [r for r in rows if r.get("is_counterexample") == "True"]
        print(f"  certified counterexamples found: {len(any_ce)}")
