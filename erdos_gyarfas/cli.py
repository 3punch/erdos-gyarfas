"""Command-line entry point.

Examples
--------
    python -m erdos_gyarfas.cli warmup
    python -m erdos_gyarfas.cli stage --stage large --workers 2
    python -m erdos_gyarfas.cli validate
    python -m erdos_gyarfas.cli benchmark --sizes 14 20 24
    python -m erdos_gyarfas.cli all
    python -m erdos_gyarfas.cli hyperscale
    python -m erdos_gyarfas.cli make-report
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="erdos-gyarfas", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--outdir", default="results")
        sp.add_argument("--workers", type=int, default=0,
                        help="parallel workers (0 = one per CPU)")
        sp.add_argument("--scale", type=float, default=1.0,
                        help="multiply every move/step budget by this factor")

    sp = sub.add_parser("all", help="run every stage plus validation")
    add_common(sp)

    sp = sub.add_parser("stage", help="run a single stage")
    sp.add_argument("--stage", required=True, choices=["warmup", "medium", "large"])
    sp.add_argument("--sizes", type=int, nargs="*", default=None)
    sp.add_argument("--seeds", type=int, default=None)
    sp.add_argument("--algorithms", nargs="*", default=None)
    add_common(sp)

    for name in ("warmup", "medium", "large"):
        sp = sub.add_parser(name, help=f"shortcut for --stage {name}")
        add_common(sp)

    sp = sub.add_parser("validate", help="check cycle counts against Poisson theory")
    sp.add_argument("--outdir", default="results/validation")
    sp.add_argument("--sizes", type=int, nargs="*", default=[50, 100, 200, 400, 1000])
    sp.add_argument("--trials", type=int, default=8)

    sp = sub.add_parser("benchmark", help="reproduce the no-C4-no-C8 benchmark")
    sp.add_argument("--outdir", default="results/benchmark")
    sp.add_argument("--sizes", type=int, nargs="*", default=[14, 20, 24, 30, 40])

    sp = sub.add_parser("pareto", help="trace the achievable (C4, C8) frontier")
    sp.add_argument("--outdir", default="results/pareto")
    sp.add_argument("--sizes", type=int, nargs="*", default=[100, 200, 400])
    sp.add_argument("--moves", type=int, default=200_000)

    sp = sub.add_parser("hyperscale",
                        help="run the N=100k early-exit scaling pipeline")
    sp.add_argument("--outdir", default="results/hyperscale")
    sp.add_argument("--sizes", type=int, nargs="*",
                    default=[10_000, 25_000, 50_000, 100_000])
    sp.add_argument("--workers", type=int, default=0)
    sp.add_argument("--max-length", type=int, default=2048)

    sp = sub.add_parser("report", help="summarise an existing results directory")
    sp.add_argument("--outdir", default="results")

    sp = sub.add_parser("make-report",
                        help="regenerate docs/EXPERIMENT_LOG.md from results/")
    sp.add_argument("--outdir", default="results")
    sp.add_argument("--out", default="docs/EXPERIMENT_LOG.md")
    return p


def main(argv: list[str] | None = None) -> int:
    from .experiments import runner

    args = build_parser().parse_args(argv)
    cmd = args.command
    if cmd == "all":
        runner.run_all(args.outdir, workers=args.workers, scale=args.scale)
    elif cmd == "stage":
        runner.run_stage(args.stage, outdir=args.outdir, workers=args.workers,
                         sizes=args.sizes, seeds=args.seeds,
                         algorithms=args.algorithms, scale=args.scale)
    elif cmd in ("warmup", "medium", "large"):
        runner.run_stage(cmd, outdir=args.outdir, workers=args.workers,
                         scale=args.scale)
    elif cmd == "validate":
        runner.run_validation(args.outdir, sizes=args.sizes, trials=args.trials)
    elif cmd == "benchmark":
        runner.run_benchmark(args.outdir, sizes=args.sizes)
    elif cmd == "pareto":
        runner.run_pareto(args.outdir, sizes=args.sizes, moves=args.moves)
    elif cmd == "report":
        from .experiments.report import summarize
        summarize(args.outdir)
    elif cmd == "hyperscale":
        from .hyperscale import run_hyperscale
        run_hyperscale(args.outdir, sizes=args.sizes, workers=args.workers,
                       max_length=args.max_length)
    elif cmd == "make-report":
        from .experiments.make_report import make_report
        path = make_report(args.outdir, args.out)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
