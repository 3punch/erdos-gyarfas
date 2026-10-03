"""Matplotlib figures for the experiment log.

All figures are written to PNG at 150 dpi and are also returned as ``Figure``
objects so notebooks can reuse them.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pow2_counts_vs_n(
    rows: Sequence[Mapping], outdir: str, name: str = "pow2_vs_n.png"
):
    """2^k cycle density against graph size -- the central scaling plot."""
    rows = [r for r in rows if r.get("algorithm") == "best"]
    lengths = sorted({int(k[1:]) for r in rows for k in r if k.startswith("c") and k[1:].isdigit()})
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for L in lengths:
        ys = [r.get(f"c{L}", np.nan) for r in rows]
        axes[0].plot(ns, ys, marker="o", ms=4, label=f"C{L}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of vertices n")
    axes[0].set_ylabel("number of cycles (log scale)")
    axes[0].set_title("Power-of-two cycle counts in best candidates")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    dens = [r["cycle_density_per_vertex"] for r in rows]
    axes[1].plot(ns, dens, marker="s", ms=4, color="crimson")
    axes[1].set_xlabel("number of vertices n")
    axes[1].set_ylabel("total 2^k cycles / n")
    axes[1].set_title("Cycle density grows with n")
    axes[1].grid(alpha=0.3)
    return _save(fig, os.path.join(outdir, name))


def plot_theory_vs_measured(rows: Sequence[Mapping], outdir: str,
                            name: str = "theory_vs_measured.png"):
    """Measured counts against the random-regular prediction 2^(L-1)/L."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ns = sorted({r["n"] for r in rows})
    lengths = [4, 8, 16, 32]
    for L in lengths:
        ys = [np.mean([r[f"c{L}"] for r in rows if r["n"] == n and f"c{L}" in r])
              for n in ns if any(f"c{L}" in r for r in rows if r["n"] == n)]
        xs = [n for n in ns if any(f"c{L}" in r for r in rows if r["n"] == n)]
        if xs:
            ax.plot(xs, ys, marker="o", ms=4, label=f"measured C{L}")
        ax.axhline(2 ** (L - 1) / L, ls="--", lw=1, alpha=0.6,
                   label=f"random cubic E[C{L}]={2 ** (L - 1) / L:.3g}")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("number of cycles")
    ax.set_title("Random cubic graphs sit on the Poisson mean; the search cannot")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _save(fig, os.path.join(outdir, name))


def plot_induced_path_vs_n(rows: Sequence[Mapping], outdir: str,
                           name: str = "induced_path_vs_n.png"):
    """Longest induced path against n, with the P13 threshold marked."""
    best = [r for r in rows if r.get("algorithm") == "best"]
    ns = [r["n"] for r in best]
    ips = [r["longest_induced_path_lb"] for r in best]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(ns, ips, marker="o", label="longest induced path (lower bound)")
    ax.axhline(13, color="crimson", ls="--",
               label="P13 threshold (required of any counterexample)")
    ax.plot(ns, ns, ls=":", color="grey", label="y = n")
    ax.set_xlabel("n")
    ax.set_ylabel("vertices in the longest induced path")
    ax.set_title("Every candidate clears the induced-P13 bar by a wide margin")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, os.path.join(outdir, name))


def plot_spectral(rows: Sequence[Mapping], outdir: str, name: str = "spectral.png"):
    """Expansion of low-cycle graphs: lambda_2 against the cycle count."""
    best = [r for r in rows if r.get("algorithm") == "best" and "lambda_2" in r]
    if not best:
        return None
    l2 = [r["lambda_2"] for r in best]
    c8 = [max(r.get("c8", 0), 0.5) for r in best]
    ns = [r["n"] for r in best]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = axes[0].scatter(ns, l2, c=c8, cmap="viridis", norm=matplotlib.colors.LogNorm())
    axes[0].axhline(2 * np.sqrt(2), ls="--", color="crimson",
                    label=r"Alon-Boppana $2\sqrt{2}\approx2.83$")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel(r"$\lambda_2$")
    axes[0].set_title("Expansion of the best candidates")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    fig.colorbar(sc, ax=axes[0], label="C8 count")

    gap = [r["spectral_gap"] for r in best]
    axes[1].plot(ns, gap, marker="o", color="seagreen")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel(r"$3-\lambda_2$")
    axes[1].set_title("Spectral gap vs size")
    axes[1].grid(alpha=0.3)
    return _save(fig, os.path.join(outdir, name))


def plot_search_curves(curves: Mapping[str, np.ndarray], outdir: str,
                       name: str = "search_curves.png", title: str = ""):
    """Best-loss traces of the competing drivers on the same instance."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, arr in curves.items():
        if arr.size == 0:
            continue
        ax.plot(arr[:, 0], arr[:, 2], label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("best loss")
    ax.set_title(title or "Search progress")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, os.path.join(outdir, name))


def plot_candidate(graph, outdir: str, name: str = "candidate.png", title: str = ""):
    """Draw a candidate graph, colouring the vertices on power-of-two cycles."""
    import networkx as nx

    G = graph.to_networkx()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    pos = nx.spring_layout(G, seed=7, k=0.6)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=42, node_color="steelblue")
    ax.set_axis_off()
    ax.set_title(title or f"cubic candidate, n={graph.n}")
    return _save(fig, os.path.join(outdir, name))


def plot_scaling_timing(rows: Sequence[Mapping], outdir: str,
                        name: str = "scaling_timing.png"):
    """Throughput and wall-clock against n -- the scalability evidence."""
    rows = [r for r in rows if r.get("moves_per_second")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    by_algo: dict[str, list] = {}
    for r in rows:
        by_algo.setdefault(r["algorithm"], []).append(r)
    for algo, rs in by_algo.items():
        rs = sorted(rs, key=lambda r: r["n"])
        ax.plot([r["n"] for r in rs], [r["moves_per_second"] for r in rs],
                marker="o", label=algo)
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("2-opt moves / second")
    ax.set_title("Search throughput scaling")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, os.path.join(outdir, name))
