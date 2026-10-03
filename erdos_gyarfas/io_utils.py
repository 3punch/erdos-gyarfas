"""Persistence for candidate graphs and experiment records.

Two formats are written for every saved graph:

* **GraphML** -- interoperable with Gephi / yEd / igraph, carrying the
  measured attributes as node and graph properties;
* **JSON** -- a compact, self-describing record including the full
  power-of-two certificate, so a claimed counterexample can be re-verified
  from the file alone.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import numpy as np

from .core.cycles import PowerOfTwoReport, count_power_of_two_cycles
from .core.graph import CubicGraph


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def graph_to_dict(
    graph: CubicGraph,
    report: PowerOfTwoReport | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """A JSON-serialisable record of a graph plus its cycle certificate."""
    rep = report or count_power_of_two_cycles(graph)
    rec: dict[str, Any] = {
        "n": graph.n,
        "m": graph.num_edges(),
        "edges": graph.edges(),
        "degree": 3,
        "power_of_two": rep.as_dict(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
    }
    if extra:
        rec["meta"] = dict(extra)
    return rec


def save_graph_json(
    path: str,
    graph: CubicGraph,
    report: PowerOfTwoReport | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(graph_to_dict(graph, report, extra), fh, indent=2, default=_json_default)
    return path


def save_graph_graphml(
    path: str,
    graph: CubicGraph,
    report: PowerOfTwoReport | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Write GraphML via networkx (igraph's GraphML writer drops graph keys)."""
    import networkx as nx

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    G = graph.to_networkx()
    G.name = f"cubic_{graph.n}"
    if report is not None:
        G.graph["pow2_counts"] = json.dumps(
            dict(zip(report.lengths, report.counts))
        )
        G.graph["pow2_total"] = report.total
        G.graph["pow2_certified"] = bool(report.all_certified)
        G.graph["is_counterexample"] = bool(report.is_counterexample)
    for k, v in (extra or {}).items():
        G.graph[str(k)] = _graphml_scalar(v)
    nx.write_graphml(G, path)
    return path


def _graphml_scalar(v: Any) -> Any:
    """Coerce a value to something the GraphML writer accepts.

    GraphML only stores primitives, so containers and numpy scalars are
    JSON-encoded rather than dropped -- the record has to stay complete
    enough to re-verify a candidate from the file alone.
    """
    if v is None:
        return ""          # GraphML has no null; keep the key, blank the value
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    try:
        return json.dumps(_json_default(v) if not isinstance(v, (list, dict, tuple, set))
                          else v, default=_json_default)
    except TypeError:
        return str(v)


def save_candidate(
    outdir: str,
    tag: str,
    graph: CubicGraph,
    report: PowerOfTwoReport | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Save both formats and return their paths."""
    base = os.path.join(outdir, tag)
    return {
        "json": save_graph_json(base + ".json", graph, report, extra),
        "graphml": save_graph_graphml(base + ".graphml", graph, report, extra),
    }


def load_graph_json(path: str) -> CubicGraph:
    with open(path) as fh:
        rec = json.load(fh)
    return CubicGraph.from_edges(int(rec["n"]), [tuple(e) for e in rec["edges"]])


def append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")


def save_table(path: str, rows: Iterable[Mapping[str, Any]]) -> str:
    """Write a list of flat records as CSV (stable column order)."""
    import csv

    rows = list(rows)
    if not rows:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "w").close()
        return path
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: _json_default(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                        for k, v in r.items()})
    return path
