"""Read one shard's inner-trace artifacts back and explain one task.

The inner trace (:mod:`landsat_lst.innertrace`) persists, per longitude group,
the graph the scheduler received and the execution history joined onto it.
This module is the reader: it answers, for one inner task, what it depended
on, when it ran and where, and every part of the delay between the moment
``dask.compute`` was entered and the moment that task started. It reads only
the persisted objects, so it works during a run against S3 and after the
cluster is gone, and never touches a scheduler.

Every number here is copied from the artifacts. Readiness in particular is
never inferred: a key that did not execute is shown with the
``not_executed_reason`` the trace recorded from scheduler evidence, or
``unknown``.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from landsat_lst import shards

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from landsat_lst.storage import StorageBackend


@dataclass
class GroupTrace:
    """One group's graph and execution, as persisted."""

    graph: dict[str, Any]
    execution: dict[str, Any]
    graph_key: str
    exec_key: str
    nodes: dict[str, dict[str, Any]] = field(init=False)
    rows: dict[str, dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.nodes = {n["key"]: n for n in self.graph.get("nodes", [])}
        self.rows = {r["key"]: r for r in self.execution.get("rows", [])}

    @property
    def group(self) -> list[int] | None:
        return self.graph.get("group")

    def dependents(self, key: str) -> list[str]:
        return sorted(k for k, n in self.nodes.items() if key in n.get("deps", []))


def _read_json_gz(storage: StorageBackend, key: str) -> dict[str, Any] | None:
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="lst_explain_") as directory:
        local = Path(directory) / "payload.json.gz"
        if not storage.download(key, local):
            return None
        with gzip.open(local, "rt", encoding="utf-8") as stream:
            return json.load(stream)


def list_groups(
    storage: StorageBackend, run_id: str, stage: str, tile: str, index: int
) -> dict[int, dict[int, tuple[str | None, str | None]]]:
    """Every (attempt, group) with its graph and exec keys, from one listing."""
    stem = shards.unit_trace_prefix(run_id, stage, tile, index)
    listing = storage.list_prefix(f"{stem}.inner-")
    found: dict[int, dict[int, tuple[str | None, str | None]]] = {}
    for key in listing:
        name = key[len(stem) + 1 :]
        if not name.startswith(("inner-graph.", "inner-exec.")):
            continue
        kind, rest = name.split(".", 1)
        parts = rest.split(".")
        try:
            attempt = int(parts[0].lstrip("a"))
            group = int(parts[1].lstrip("g"))
        except (IndexError, ValueError):
            continue
        graph_key, exec_key = found.setdefault(attempt, {}).get(group, (None, None))
        if kind == "inner-graph":
            graph_key = key
        else:
            exec_key = key
        found[attempt][group] = (graph_key, exec_key)
    return found


def load_group(
    storage: StorageBackend,
    run_id: str,
    stage: str,
    tile: str,
    index: int,
    *,
    attempt: int,
    group: int,
) -> GroupTrace | None:
    """Load one group's pair of artifacts, or ``None`` when either is absent."""
    groups = list_groups(storage, run_id, stage, tile, index)
    graph_key, exec_key = groups.get(attempt, {}).get(group, (None, None))
    if graph_key is None or exec_key is None:
        return None
    graph = _read_json_gz(storage, graph_key)
    execution = _read_json_gz(storage, exec_key)
    if graph is None or execution is None:
        return None
    return GroupTrace(graph=graph, execution=execution, graph_key=graph_key, exec_key=exec_key)


def _fmt_s(ns: int | float | None) -> str:
    if ns is None:
        return "-"
    return f"{ns / 1e9:.3f}s"


def delay_breakdown(trace: GroupTrace, key: str) -> list[tuple[str, float | None]]:
    """The delay from ``dask.compute`` entry to this key's execution, in order.

    Each entry is a named interval in seconds, copied from the group's
    construction record and the key's own row. The intervals are sequential on
    the shard's main thread up to first dispatch; the last one, the wait from
    first dispatch to *this* key's start, is the scheduler's queueing and the
    key's dependencies executing, which the dependency tree shows in detail.
    """
    c = trace.execution.get("construction") or {}
    row = trace.rows.get(key) or {}
    entry = c.get("compute_entry_ns")
    first = c.get("first_dispatch_ns")
    start = row.get("start_ns")
    out: list[tuple[str, float | None]] = [
        ("store_build", c.get("store_build_s")),
        ("pre_scheduler (dask optimize+materialize)", c.get("pre_scheduler_s")),
        ("materialize (convert to task specs)", c.get("materialize_s")),
        ("capture_overhead (graph file)", c.get("capture_overhead_s")),
    ]
    if c.get("translate_s") is not None:
        out.append(("translate (frisky task specs)", c.get("translate_s")))
        out.append(("submit", c.get("submit_s")))
        out.append(("submitted_to_first_dispatch", c.get("submitted_to_first_dispatch_s")))
    else:
        out.append(("captured_to_first_dispatch", c.get("captured_to_first_dispatch_s")))
    out.append(("construction_total (entry -> first dispatch)", c.get("construction_s")))
    if first is not None and start is not None:
        out.append(("first_dispatch -> this key started", round((start - first) / 1e9, 6)))
    if entry is not None and start is not None:
        out.append(("entry -> this key started", round((start - entry) / 1e9, 6)))
    return out


def _tree(
    trace: GroupTrace, key: str, *, depth: int, seen: set[str], prefix: str = ""
) -> list[str]:
    row = trace.rows.get(key) or {}
    node = trace.nodes.get(key) or {}
    when = (
        f"ran {_fmt_s(row.get('duration_ns'))} on {row.get('worker') or row.get('thread') or '-'}"
        if row.get("start_ns") is not None
        else f"not executed: {row.get('not_executed_reason') or 'unknown'} ({row.get('state')})"
    )
    lines = [f"{prefix}{key}  [{node.get('class', '?')}]  {when}"]
    if depth <= 0 or key in seen:
        return lines
    seen.add(key)
    deps = node.get("deps", [])
    for i, dep in enumerate(deps):
        last = i == len(deps) - 1
        branch = "└─ " if last else "├─ "
        child_prefix = prefix.replace("├─ ", "│  ").replace("└─ ", "   ") + branch
        lines.extend(_tree(trace, dep, depth=depth - 1, seen=seen, prefix=child_prefix))
    return lines


def explain_task(trace: GroupTrace, key: str, *, depth: int = 3) -> str:
    """Render one task: its row, its dependency tree, its dependents, its delay."""
    if key not in trace.nodes:
        candidates = [k for k in trace.nodes if key in k]
        if len(candidates) == 1:
            key = candidates[0]
        else:
            hint = "; ".join(candidates[:5])
            return f"no node {key!r} in group {trace.group}; candidates: {hint or 'none'}"
    row = trace.rows.get(key) or {}
    out = io.StringIO()
    g = trace.group
    print(f"task {key}", file=out)
    print(
        f"  group {g[0] if g else '?'}/{g[1] if g else '?'}  class {trace.nodes[key].get('class')}  prefix {trace.nodes[key].get('prefix')}",
        file=out,
    )
    if row.get("start_ns") is not None:
        print(
            f"  started {row['start_ns']}  ended {row.get('end_ns')}  duration {_fmt_s(row.get('duration_ns'))}"
            f"  worker {row.get('worker') or '-'}  thread {row.get('thread') or '-'}"
            f"  gil {_fmt_s(row.get('gil_ns'))}  deserialize {_fmt_s(row.get('deserialize_ns'))}",
            file=out,
        )
        if row.get("received_ns") is not None:
            print(
                f"  queued on worker for {_fmt_s(row['start_ns'] - row['received_ns'])} before it started",
                file=out,
            )
    else:
        print(
            f"  not executed: {row.get('not_executed_reason') or 'unknown'}  scheduler state {row.get('state')}",
            file=out,
        )
    print("  delay before this task, from dask.compute entry:", file=out)
    for name, seconds in delay_breakdown(trace, key):
        print(f"    {name:<48} {'-' if seconds is None else f'{seconds:.4f}s'}", file=out)
    print(f"  dependencies (depth {depth}):", file=out)
    for line in _tree(trace, key, depth=depth, seen=set()):
        print(f"    {line}", file=out)
    dependents = trace.dependents(key)
    print(
        f"  dependents ({len(dependents)}): {', '.join(dependents[:8])}{' ...' if len(dependents) > 8 else ''}",
        file=out,
    )
    return out.getvalue()


def explain_group(trace: GroupTrace) -> str:
    """Render the group: counts by class, construction, top prefixes, non-executed."""
    out = io.StringIO()
    g = trace.group
    print(
        f"group {g[0] if g else '?'}/{g[1] if g else '?'}: {trace.graph.get('n_tasks')} tasks, {trace.graph.get('n_edges')} edges",
        file=out,
    )
    by_class: dict[str, int] = {}
    for node in trace.nodes.values():
        by_class[node.get("class", "other")] = by_class.get(node.get("class", "other"), 0) + 1
    print(
        "  nodes by class: " + ", ".join(f"{k} {v}" for k, v in sorted(by_class.items())), file=out
    )
    c = trace.execution.get("construction") or {}
    print(
        f"  construction {c.get('construction_s')}s = pre_scheduler {c.get('pre_scheduler_s')} + materialize {c.get('materialize_s')} + capture {c.get('capture_overhead_s')}"
        + (
            f" + translate {c.get('translate_s')} + submit {c.get('submit_s')} + submitted_to_first_dispatch {c.get('submitted_to_first_dispatch_s')}"
            if c.get("translate_s") is not None
            else f" + captured_to_first_dispatch {c.get('captured_to_first_dispatch_s')}"
        ),
        file=out,
    )
    print(
        f"  executed {trace.execution.get('n_executed')}  not executed {trace.execution.get('n_not_executed')}  stories queried {trace.execution.get('stories_queried')}",
        file=out,
    )
    totals: dict[str, float] = {}
    for key, row in trace.rows.items():
        if row.get("duration_ns") is None:
            continue
        prefix = trace.nodes.get(key, {}).get("prefix", "?")
        totals[prefix] = totals.get(prefix, 0.0) + row["duration_ns"] / 1e9
    print("  top prefixes by summed task seconds:", file=out)
    for prefix, seconds in sorted(totals.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {prefix:<56} {seconds:8.3f}s", file=out)
    reasons: dict[str, int] = {}
    for row in trace.rows.values():
        if row.get("start_ns") is None:
            reasons[row.get("not_executed_reason") or "unknown"] = (
                reasons.get(row.get("not_executed_reason") or "unknown", 0) + 1
            )
    if reasons:
        print(
            "  not executed, by scheduler-evidenced reason: "
            + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())),
            file=out,
        )
    return out.getvalue()


def pick_output_key(trace: GroupTrace) -> str | None:
    """A representative key: the first output, else the longest-running task."""
    outputs = trace.graph.get("outputs") or []
    for key in outputs:
        if key in trace.nodes:
            return key
    best = max(
        (r for r in trace.rows.values() if r.get("duration_ns") is not None),
        key=lambda r: r["duration_ns"],
        default=None,
    )
    return best["key"] if best else None


def critical_path(trace: GroupTrace, key: str) -> list[str]:
    """Walk from ``key`` down its latest-finishing dependency until a root."""
    path = [key]
    seen = {key}
    current = key
    while True:
        deps = [d for d in trace.nodes.get(current, {}).get("deps", []) if d not in seen]
        if not deps:
            return path
        current = max(deps, key=lambda d: (trace.rows.get(d) or {}).get("end_ns") or 0)
        seen.add(current)
        path.append(current)


def sections_table(final: Mapping[str, Any]) -> Iterable[str]:
    """Rows for the final object's sections, grouped by longitude group."""
    for section in final.get("sections", []):
        group = section.get("group")
        label = f"group {group[0]}/{group[1]}" if group else section.get("name")
        yield f"{label:<28} {section.get('s') if section.get('s') is not None else '-':>10}  {section.get('error') or ''}"


def render_sections(sections: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(sections_table({"sections": list(sections)}))
