"""Command-line interface for Landsat LST pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from rashid import Report

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Landsat Land Surface Temperature annual composites."""
    # Plain tracebacks, before anything can raise. structlog's default console
    # renderer formats exceptions through rich with frame locals on, and one
    # such traceback rendered 3.8 MB of deserialized STAC collection into a
    # tile's log and evicted its phase history. See landsat_lst.logging_config.
    from landsat_lst.logging_config import configure_logging

    configure_logging()


#: Shared by every command that runs the offset pass. Off means neither read
#: nor write, so a run meant to validate the estimator cannot be served its own
#: stale answer and cannot overwrite a good record with a suspect one.
_offset_cache_option = click.option(
    "--no-offset-cache",
    is_flag=True,
    help="Recompute the per-scene offsets instead of reading them from "
    "_offsets/, and do not write what this run computes. Use when validating a "
    "change to the estimator itself; everything downstream of it reuses the "
    "cache safely, because the key covers every input that moves the result.",
)


def _build_jobs(
    year: int | None,
    end_year: int | None,
    tiles: tuple[str, ...],
    max_scenes: int | None = None,
) -> list:
    """Resolve CLI options into the job list to run.

    Omitting ``--year`` selects the production window. Passing ``--year`` alone
    keeps the single-year behavior that predates multi-year windows.
    """
    from landsat_lst.job import DEFAULT_WINDOW, generate_jobs
    from landsat_lst.models import ProcessingJob
    from landsat_lst.tiling import LAND_TILES, parse_tile_name

    if year is None:
        year, end_year = DEFAULT_WINDOW

    if tiles:
        for tile in tiles:
            if tile not in LAND_TILES:
                console.print(f"[red]Warning: {tile} is not in the land tiles set[/red]")
        return [
            ProcessingJob(
                tile=parse_tile_name(tile),
                year=year,
                end_year=end_year,
                max_scenes=max_scenes,
            )
            for tile in tiles
        ]

    if max_scenes is not None:
        msg = "--max-scenes is for exercising the machinery on named tiles; pass --tile."
        raise click.UsageError(msg)

    if end_year:
        return generate_jobs(window=(year, end_year))
    return generate_jobs([year])


@main.command()
@click.option("--year", "-y", type=int, help="Start year. Omit to use the default window.")
@click.option("--end-year", type=int, help="End year (inclusive) for a multi-year window")
@click.option(
    "--tile",
    "-t",
    "tiles",
    type=str,
    multiple=True,
    help="Tile to process (e.g., N40W075); repeatable",
)
@click.option("--dry-run", is_flag=True, help="Show what would be processed without running")
@click.option("--force", "-f", is_flag=True, help="Reprocess even if the COGs exist")
@click.option(
    "--distributed",
    "-d",
    is_flag=True,
    help="Submit to Coiled Batch, one VM per tile, instead of running locally",
)
@click.option(
    "--wait",
    is_flag=True,
    help="With --distributed, block until the run finishes and reconcile it",
)
@click.option(
    "--run-id",
    default=None,
    help="Run token. Set by the batch task so each VM reports into the same run.",
)
@click.option("--limit", type=int, help="Process at most N tiles from the job list")
@click.option(
    "--max-scenes",
    type=int,
    help="Keep at most N scenes, sampled evenly across the window. Exercises the "
    "machinery at tile geometry in minutes; output is a sample, not the product, "
    "and is written under a -sampleN window so it cannot overwrite a real tile.",
)
@_offset_cache_option
def process(
    *,
    year: int | None,
    end_year: int | None,
    tiles: tuple[str, ...],
    dry_run: bool,
    force: bool,
    distributed: bool,
    wait: bool,
    run_id: str | None,
    limit: int | None,
    max_scenes: int | None,
    no_offset_cache: bool,
) -> None:
    """Process Landsat data to COG composites.

    With no --year, processes the production window (2021-2025). Passing --year
    alone builds a single-year composite; add --end-year for a custom window.
    With --distributed, tiles are submitted to Coiled Batch and this command
    returns immediately; run `landsat-lst reconcile RUN_ID` afterwards to write
    the manifest, or pass --wait to do both in one go.
    """
    # The capture opens before the jobs are built, so an unusable --tile is
    # explained by an uploaded log rather than by silence on a dead VM.
    capture, attempt = _task_log(tiles, run_id)
    with capture:
        jobs = _build_jobs(year, end_year, tiles, max_scenes)
        _profile_sampled_run(max_scenes)
        if limit is not None:
            jobs = jobs[:limit]
        console.print(f"[bold]Processing window {jobs[0].window_label}[/bold]")
        if tiles:
            console.print(f"  Tiles: {', '.join(tiles)}")
        else:
            console.print(f"  Tiles: {len(jobs)} land tiles")

        if force:
            console.print("  [yellow]Force mode: reprocessing existing COGs[/yellow]")

        if dry_run:
            console.print("[yellow]Dry run - no processing performed[/yellow]")
            mode = "on Coiled" if distributed else "locally"
            for job in jobs[:5]:
                console.print(f"    Would process {mode}: {job.tile.name} {job.window_label}")
            if len(jobs) > 5:
                console.print(f"    ... and {len(jobs) - 5} more")
            return

        if distributed:
            _process_distributed(
                jobs,
                force=force,
                run_id=run_id,
                wait=wait,
                use_offset_cache=not no_offset_cache,
            )
        else:
            _process_local(
                jobs,
                force=force,
                run_id=run_id,
                use_offset_cache=not no_offset_cache,
                attempt=attempt,
            )


def _profile_sampled_run(max_scenes: int | None) -> None:
    """Turn per-task profiling on for a sampled run, unless the operator spoke.

    ``--max-scenes`` means a sample, and a sample exists to be measured: it
    writes no product, it runs in minutes, and the question it was started to
    answer is nearly always which operation owns the wall clock. Leaving the
    dump behind a second flag is how the run that mattered on 2026-08-14
    produced one only because ``LST_PROFILE_DASK`` happened to be set by hand.

    ``settings.profile_dask``'s own docstring reasoning still holds for a
    700-tile build, and that path passes no ``--max-scenes``, so it is untouched.
    ``profile_dask_cache`` stays gated on its own: it retains a record per task,
    and a sampled de-striping graph still reaches hundreds of thousands.

    An explicit ``LST_PROFILE_DASK`` wins either way. The environment is read
    directly rather than through ``settings.model_fields_set``: pydantic adds a
    field to that set on plain attribute assignment, so anything that toggled
    ``settings.profile_dask`` at runtime would read as an operator decision.
    """
    import os

    from landsat_lst.config import settings

    spoken = any(key.upper() == "LST_PROFILE_DASK" for key in os.environ)
    if max_scenes is None or spoken:
        return
    settings.profile_dask = True
    console.print(
        f"  [dim]Sampled run ({max_scenes} scenes): dask profiling on. "
        f"Set LST_PROFILE_DASK=0 to opt out.[/dim]"
    )


def _task_log(tiles: tuple[str, ...], run_id: str | None):
    """Capture this process's output, and settle which attempt this is.

    Returns the capture context and the attempt number, or ``None`` for a run
    that is not a batch task.

    Coiled keeps a task's stdout on the VM and reports the tee wrapper's exit
    code rather than the pipeline's, so a tile that dies explains itself only if
    it uploads its own log. Anything else -- a local run, a multi-tile sweep --
    already has its output in front of somebody, and is left alone.

    Keyed off the raw ``--tile`` argument rather than a built job, so that a
    task which dies *building* its jobs still uploads a log. That was the one
    failure mode with no evidence at all: a malformed tile argument raised in
    :func:`_build_jobs`, before any capture existed, and the task died in 0.6s
    having written nothing. The name is sanitized because the argument reaching
    this point has not been validated.
    """
    from contextlib import nullcontext

    if run_id is None or len(tiles) != 1:
        return nullcontext(), None

    from landsat_lst.progress import capture_task_log
    from landsat_lst.runs import resolve_attempt
    from landsat_lst.storage import get_storage

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tiles[0]) or "unnamed-tile"
    storage = get_storage()
    # Resolved once, here, and threaded down. Every artifact this process
    # writes has to carry the same attempt number, and the log is uploaded
    # last: a second caller asking the bucket again would see this process's
    # own state object and number itself one higher.
    attempt = resolve_attempt(storage, run_id, safe)
    return capture_task_log(run_id=run_id, tile=safe, storage=storage, attempt=attempt), attempt


def _process_local(
    jobs: list,
    *,
    force: bool,
    run_id: str | None = None,
    use_offset_cache: bool = True,
    attempt: int | None = None,
) -> None:
    """Process jobs sequentially in this process with a progress bar.

    This is also the code path a Coiled Batch VM runs, with one tile and a
    ``run_id``. The progress bar renders to the task's log; the run record it
    writes for that tile is what reconciliation reads.
    """
    from landsat_lst.job import process_tile_job

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {len(jobs)} tiles...", total=len(jobs))

        completed = 0
        skipped = 0
        failed = 0

        for job in jobs:
            progress.update(task, description=f"Processing {job.tile.name}...")
            result = process_tile_job(
                job,
                force=force,
                run_id=run_id,
                use_offset_cache=use_offset_cache,
                attempt=attempt,
            )

            if result.status == "completed":
                completed += 1
            elif result.status == "skipped":
                skipped += 1
            else:
                failed += 1
                console.print(f"[red]Failed: {job.tile.name} - {result.error}[/red]")

            progress.advance(task)

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Completed: [green]{completed}[/green]")
    console.print(f"  Skipped: [yellow]{skipped}[/yellow]")
    if failed:
        console.print(f"  Failed: [red]{failed}[/red]")
        # A batch task must exit non-zero on failure, or Coiled's retries and
        # the task state a manifest falls back on both report success.
        raise SystemExit(1)


def _print_results(results: list) -> None:
    """Render reconciled per-tile outcomes."""
    completed = sum(1 for r in results if r.status == "completed")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = [r for r in results if r.status == "failed"]

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Completed: [green]{completed}[/green]")
    console.print(f"  Skipped: [yellow]{skipped}[/yellow]")
    if failed:
        console.print(f"  Failed: [red]{len(failed)}[/red]")
        for r in failed:
            console.print(f"    [red]{r.job.tile.name}: {r.error}[/red]")


def _process_distributed(
    jobs: list, *, force: bool, run_id: str | None, wait: bool, use_offset_cache: bool = True
) -> None:
    """Submit jobs to Coiled Batch and hand back the run token.

    No per-tile progress bar and no held connection: `watch` is the live view,
    reading the heartbeats the tiles publish, and the manifest built by
    `reconcile` is the durable record. Closing this shell does not touch the
    run, and neither command needs it to have stayed open.
    """
    from landsat_lst.batch import reconcile_run, submit_batch, wait_for_batch
    from landsat_lst.config import settings

    console.print(
        f"  Region: {settings.coiled_region}  "
        f"VMs: up to {settings.coiled_max_workers}  "
        f"Types: {', '.join(settings.coiled_vm_types)}"
    )

    submission = submit_batch(jobs, force=force, run_id=run_id, use_offset_cache=use_offset_cache)
    console.print(f"  Run id: [bold]{submission.run_id}[/bold]")

    if submission.cluster_id is None:
        console.print("  [yellow]Every tile is already complete; no cluster started.[/yellow]")
        _print_results(reconcile_run(submission.run_id))
        return

    # The cluster id stays, because it is the right handle for billing and for
    # Coiled support. Its dashboard does not: a batch task never registers with
    # the dask scheduler, so that page describes a scheduler this run never
    # joined, and `coiled logs` never receives task stdout either. See ADR-010.
    console.print(f"  Cluster: {submission.cluster_id}  Tasks: {len(submission.submitted_tiles)}")

    if not wait:
        from rich.panel import Panel

        console.print(
            Panel(
                f"Submitted. This shell is free to close.\n\n"
                f"Live view:   [bold]landsat-lst watch {submission.run_id}[/bold]\n"
                f"Explain one: [bold]landsat-lst explain {submission.run_id} <tile>[/bold]\n"
                f"When done:   [bold]landsat-lst reconcile {submission.run_id}[/bold]",
                title=f"Run {submission.run_id}",
                title_align="left",
            )
        )
        return

    console.print("\n  Waiting for the run to finish (Ctrl-C is safe; the run continues)...")
    state = wait_for_batch(submission.run_id)
    console.print(f"  Final job state: {state or 'unknown'}")
    _print_results(reconcile_run(submission.run_id))
    console.print(f"  Manifest: {settings.manifest_dir / (submission.run_id + '.json')}")


def _window_options(command):
    """The window and sampling options every per-tile phase command shares."""
    for option in reversed(
        [
            click.option(
                "-t",
                "--tile",
                "tiles",
                multiple=True,
                required=True,
                help="Tile to run (e.g. N40W075); repeatable",
            ),
            click.option(
                "-y",
                "--year",
                type=int,
                default=None,
                help="Start year. Omit for the production window.",
            ),
            click.option("--end-year", type=int, default=None, help="End year (inclusive)"),
            click.option(
                "--max-scenes",
                type=int,
                default=None,
                help="Keep at most N scenes, sampled evenly across the window.",
            ),
        ]
    ):
        command = option(command)
    return command


@main.command()
@_window_options
@_offset_cache_option
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Recompute and overwrite the stored offsets even on a cache hit",
)
def offsets(
    *,
    tiles: tuple[str, ...],
    year: int | None,
    end_year: int | None,
    max_scenes: int | None,
    no_offset_cache: bool,
    force: bool,
) -> None:
    """Estimate one tile-window's per-scene offsets and persist them.

    The offset pass is the longest compute in a tile and its result is a few
    kilobytes: 26.9 minutes and 598,604 dask tasks for roughly 600 float64
    values on the 300-scene N40W075 sample. Running it here writes those values
    under `_offsets/`, so the next `process` on the same scene set skips it. See
    issue #77 item 2.

    Two things this is for. Warming the cache before a batch of experiments that
    cannot change the offsets, which is most of them: the rejection cap, both
    sparse floors, the land mask, the encoding, the COG writer. And reading a
    tile's rejection fraction on its own, which is the number
    `destripe_max_offset_c` was calibrated against on mid-latitude cropland and
    the one worth checking before trusting it on a humid tropical tile.

    Nothing here is written to the published prefix, so this never produces or
    overwrites a tile's COGs.
    """
    from landsat_lst.pipeline import compute_tile_offsets

    jobs = _build_jobs(year, end_year, tiles, max_scenes)
    console.print(f"[bold]Offsets for window {jobs[0].window_label}[/bold]")

    for job in jobs:
        estimate = compute_tile_offsets(job, use_offset_cache=not no_offset_cache, refresh=force)
        source = "[green]cached[/green]" if estimate.cached else "computed"
        diagnostics = estimate.diagnostics
        console.print(
            f"  {job.tile.name}  {estimate.scenes} scenes  {source} in {estimate.duration_s:.0f}s"
        )
        console.print(
            f"    kept      {int(diagnostics['n_kept'])}/{int(diagnostics['n_scenes'])}  "
            f"rejected {100 * estimate.rejected_frac:.1f}%"
        )
        if "p50" in diagnostics:
            console.print(
                f"    offset    p1 {diagnostics['p1']:+.2f}  p50 {diagnostics['p50']:+.2f}  "
                f"p99 {diagnostics['p99']:+.2f}  std {diagnostics['std']:.2f} degC"
            )
        console.print(f"    key       {estimate.key.storage_key}")

    # The cap was fitted at Pergamino, where 21.8 percent of scenes were
    # rejected. A tile far from that is telling you something about either the
    # cap or the climate, and it is cheaper to notice here than in a composite.
    console.print(
        "\n  [dim]Calibration reference: 21.8% rejected at Pergamino "
        "(mid-latitude cropland, 2021-2025). A sampled window rejects more, "
        "because a thinner climatology inflates apparent offsets.[/dim]"
    )


@main.command()
@_window_options
@_offset_cache_option
@click.option("--force", "-f", is_flag=True, help="Rebuild even if the COGs exist")
def composite(
    *,
    tiles: tuple[str, ...],
    year: int | None,
    end_year: int | None,
    max_scenes: int | None,
    no_offset_cache: bool,
    force: bool,
) -> None:
    """Build and export one tile-window's COGs, reading any cached offsets.

    The single-tile companion to `process`, which is the fleet driver: no Coiled
    path, no job list, no progress bar over tiles. Reach for this when iterating
    on one tile, and pair it with `offsets` to pay the estimator once across
    however many attempts follow.

    Skipping is by output, not by input: a tile whose COGs already exist is left
    alone unless --force. That check answers "did this tile produce output?" and
    deliberately not "is that output current" -- the offset cache is where input
    identity is enforced, because its key covers the scene set and the settings
    that shape the estimate. See issue #77 item 1.
    """
    from landsat_lst.job import process_tile_job

    jobs = _build_jobs(year, end_year, tiles, max_scenes)
    console.print(f"[bold]Compositing window {jobs[0].window_label}[/bold]")

    failed = 0
    for job in jobs:
        result = process_tile_job(job, force=force, use_offset_cache=not no_offset_cache)
        if result.status == "completed":
            console.print(
                f"  [green]OK[/green] {job.tile.name}  {result.scene_count} scenes  "
                f"{result.duration_s:.0f}s  peak {result.peak_rss_mb or 0:.0f} MB"
            )
            console.print(f"       {result.lst_key}")
            console.print(f"       {result.qa_key}")
        elif result.status == "skipped":
            console.print(
                f"  [yellow]skipped[/yellow] {job.tile.name}: COGs exist (--force to rebuild)"
            )
        else:
            failed += 1
            console.print(f"  [red]FAIL[/red] {job.tile.name}: {result.error}")

    if failed:
        raise SystemExit(1)


def _print_plan_phases(phases: tuple, vm_gib: float) -> None:
    """Render one configuration: what each phase builds and what it needs."""
    for phase in phases:
        peak = phase.peak
        graph = phase.graph
        verdict = "[green]fits[/green]" if peak.fits_in(vm_gib) else "[red]over[/red]"
        console.print(f"\n  [bold]{phase.name}[/bold]  {phase.height}x{phase.width} px")
        if graph.optimized:
            # Fused count: what the scheduler runs, so it lines up with the
            # fraction `landsat-lst watch` shows for a live tile.
            console.print(
                f"    tasks     {graph.tasks:,} after fusion "
                f"({graph.raw_tasks:,} raw, {graph.fusion:.2f}x), "
                f"{graph.blocks:,} output blocks"
            )
        else:
            console.print(
                f"    tasks     {graph.raw_tasks:,} raw, unfused -- not comparable "
                f"to a heartbeat; {graph.blocks:,} output blocks"
            )
        top = "  ".join(f"{s.prefix} {s.tasks:,}" for s in phase.graph.top(5))
        console.print(f"    heaviest  {top}")
        # Square brackets are rich's markup delimiters, so a literal label has
        # to be escaped or the console silently swallows it.
        terms = [f"stack {peak.stack_bytes / (1024**3):.1f}"]
        if peak.climatology_bytes:
            terms.append(f"climatology {peak.climatology_bytes / (1024**3):.1f}")
        terms.append(rf"baseline {peak.baseline_bytes / (1024**3):.1f} \[assumed]")
        console.print(
            rf"    memory    floor {peak.total_gib:.1f} GiB {verdict} in {vm_gib:.0f} GiB "
            rf"\[derived]  ({', '.join(terms)})"
        )
        _print_calibration(phase, vm_gib)


def _print_calibration(phase, vm_gib: float) -> None:
    """State what a real run measured against this phase, or that none has.

    Issue #77 item 3: an unvalidated term says so where it is printed, not only
    in a docstring. A floor with no measurement behind it is useful for ruling a
    configuration out and useless for sizing one, and the output has to say
    which of those it is offering.
    """
    from landsat_lst.calibration import peak_residuals, throughput_for, wall_time_minutes

    # Rates are recorded per phase: the composite retires a different mix of
    # work, so the offset pass's rate says nothing about it. No record for this
    # phase means no wall-time line at all, rather than a transferred guess.
    rate = throughput_for(vm_type="r6i.4xlarge", threads=phase.peak.threads, phase=phase.name)
    if rate is not None and phase.graph.optimized:
        minutes = wall_time_minutes(phase.graph.tasks, rate)
        console.print(
            rf"    wall      ~{minutes:.0f} min at {rate.tasks_per_second:.0f} tasks/s "
            rf"\[measured: {rate.vm_type}, {rate.threads} threads, {rate.measured_on}]"
        )

    # Laptop numbers are kept in the file but excluded here: the gap between
    # synthetic and production hardware is the open discrepancy, not evidence.
    observed = peak_residuals(phase=phase.name, exclude_vm="laptop")
    if not observed:
        console.print(
            r"    residual  [red]no real run measured against this phase[/red]; "
            r"the floor rules a configuration out, it does not size one \[unvalidated]"
        )
        return
    worst = max(observed, key=lambda r: r.measured_peak_gib)
    ratio = worst.measured_peak_gib / phase.peak.total_gib if phase.peak.total_gib else 0.0
    fits = "[green]fits[/green]" if worst.measured_peak_gib < vm_gib else "[red]over[/red]"
    console.print(
        rf"    residual  a real run peaked at {worst.measured_peak_gib:.1f} GiB, "
        rf"{ratio:.1f}x this floor {fits} in {vm_gib:.0f} GiB "
        rf"\[measured: {worst.vm_type}, {worst.threads} threads, "
        rf"{worst.scenes} scenes, {worst.measured_on}]"
    )


def _print_sweep(rows: tuple) -> None:
    """Render the sweep table: one line per configuration, cheapest first."""
    label = "tasks" if rows and rows[0].optimized else "raw tasks"
    console.print(
        f"\n  {'chunk':>6}{'threads':>9}{'offset ' + label:>20}"
        f"{'composite ' + label:>23}{'floor GiB':>12}{'':>7}"
    )
    console.print("  " + "-" * 77)
    for row in rows:
        verdict = "[green]fits[/green]" if row.fits else "[red]over[/red]"
        console.print(
            f"  {row.chunk_size:>6}{row.threads:>9}{row.offsets_tasks:>20,}"
            f"{row.composite_tasks:>23,}{row.floor_gib:>12.1f}   {verdict}"
        )


@main.command()
@click.option("-t", "--tile", "tile_name", required=True, help="Tile to plan, e.g. N40W075")
@click.option(
    "--scenes",
    type=int,
    default=None,
    help="Scenes to plan against (default: 300, the validation sample). "
    "A five-year tile pulls 2,930; planning that many takes over 15 minutes.",
)
@click.option("--chunk", type=int, default=None, help="Spatial chunk edge in px")
@click.option("--threads", type=int, default=None, help="Concurrent dask threads")
@click.option(
    "--offset-factor",
    type=int,
    default=None,
    help="Resolution factor for the offset pass, to price it before changing "
    "the default. Task count falls as factor**2.",
)
@click.option("--vm-gib", type=float, default=None, help="VM memory to judge against")
@click.option("--sweep", is_flag=True, help="Cross chunk size with thread count instead")
@click.option(
    "--fast",
    is_flag=True,
    help="Skip graph fusion. Much quicker, but the counts stop matching a heartbeat.",
)
@click.option(
    "--max-tasks",
    type=int,
    default=None,
    help="Refuse to build a graph estimated above this many raw tasks.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the plan as JSON")
def plan(
    *,
    tile_name: str,
    scenes: int | None,
    chunk: int | None,
    threads: int | None,
    offset_factor: int | None,
    vm_gib: float | None,
    sweep: bool,
    fast: bool,
    max_tasks: int | None,
    as_json: bool,
) -> None:
    """Price a tile's dask graphs without loading a pixel or starting a VM.

    Task count follows from array shape and chunking, never from pixel values,
    so both of a tile's graphs can be built against synthetic data on a laptop.
    That is the difference between reading 598,604 tasks here and learning it
    sixty seconds into a cloud run. See issue #76.

    Task counts are taken after fusion, so they match the fraction
    `landsat-lst watch` shows for a live tile. The unfused graph is a poor
    stand-in: for a 300-scene N40W075 offset pass it holds 905,923 tasks where
    the run reported 598,604, and fusion is not a constant factor to divide out.

    Reported memory is a **floor**, not a forecast: the concurrent block stacks,
    the resident monthly climatology, and a process baseline. A configuration
    that cannot fit the floor is disqualified for free. One that fits may still
    OOM, which is what `scripts/synthetic_scaling.py` measures.

    Planning defaults to 300 scenes, the validation sample, and takes about half
    a minute. A five-year tile pulls 2,930, and planning that many is not a
    laptop operation: the composite's quantile rechunks 293 time chunks into one
    across 1,296 spatial blocks, and one attempt ran past fifteen minutes before
    being killed. Building a graph allocates Python objects whether or not you
    ever compute it, so --max-tasks refuses a configuration that would exhaust
    the machine. A --sweep pays that cost once per chunk size, and --fast skips
    fusion; a swept comparison survives --fast intact, since ranking turns on
    the memory floor, which is exact either way.
    """
    import json as json_module

    from landsat_lst.profiling import (
        DEFAULT_PLAN_SCENES,
        DEFAULT_VM_GIB,
        MAX_PLAN_TASKS,
        SWEEP_CHUNK_SIZES,
        PlanTooLarge,
        plan_tile,
        sweep_plan,
    )
    from landsat_lst.tiling import LAND_TILES, parse_tile_name

    if tile_name not in LAND_TILES:
        # On stderr, so `--json` output stays pipeable into jq either way.
        click.echo(f"Note: {tile_name} is not in the land tiles set", err=True)
    tile = parse_tile_name(tile_name)
    n_scenes = DEFAULT_PLAN_SCENES if scenes is None else scenes
    vm = DEFAULT_VM_GIB if vm_gib is None else vm_gib
    ceiling = MAX_PLAN_TASKS if max_tasks is None else max_tasks

    if sweep:
        rows = sweep_plan(
            tile=tile, scenes=n_scenes, vm_gib=vm, optimize=not fast, max_tasks=ceiling
        )
        if as_json:
            # Nothing but JSON on stdout, so the output pipes straight into jq.
            click.echo(json_module.dumps([r.as_dict() for r in rows], indent=2))
            return
        console.print(f"[bold]Sweep {tile_name}[/bold]  {n_scenes:,} scenes  {vm:.0f} GiB VM")
        _print_sweep(rows)
        # Never let a dropped chunk size pass unmentioned: a table missing its
        # smallest chunk reads as though that chunk was considered and lost.
        planned = {r.chunk_size for r in rows}
        skipped = [c for c in SWEEP_CHUNK_SIZES if c not in planned]
        if skipped:
            console.print(
                f"\n  [yellow]Skipped chunk {', '.join(str(c) for c in skipped)}[/yellow]: "
                f"building the graph would allocate past the {ceiling:,}-task ceiling "
                f"at {n_scenes:,} scenes. Lower --scenes or raise --max-tasks."
            )
        return

    try:
        phases = plan_tile(
            tile=tile,
            scenes=n_scenes,
            chunk_size=chunk,
            threads=threads,
            offset_factor=offset_factor,
            optimize=not fast,
            max_tasks=ceiling,
        )
    except PlanTooLarge as e:
        raise click.ClickException(str(e)) from e
    if as_json:
        click.echo(json_module.dumps([p.as_dict() for p in phases], indent=2))
        return

    first = phases[0].peak
    console.print(
        f"[bold]Plan {tile_name}[/bold]  {n_scenes:,} scenes  "
        f"chunk {first.chunk_size}  threads {first.threads}"
    )
    _print_plan_phases(phases, vm)
    console.print(
        "\n  [dim]The floor counts concurrent block stacks, the monthly climatology "
        "de-striping holds resident, and a process baseline. It is a lower bound: "
        "read the residual line for what a real run did.[/dim]"
    )


#: Scenes this command will build locally without being told twice. The dev box
#: carries less memory than a production VM, and an unbounded local graph build
#: has taken a 64 GB desktop down before. Execution belongs on the VM; the
#: laptop gets the graph-inspection tier and a small smoke sweep.
LOCAL_SCENE_CEILING = 200


def _print_sweep_measurements(results: list) -> None:
    """One row per configuration, with the ratio the sweep exists to produce."""
    from rich.table import Table

    table = Table(box=None, pad_edge=False)
    for column, justify in (
        ("scenes", "right"),
        ("peak GB", "right"),
        ("floor GB", "right"),
        ("ratio", "right"),
        ("offset tasks", "right"),
        ("min", "right"),
    ):
        table.add_column(column, justify=justify)

    for m in results:
        if not m.ok:
            table.add_row(f"{m.geometry.scenes:,}", "[red]FAILED[/red]", "", "", "", "")
            continue
        table.add_row(
            f"{m.geometry.scenes:,}",
            f"{m.peak_rss_mb / 1024:.1f}",
            f"{m.floor_mb / 1024:.1f}",
            f"{m.peak_over_floor:.1f}",
            f"{m.offset_tasks:,}",
            f"{m.wall_s / 60:.1f}",
        )
    console.print(table)


#: What each verdict means for `plan`, stated once so a reader does not have to
#: re-derive it from the fit. These are the three outcomes issue #94 enumerates.
_VERDICTS = {
    "constant_ratio": (
        "The ratio holds across the sweep. Give predict_peak this correction "
        "factor and `landsat-lst plan` becomes predictive rather than one-sided."
    ),
    "growing_ratio": (
        "The ratio grows with scene count. Something scales that the model "
        "treats as fixed, which localizes the leak to the groupby shuffle or "
        "the anomaly broadcast. Direct evidence for issue #93."
    ),
    "not_streaming": (
        "Peak RSS barely moved: the stack still fits in RAM at this geometry, "
        "so dask never streams and there is no memory scaling to fit. Per "
        "ADR-011 no projection is printed. Raise --blocks or --scenes."
    ),
    "insufficient": "Too few configurations survived to fit a curve.",
}


def _follow_sweep(run_id: str, poll_s: float) -> None:
    """Stream a running sweep to this terminal until it settles.

    Appends as things happen rather than repainting, so scrollback keeps the
    whole run. Nothing here can be a true tail: Coiled keeps a task's stdout on
    the VM, and the only channel back is the object the sweep republishes after
    it starts each scene count and again after that point lands. So this polls
    that object and prints what is new, which at one line per transition is the
    real resolution of the underlying work.

    Ctrl-C detaches the terminal and leaves the VM running, because the sweep
    outlives this process by design.
    """
    import time

    from landsat_lst.benchmarks import benchmark_log_key, fetch_sweep

    seen: set = set()
    started_at = time.monotonic()
    waiting_printed = False

    while True:
        try:
            payload = fetch_sweep(run_id)
        except Exception as e:
            console.print(f"[yellow]  poll failed ({e}); retrying[/yellow]")
            payload = None

        if payload is None:
            if not waiting_printed:
                console.print("[dim]  waiting for the VM to start its first point...[/dim]")
                waiting_printed = True
        else:
            for line in _sweep_transitions(payload, seen):
                console.print(line)
            if payload.get("status") == "finished":
                console.print()
                _print_fetched_sweep(payload, run_id)
                return

        elapsed = time.monotonic() - started_at
        if elapsed > _FOLLOW_GIVE_UP_S:
            console.print(
                f"\n[yellow]Nothing new for {elapsed / 60:.0f} minutes. Detaching.[/yellow]\n"
                f"  The VM may still be working. Check {benchmark_log_key(run_id)}, "
                f"or re-attach with: landsat-lst benchmark --follow {run_id}"
            )
            return
        time.sleep(poll_s)


#: Stop following after this long with the sweep still unsettled. Comfortably
#: past SWEEP_JOB_TIMEOUT, so a live sweep is never abandoned early and a dead
#: one does not hold a terminal forever.
_FOLLOW_GIVE_UP_S = 75 * 60


def _sweep_transitions(payload: dict, seen: set) -> list[str]:
    """Lines for whatever changed since the last poll, in order."""
    lines = []

    # Completed points first, then whatever is now in flight. When two
    # transitions land in the same poll -- which they do whenever a point runs
    # faster than the poll interval -- the other order announces the next point
    # as starting before reporting the one it followed.
    for row in payload.get("measurements", []):
        scenes = row["geometry"]["scenes"]
        if ("done", scenes) in seen:
            continue
        seen.add(("done", scenes))
        seen.add(("start", scenes))
        if row.get("error"):
            first = str(row["error"]).splitlines()[-1][:80]
            lines.append(f"  [red]{scenes:>5} scenes: FAILED - {first}[/red]")
        else:
            lines.append(
                f"  [green]{scenes:>5} scenes:[/green] "
                f"{row['peak_rss_mb'] / 1024:.1f} GB peak, "
                f"{row['peak_over_floor']:.1f}x floor, "
                f"{row['offset_tasks']:,} tasks, {row['wall_s'] / 60:.1f} min"
            )

    in_flight = payload.get("in_flight")
    if in_flight is not None and ("start", in_flight) not in seen:
        seen.add(("start", in_flight))
        lines.append(f"  [dim]{in_flight:>5} scenes: running...[/dim]")

    return lines


def _print_fetched_sweep(payload: dict, run_id: str) -> None:
    """Render a published sweep, whether it has finished or is still working."""
    from landsat_lst.benchmarks import Geometry, Measurement, benchmark_log_key

    requested = payload.get("requested_scenes") or []
    completed = payload.get("completed", 0)
    status = payload.get("status", "unknown")

    colour = "green" if status == "finished" else "yellow"
    console.print(
        f"[bold]{run_id}[/bold]  [{colour}]{status}[/{colour}]  {completed}/{len(requested)} points"
    )

    results = []
    for row in payload.get("measurements", []):
        geometry = Geometry(**row.pop("geometry"))
        row.pop("peak_over_floor", None)
        row.pop("native_passes", None)
        results.append(Measurement(geometry=geometry, **row))
    if results:
        console.print()
        _print_sweep_measurements(results)

    report = payload.get("report") or {}
    verdict = str(report.get("verdict") or "unknown")
    if status != "finished":
        remaining = [n for n in requested if n not in [m.geometry.scenes for m in results]]
        console.print(
            f"\n  [dim]Still to run: {', '.join(str(n) for n in remaining)}. "
            f"Live output at {benchmark_log_key(run_id)}.[/dim]"
        )
        return

    console.print(f"\n[bold]Verdict: {verdict}[/bold]")
    console.print(f"  {_VERDICTS.get(verdict, '')}")
    if report.get("streaming_regime"):
        console.print(
            f"\n  At {report['target_scenes']:,} scenes this geometry projects to "
            f"{report['projected_peak_mb'] / 1024:.1f} GB, against a 64 GiB VM."
        )

    # The requirement this whole tier serves: what does a production tile
    # cost in wall clock and VM-hours, at the probe-measured rates.
    from landsat_lst.projection import tile_projection

    console.print("\n[bold]Production-tile projection[/bold]")
    for line in tile_projection().summary_lines():
        console.print(f"  {line}")

    console.print("\n  [dim]Write the outcome into docs/findings-memory-model.md.[/dim]")


@main.command()
@click.option("--scenes", multiple=True, type=int, help="Scene count to measure; repeatable")
@click.option("--blocks", type=int, default=8, help="Blocks per side, in chunks")
@click.option("--chunk", type=int, default=512, help="Spatial chunk edge in px")
@click.option("--threads", type=int, default=4, help="Concurrent dask threads")
@click.option(
    "--distributed",
    "-d",
    is_flag=True,
    help="Run the sweep on one Coiled VM of the production instance type",
)
@click.option("--fetch", "fetch_id", default=None, help="Read a published sweep back by run id")
@click.option(
    "--follow",
    "follow_id",
    default=None,
    help="Stream a running sweep to this terminal until it settles",
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="With --distributed, print the run id and return instead of streaming",
)
@click.option("--poll", "poll_s", type=float, default=20.0, help="Seconds between polls")
@click.option(
    "--run-id",
    default=None,
    help="Publish the result under this run id. Set by the VM; rarely useful locally.",
)
@click.option(
    "--force-local",
    is_flag=True,
    help=f"Build more than {LOCAL_SCENE_CEILING} scenes on this machine anyway",
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=None, help="Where to write the JSON"
)
def benchmark(
    *,
    scenes: tuple[int, ...],
    blocks: int,
    chunk: int,
    threads: int,
    distributed: bool,
    fetch_id: str | None,
    follow_id: str | None,
    no_follow: bool,
    poll_s: float,
    run_id: str | None,
    force_local: bool,
    out: Path | None,
) -> None:
    """Measure peak RSS against scene count on synthetic data at real geometry.

    `landsat-lst plan` reports a memory **floor**: the concurrent block stacks,
    the resident climatology, and a process baseline. A configuration that
    cannot fit the floor is disqualified for free, but one that fits may still
    OOM, and the size of that gap is what this measures. On the 300-scene
    N40W075 sample the floor landed far under the 78.6 GB actually observed.

    Run it on a VM, not here. The dev box carries less memory than the VM, so
    the ceiling under test is unreachable; the answer is about production
    hardware, which is the only hardware whose peak RSS matters; and synthetic
    data means the VM does no I/O, so the sweep is about 20 minutes and well
    under a dollar.

        landsat-lst benchmark --distributed     # submits, prints a run id
        landsat-lst benchmark --fetch <run-id>  # reads the result back

    The verdict is the deliverable, not the numbers. Write it up in
    docs/findings-memory-model.md whichever way it lands.
    """

    from landsat_lst.benchmarks import (
        DEFAULT_SWEEP_SCENES,
        benchmark_key,
        benchmark_log_key,
        fetch_sweep,
        submit_sweep,
        sweep,
        sweep_report,
    )

    counts = list(scenes) if scenes else list(DEFAULT_SWEEP_SCENES)

    if follow_id:
        _follow_sweep(follow_id, poll_s)
        return

    if fetch_id:
        payload = fetch_sweep(fetch_id)
        if payload is None:
            raise click.ClickException(
                f"Nothing published at {benchmark_key(fetch_id)} yet. The VM "
                "publishes after its first scene count, so this means the sweep "
                "has not finished one yet, or it died before it could. Its own "
                f"stdout is at {benchmark_log_key(fetch_id)}."
            )
        _print_fetched_sweep(payload, fetch_id)
        return

    if distributed:
        submission = submit_sweep(counts, blocks=blocks, chunk=chunk, threads=threads)
        run = submission["run_id"]
        console.print(f"[bold]Submitted[/bold] {run}")
        console.print(f"  Cluster: {submission['cluster_id']}")
        console.print(f"  Result:  {submission['key']}")
        console.print(f"  Log:     {benchmark_log_key(run)}")
        if no_follow:
            console.print(f"\n  Follow it:  landsat-lst benchmark --follow {run}")
            console.print(f"  Read once:  landsat-lst benchmark --fetch  {run}")
            return
        console.print("\n[dim]Following. Ctrl-C detaches; the VM keeps going.[/dim]\n")
        _follow_sweep(run, poll_s)
        return

    # The capture opens before anything can reject the arguments, so a task that
    # dies *validating* them still uploads a log. The first version wrapped only
    # the sweep loop, and the scene-ceiling rejection below then killed two VMs
    # in under a minute each having written nothing at all -- no result, no log,
    # nothing under _benchmarks/ to read. Same lesson, same fix, as the tile path
    # in _task_log.
    capture, publish = _sweep_publisher(run_id, counts, blocks, chunk, threads)

    with capture:
        # The ceiling protects an interactive machine from a graph build that has
        # taken a 64 GB desktop down. A batch task is not that machine: it passes
        # --run-id, it exists to run the points a laptop cannot, and the default
        # sweep's top two exceed the ceiling by design. Applying it there killed
        # the run this guard was written to make possible.
        over = [n for n in counts if n > LOCAL_SCENE_CEILING]
        if over and not force_local and run_id is None:
            raise click.UsageError(
                f"{', '.join(str(n) for n in over)} scenes is past the "
                f"{LOCAL_SCENE_CEILING}-scene local ceiling. Building a graph "
                "allocates Python objects whether or not you compute it, and an "
                "unbounded local build has taken a 64 GB desktop down. Use "
                "--distributed for the real sweep, or --force-local to override."
            )

        side = blocks * chunk
        console.print(
            f"[bold]Sweep[/bold] {side}x{side} px ({blocks**2} blocks of {chunk}), "
            f"{threads} threads"
        )

        results = sweep(
            counts,
            blocks=blocks,
            chunk=chunk,
            threads=threads,
            on_start=lambda g: (
                console.print(f"  [dim]{g.scenes:>5} scenes: running...[/dim]"),
                publish(None, starting=g.scenes),
            ),
            on_result=lambda m: (
                console.print(
                    f"  [dim]{m.geometry.scenes:>5} scenes: "
                    + (
                        f"FAILED - {m.error.splitlines()[-1][:80]}"
                        if not m.ok
                        else f"{m.peak_rss_mb / 1024:.1f} GB, {m.wall_s / 60:.1f} min"
                    )
                    + "[/dim]"
                ),
                publish(m),
            ),
        )

        console.print()
        _print_sweep_measurements(results)

        report = sweep_report(results)
        console.print(f"\n[bold]Verdict: {report['verdict']}[/bold]")
        console.print(f"  {_VERDICTS.get(report['verdict'], '')}")
        if report.get("streaming_regime"):
            projected = report["projected_peak_mb"] / 1024
            console.print(
                f"\n  At {report['target_scenes']:,} scenes this geometry projects to "
                f"{projected:.1f} GB, against a 64 GiB VM."
            )
            if not report["projected_fits_vm"]:
                console.print(
                    "  [yellow]That does not fit. Cut threads or chunk size and re-run, "
                    "or plan on a larger instance.[/yellow]"
                )

        text = publish(None, results=results, report=report)

    destination = out or Path("results/decision/synthetic_scaling.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    console.print(f"Wrote {destination}")


def _sweep_publisher(run_id, counts, blocks, chunk, threads):
    """A log capture and an incremental publisher for one sweep.

    Returns ``(capture, publish)``. Without a ``run_id`` there is nowhere to
    publish, so both are inert and the sweep behaves as a local command. With
    one, every measurement rewrites the same object, carrying ``status`` and
    ``completed`` so a reader can tell a sweep still working from one that
    settled, and so a crash at the fourth point leaves the first three.

    Publishing is best-effort, like every other instrument in this project: a
    failed write is reported and swallowed, because losing a progress update
    costs less than losing the sweep.
    """
    import json as json_module
    from contextlib import nullcontext

    from landsat_lst.benchmarks import benchmark_key, benchmark_log_key, sweep_report

    done: list = []

    in_flight: list = [None]

    def _payload(results, report) -> dict:
        return {
            "blocks": blocks,
            "chunk": chunk,
            "threads": threads,
            "requested_scenes": list(counts),
            "completed": len(results),
            # The scene count currently being measured. The top point of a
            # production sweep runs for twelve minutes, so without this a
            # follower cannot tell working from wedged.
            "in_flight": in_flight[0] if report is None else None,
            "status": "running" if report is None else "finished",
            "report": report if report is not None else sweep_report(results),
            "measurements": [m.as_dict() for m in results],
        }

    if run_id is None:

        def publish(measurement, *, results=None, report=None, starting=None) -> str:
            if starting is not None:
                in_flight[0] = starting
                return ""
            if measurement is not None:
                done.append(measurement)
                return ""
            return json_module.dumps(
                _payload(results if results is not None else done, report), indent=2
            )

        return nullcontext(), publish

    from landsat_lst.benchmarks import published_storage
    from landsat_lst.progress import capture_task_log

    # The same backend --follow reads. A VM always runs with
    # LST_STORAGE_BACKEND=s3 so this is S3 there either way; going through the
    # one helper keeps writer and reader from ever disagreeing.
    storage = published_storage()
    key = benchmark_key(run_id)

    def publish(measurement, *, results=None, report=None, starting=None) -> str:
        if starting is not None:
            in_flight[0] = starting
        if measurement is not None:
            done.append(measurement)
        settled = results if results is not None else done
        text = json_module.dumps(_payload(settled, report), indent=2)
        try:
            storage.write_text(key, text)
        except Exception as e:
            console.print(f"  [yellow]progress not published: {e}[/yellow]")
        return text

    capture = capture_task_log(
        run_id=run_id,
        tile="sweep",
        storage=storage,
        key=benchmark_log_key(run_id),
    )
    return capture, publish


@main.command()
@click.option("--tile", "-t", default="N40W075", help="Tile to cache")
@click.option("--year", "-y", type=int, default=2021, help="First year of the window")
@click.option("--end-year", type=int, default=2025, help="Last year, inclusive")
@click.option("--max-scenes", type=int, default=300, help="Scenes to keep, sampled evenly")
@click.option(
    "--factor",
    type=int,
    default=8,
    help="Resolution factor. Each doubling divides the stack by four. "
    "Production estimates offsets at 2, which for a five-degree tile is 97 GB.",
)
@click.option("--max-gb", type=float, default=None, help="Refuse a fetch larger than this")
@click.option("--force", "-f", is_flag=True, help="Refetch even if the fixture exists")
@click.option("--list", "show_list", is_flag=True, help="Show what is already cached")
@click.option("--dry-run", is_flag=True, help="Print the size arithmetic and stop")
def fixture(
    *,
    tile: str,
    year: int,
    end_year: int,
    max_scenes: int,
    factor: int,
    max_gb: float | None,
    force: bool,
    show_list: bool,
    dry_run: bool,
) -> None:
    """Cache a real tile's coarse stack locally, for accuracy work.

    Comparing two offset estimators means running both over the same scenes.
    Without a fixture that is a STAC query and hundreds of gigabytes of coarse
    reads per iteration, for an answer that is 600 floats. The first fetch is
    slow; every later one is a local memory-map.

    Size is arithmetic over the grid, so check it before you fetch:

    \b
        factor  2:  9000x9000  ->  97.2 GB    (production's offset grid)
        factor  4:  4500x4500  ->  24.3 GB
        factor  8:  2250x2250  ->   6.1 GB    (default)
        factor 16:  1125x1125  ->   1.5 GB

    Coarsening costs absolute accuracy, and this fixture answers a relative
    question: both estimators read the same pixels, so the comparison is exact
    at any factor. It cannot answer the memory question -- below the streaming
    regime the stack fits in RAM and dask never streams. Use
    `landsat-lst benchmark` for that.

    Fetch through Planetary Computer, which is free from a laptop. Earth Search
    is for AWS, where the read is same-region.
    """
    from landsat_lst.config import settings
    from landsat_lst.fixture import (
        DEFAULT_MAX_GB,
        FixtureSpec,
        build_fixture,
        estimate_bytes,
        grid_shape,
        list_fixtures,
    )

    if show_list:
        cached = list_fixtures()
        if not cached:
            console.print("[yellow]No fixtures built yet.[/yellow]")
            return
        for meta in cached:
            spec = FixtureSpec(**meta.spec)
            console.print(
                f"  {spec.name:<34} {meta.scene_count:>5} scenes  "
                f"{meta.bytes_on_disk / 1e9:>6.1f} GB  {meta.stac_url}"
            )
        return

    spec = FixtureSpec(
        tile=tile, year=year, end_year=end_year, max_scenes=max_scenes, factor=factor
    )
    height, width = grid_shape(spec)
    planned = estimate_bytes(spec)
    console.print(f"[bold]{spec.name}[/bold]")
    console.print(f"  Grid:   {height} x {width} px at factor {factor}")
    console.print(f"  Stack:  {max_scenes} scenes x 2 uint16 bands = {planned / 1e9:.1f} GB")
    console.print(f"  STAC:   {settings.stac_url}")
    console.print(f"  Path:   {spec.path}")

    if dry_run:
        return

    try:
        build_fixture(
            spec,
            max_gb=DEFAULT_MAX_GB if max_gb is None else max_gb,
            force=force,
            progress=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@main.command()
@click.argument("run_id")
def reconcile(run_id: str) -> None:
    """Build the run manifest for a submitted batch run.

    Reads the COG listing for what finished, the per-tile run records for
    duration, scene count, and peak memory, and Coiled's task states for why a
    tile without output has none. Safe to run more than once, and safe to run
    while tasks are still going: tiles with no COGs yet report as failed.
    """
    from landsat_lst.batch import reconcile_run
    from landsat_lst.config import settings

    console.print(f"[bold]Reconciling run {run_id}[/bold]")
    results = reconcile_run(run_id)
    _print_results(results)
    console.print(f"  Manifest: {settings.manifest_dir / (run_id + '.json')}")


@main.command()
@click.argument("run_id")
@click.option(
    "--interval",
    type=float,
    default=None,
    help="Seconds between polls (default from settings.watch_poll_interval_s)",
)
@click.option("--once", is_flag=True, help="Poll a single time and exit")
@click.option("--all", "show_all", is_flag=True, help="Show every tile, not only the live ones")
@click.option(
    "--detail",
    is_flag=True,
    help="Add a per-tile panel: memory trend, task rate, phase history, and cost",
)
def watch(run_id: str, interval: float | None, once: bool, show_all: bool, detail: bool) -> None:
    """Follow a running batch run's tiles from their heartbeats.

    Each tile republishes its phase, elapsed time, and memory every minute
    while it works, so a wedged or preempted tile shows a stale heartbeat
    within two minutes instead of looking identical to a busy one. Reads only
    the run's storage prefix, so it works from any machine, including one that
    did not submit the run.

    This reports liveness, not outcome: run `landsat-lst reconcile` afterwards
    for the verdict, which comes from the COGs in the bucket.

    It returns on its own once every submitted tile has stopped. From a machine
    without the run's submission record there is no tile list to check that
    against, so it keeps watching until you stop it; Ctrl-C is safe and leaves
    the run alone.
    """
    from landsat_lst.watch import watch_run

    try:
        snapshot = watch_run(
            run_id,
            interval_s=interval,
            once=once,
            show_all=show_all,
            detail=detail,
            console=console,
        )
    except KeyboardInterrupt:
        console.print("\n  Stopped watching. The run is untouched.")
        return

    if snapshot.finished:
        console.print(
            f"\n  Every tile has stopped. Next: [bold]landsat-lst reconcile {run_id}[/bold]"
        )


#: Log lines shown per attempt. The uploaded log is a tail already, capped at
#: ``settings.task_log_max_bytes``, so this is a tail of a tail: enough to carry
#: a traceback, short enough to read.
_LOG_TAIL_LINES = 40


def _explain_storage(run_id: str):
    """The backend the run wrote to, falling back to this machine's config."""
    from landsat_lst.batch import load_submission
    from landsat_lst.storage import get_storage

    try:
        return load_submission(run_id).storage()
    except Exception:
        return get_storage()


def _read_json(storage, key: str | None) -> dict | None:
    """One published object, or ``None`` if it is missing or unreadable."""
    import json as json_module

    if key is None:
        return None
    try:
        raw = storage.read_text(key)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return json_module.loads(raw)
    except ValueError:
        return None


def _print_attempt_state(state: dict, phase: str, status: str) -> None:
    """The attempt's own numbers, and where its time went."""
    from landsat_lst.render import format_duration, phase_rows, truncate

    elapsed = state.get("duration_s") or state.get("elapsed_s")
    console.print(
        f"  elapsed {format_duration(elapsed)}"
        f"   peak RSS {(state.get('peak_rss_mb') or 0) / 1024:.1f}G"
        f"   host {state.get('host', '-')}"
    )
    if state.get("instance_type"):
        console.print(
            f"  instance {state['instance_type']}"
            f" ({state.get('instance_lifecycle', 'unknown')},"
            f" {state.get('instance_source', 'unknown')})"
        )

    for name, duration, bar_text in phase_rows(
        state.get("phase_seconds") or {}, current=phase if status == "unsettled" else None
    ):
        console.print(f"    {name:<16}{duration:>8}  {bar_text}")

    if state.get("error"):
        console.print(f"  [red]{truncate(state['error'], 200)}[/red]")


def _print_attempt_profiles(storage, artifacts, attempt: int) -> None:
    """The dask profile dump, which nothing surfaced before this command."""
    from landsat_lst.render import format_duration

    for label, key in sorted(artifacts.profiles.get(attempt, {}).items()):
        profile = _read_json(storage, key) or {}
        tasks = profile.get("tasks", {})
        console.print(
            f"  profile {label}: {format_duration(profile.get('wall_s'))}"
            f" over {tasks.get('count', '?')} tasks"
        )
        for entry in (tasks.get("by_prefix") or [])[:5]:
            console.print(f"      {entry.get('prefix', '?'):<28}{entry.get('seconds', 0):>8.1f}s")


def _print_attempt_log(storage, artifacts, attempt: int) -> None:
    """The tail of the uploaded log, which is already a tail.

    Coiled never carries task stdout, so this object is the only place a failed
    tile explains itself.
    """
    from landsat_lst.render import strip_ansi, truncate

    log_key = artifacts.logs.get(attempt)
    if log_key is None:
        console.print("  [dim]no log uploaded[/dim]")
        return
    try:
        text = storage.read_text(log_key)
    except Exception:
        text = None
    if not text:
        console.print(f"  [dim]log at {log_key} is empty or unreadable[/dim]")
        return
    console.print(f"  [dim]log tail ({log_key}):[/dim]")
    for line in strip_ansi(text).splitlines()[-_LOG_TAIL_LINES:]:
        # markup=False: a captured log is data, and rich would read a bracketed
        # token in a traceback as a style tag.
        console.print(f"    {truncate(line, 160)}", style="dim", markup=False)


def _print_attempt(storage, artifacts, attempt: int) -> None:
    """Everything one attempt published, in the order it becomes useful."""
    state = _read_json(storage, artifacts.states.get(attempt)) or {}
    phase = state.get("phase", "unknown")
    status = state.get("status") or "unsettled"

    console.print(f"\n[bold]Attempt {attempt}[/bold]  {phase}  ({status})")
    _print_attempt_state(state, phase, status)
    _print_attempt_profiles(storage, artifacts, attempt)
    _print_attempt_log(storage, artifacts, attempt)


@main.command()
@click.argument("run_id")
@click.argument("tile", required=False)
def explain(run_id: str, tile: str | None) -> None:
    """Print everything one run, or one tile, published about itself.

    Diagnosing the STAC failure of 2026-08-14 took four manual steps: read the
    manifest, list the run prefix, download the task log, then slice its head
    and tail. Every input was already in `_runs/{run_id}/`. This reads that
    prefix and nothing else, so it works from any machine, including one that
    did not submit the run.

    With a tile, prints each attempt in turn: its state object, where its time
    went, the dask profile when one exists, and the tail of its log. A tile that
    was retried shows every attempt, because a tile that succeeded on the third
    try after two infrastructure failures has to read differently from one that
    succeeded on the first.
    """
    from landsat_lst.render import format_duration
    from landsat_lst.runs import classify

    storage = _explain_storage(run_id)
    found = classify(storage.list_prefix(storage.run_prefix(run_id)))
    if not found:
        console.print(f"[yellow]Nothing published under run {run_id}.[/yellow]")
        return

    if tile is None:
        console.print(f"[bold]Run {run_id}[/bold]  {len(found)} tiles reported")
        for name in sorted(found):
            artifacts = found[name]
            state = _read_json(storage, artifacts.body_key) or {}
            attempts = artifacts.attempts or [0]
            console.print(
                f"  {name:<10}{state.get('phase', 'unknown'):<16}"
                f"{format_duration(state.get('duration_s') or state.get('elapsed_s')):>8}"
                f"   attempts {len(attempts)}"
            )
        console.print(f"\n  Detail: [bold]landsat-lst explain {run_id} <tile>[/bold]")
        return

    artifacts = found.get(tile)
    if artifacts is None:
        console.print(f"[yellow]Tile {tile} published nothing under run {run_id}.[/yellow]")
        return

    console.print(f"[bold]{tile}[/bold] in run {run_id}")
    for attempt in artifacts.attempts or [0]:
        _print_attempt(storage, artifacts, attempt)


@main.command()
@click.option("-t", "--tile", "tiles", multiple=True, required=True, help="Tile to verify")
@click.option(
    "-y", "--year", type=int, default=None, help="Start year. Omit for the default window."
)
@click.option("--end-year", type=int, default=None, help="End year (inclusive)")
@click.option("--urls", is_flag=True, help="Print the access URLs for each verified tile")
def verify(*, tiles: tuple[str, ...], year: int | None, end_year: int | None, urls: bool) -> None:
    """Check published COGs open over public HTTPS with the right encoding.

    Answers the question a bucket listing cannot: does the person who pastes
    this URL into QGIS get a raster that decodes to Celsius? Each asset is
    opened unauthenticated over the public read host, so a tile that reads only
    with credentials fails here.

    Exits non-zero if any tile fails.
    """
    from landsat_lst.job import DEFAULT_WINDOW
    from landsat_lst.verify import verify_tile

    start = year or DEFAULT_WINDOW[0]
    if year is None:
        end_year = DEFAULT_WINDOW[1]
    window = str(start) if end_year is None or end_year == start else f"{start}-{end_year}"

    console.print(f"[bold]Verifying window {window}[/bold]")
    failed = 0
    for tile in tiles:
        check = verify_tile(tile, window)
        if check.ok:
            lst = next(a for a in check.assets if a.product == "lst_p95")
            shape = f"{lst.shape[0]}x{lst.shape[1]}" if lst.shape else "unknown shape"
            console.print(
                f"  [green]OK[/green] {tile}: {lst.dtype} {shape}, "
                f"nodata {lst.nodata}, scale {lst.scale}, offset {lst.offset}, "
                f"{len(lst.overviews)} overviews"
            )
        else:
            failed += 1
            for asset in check.assets:
                if not asset.ok:
                    console.print(f"  [red]FAIL[/red] {tile} {asset.product}: {asset.error}")
        if urls:
            for asset in check.assets:
                console.print(f"       {asset.url}")

    console.print(f"\n  Verified: [green]{len(tiles) - failed}[/green]")
    if failed:
        console.print(f"  Failed: [red]{failed}[/red]")
        raise SystemExit(1)


@main.command()
def list_tiles() -> None:
    """List all tiles in the global grid."""
    from landsat_lst.tiling import generate_global_tiles

    tiles = list(generate_global_tiles())
    console.print(f"[bold]Total tiles: {len(tiles)}[/bold]")

    for t in tiles[:10]:
        console.print(f"  {t.name}: {t.bbox}")

    if len(tiles) > 10:
        console.print(f"  ... and {len(tiles) - 10} more")


@main.command()
@click.argument("tile_name")
def tile_info(tile_name: str) -> None:
    """Show information about a specific tile."""
    console.print(f"[bold]Tile: {tile_name}[/bold]")
    console.print("[red]Not yet implemented[/red]")


@main.command()
@click.option(
    "--ged-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Granule archive to measure (default: settings.ged_dir)",
)
@click.option(
    "--artifact",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Measure this artifact's consumed manifest instead of a directory",
)
@click.option(
    "--fetch-domain",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="180x360 .npy of the 1-degree cells the archive's fetch requested. "
    "Refines *why* a granule is absent; it is not an upstream inventory.",
)
@click.option("--buffer-cells", type=int, default=None, help="Margin ring, default from settings")
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the machine-readable record here",
)
@click.option("--json", "as_json", is_flag=True, help="Print the record instead of a table")
def ged_coverage(
    *,
    ged_dir: Path | None,
    artifact: Path | None,
    fetch_domain: Path | None,
    buffer_cells: int | None,
    out: Path | None,
    as_json: bool,
) -> None:
    """Check a GED source against what all 700 production land tiles need.

    The expected manifest is local arithmetic over the production tile list,
    the global grid, the configured buffer, and the granule naming grammar --
    no network, no credentials, no fetching.

    \b
    Exits non-zero when the source cannot cover production, because
    `ged_gap_mask` defaults on and a tile that reaches an unheld granule
    fails rather than shipping unmasked.

    Absence is classified by what was *requested*, never by what the
    collection holds: no authoritative offline inventory of AG100 v003
    exists, so upstream absence cannot be established here.
    """
    import json as json_module

    from landsat_lst.ged_coverage import build_report

    report = build_report(
        ged_dir=ged_dir,
        artifact=artifact,
        buffer_cells=buffer_cells,
        fetch_domain=fetch_domain,
    )
    counts = report.counts()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_module.dumps(report.as_dict(), indent=2) + "\n")
    if as_json:
        click.echo(json_module.dumps(report.as_dict(), indent=2))
    else:
        _print_ged_coverage(report, counts)
        if out is not None:
            console.print(f"\n  Written to {out}")
    if not report.complete:
        raise SystemExit(1)


def _print_ged_coverage(report, counts: dict) -> None:
    """Render the completeness verdict."""
    source = report.fetch_domain_source
    console.print("[bold]ASTER GED coverage against the production tile list[/bold]")
    console.print(
        f"  {counts['tiles']} land tiles, buffer {report.buffer_cells} cell "
        f"-> {counts['expected']:,} granules expected"
    )
    console.print(
        f"  held: {counts['consumed_of_expected']:,}   "
        f"missing: {counts['missing']:,}   "
        f"(of which inside a tile, not its margin: {counts['missing_core']:,})"
    )
    console.print(
        f"  tiles that would fail: {counts['tiles_missing_core']:,} of {counts['tiles']}"
        f"   tiles missing any granule: {counts['tiles_missing_any']:,}"
    )
    console.print(f"  held but not expected: {counts['extra_not_expected']:,}")
    for key, value in sorted(counts.items()):
        if key.startswith("class_"):
            console.print(f"    {key[len('class_') :]:<32} {value:,}")
    if source is None:
        console.print(
            "  [dim]No fetch-domain grid given, so every absence is unverified-upstream.[/dim]"
        )
    else:
        console.print(f"  [dim]fetch domain: {source}[/dim]")
    console.print(
        "  [dim]No offline inventory of AG100 v003 exists, so none of these "
        "labels claims a granule is absent upstream.[/dim]"
    )
    if report.complete:
        console.print("\n[green]COMPLETE[/green] -- this source covers production.")
    else:
        console.print("\n[red]INCOMPLETE[/red] -- must not be packaged as the production mask.")


@main.command()
@click.option(
    "--raster",
    required=True,
    help="Published LST P95 COG: a local path, or an https/vsicurl URL. "
    "S30W065 is published at https://s3.us-west-2.amazonaws.com/"
    "us-west-2.opendata.source.coop/nlebovits/landsat-lst/"
    "lst-p95-2021-2025/S30W065/lst_p95_2021-2025_S30W065.tif",
)
@click.option("--tile", "-t", required=True, help="Tile the raster must be, e.g. S30W065")
@click.option(
    "--ged-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="AG100 v003 granule directory. Defaults to settings.ged_dir. The "
    "tiering needs observation counts, which the compact gap artifact does "
    "not carry -- it stores only the NumObs == 0 cells.",
)
@click.option("--threshold-c", type=float, default=70.0, help="Hot-tail lower bound, Celsius")
@click.option("--buffer-cells", type=int, default=1, help="Dilation radius for the mask rules")
@click.option("--block-rows", type=int, default=512, help="Rows per windowed read")
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the machine-readable record here",
)
@click.option("--json", "as_json", is_flag=True, help="Print the record instead of a table")
def ged_analyze(
    *,
    raster: str,
    tile: str,
    ged_dir: Path | None,
    threshold_c: float,
    buffer_cells: int,
    block_rows: int,
    out: Path | None,
    as_json: bool,
) -> None:
    """Cross-tab a published LST tile by ASTER GED observation count.

    Every output pixel is placed in the ~1 km GED cell it falls inside and
    counted as valid, missing, or hot-tail, per NumObs tier. The mapping is
    `ged.cell_indices_for_geobox`, the same one the production mask uses, so
    the analysis and the mask cannot drift apart.

    \b
    The result is a **spatial association**, not a causal trace: it does not
    follow which ASTER observations produced any pixel's emissivity.

    Reads are windowed, so an 18,000-squared tile costs one pass and a few
    hundred MB whether the raster is local or an https URL.
    """
    import json as json_module

    from landsat_lst.config import settings
    from landsat_lst.ged_analysis import AnalysisInputError, analyze

    source = ged_dir if ged_dir is not None else settings.ged_dir
    if not Path(source).is_dir():
        msg = f"GED granule directory {source} does not exist; pass --ged-dir or set LST_GED_DIR"
        raise click.ClickException(msg)

    try:
        record = analyze(
            raster=raster,
            tile=tile,
            ged_dir=Path(source),
            hot_threshold_c=threshold_c,
            buffer_cells=buffer_cells,
            block_rows=block_rows,
        )
    except AnalysisInputError as e:
        raise click.ClickException(str(e)) from e

    text = json_module.dumps(record, indent=2, sort_keys=False)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    if as_json:
        click.echo(text)
        return
    _print_ged_analysis(record)
    if out is not None:
        console.print(f"\n  Written to {out}")


def _print_ged_analysis(record: dict) -> None:
    """Render the cross-tab and the mask tradeoffs as tables."""
    from rich.table import Table

    raster, totals = record["raster"], record["tile_totals"]
    console.print(f"[bold]{raster['source']}[/bold]")
    console.print(
        f"  {raster['height']} x {raster['width']} {raster['dtype']}  {raster['crs']}  "
        f"nodata {raster['nodata']}  scale {raster['scale']}  offset {raster['offset']}"
    )
    console.print(f"  scenes: {raster['scene_count']}  ({raster['scene_count_source']})")
    console.print(
        f"  threshold: >= {record['threshold']['hot_threshold_c']} C "
        f"= DN {record['threshold']['hot_threshold_dn']}"
    )
    console.print(
        f"  tile: {totals['total_pixels']:,} px, {totals['valid_pixels']:,} valid, "
        f"{totals['missing_pixels']:,} missing, {totals['hot_pixels']:,} hot "
        f"({totals['hot_pct_of_valid']:.6f}% of valid)"
    )

    table = Table(title="Pixels by ASTER GED NumObs tier (spatial association)")
    for name in ("NumObs", "total", "valid", "missing", "hot", "hot/valid %", "hot enrich."):
        table.add_column(name, justify="left" if name == "NumObs" else "right")
    for row in record["by_numobs_tier"]:
        enrich = row["hot_enrichment_vs_tile"]
        table.add_row(
            row["tier"],
            f"{row['total_pixels']:,}",
            f"{row['valid_pixels']:,}",
            f"{row['missing_pixels']:,}",
            f"{row['hot_pixels']:,}",
            "-" if row["hot_pct_of_tier_valid"] is None else f"{row['hot_pct_of_tier_valid']:.4f}",
            "-" if enrich is None else f"{enrich:,.1f}x",
        )
    console.print(table)

    trade = Table(title="Candidate mask rules")
    for name in ("rule", "valid removed", "valid %", "hot removed", "hot %", "missing annotated"):
        trade.add_column(name, justify="left" if name == "rule" else "right")
    for row in record["mask_tradeoffs"]:
        trade.add_row(
            row["rule"],
            f"{row['valid_pixels_removed']:,}",
            f"{row['valid_pixels_removed_pct']:.3f}",
            f"{row['hot_pixels_removed']:,}",
            f"{row['hot_pixels_removed_pct']:.2f}",
            f"{row['missing_pixels_annotated']:,}",
        )
    console.print(trade)


@main.group()
def catalog() -> None:
    """Build and validate the published STAC catalog."""


@catalog.command("build")
@click.option("--source", required=True, help="Directory or s3:// prefix of finished COGs")
@click.option("--out", default="./catalog", help="Directory to write the catalog into")
@click.option("--window", default=None, help="Observation window label, e.g. 2021-2025")
@click.option("--tiles", default=None, help="Comma-separated tile names to include")
@click.option(
    "--thumbnail",
    default=None,
    help="PNG to register as the thumbnail, instead of rendering one from the tiles",
)
@click.option(
    "--metadata-only",
    is_flag=True,
    help=(
        "Write a JSON-only staging tree: read every COG header but do not "
        "copy or download the COGs beside their items. For s3:// sources "
        "whose COGs already sit at their published paths."
    ),
)
def catalog_build(
    source: str,
    out: str,
    *,
    window: str | None,
    tiles: str | None,
    thumbnail: str | None,
    metadata_only: bool,
) -> None:
    """Build a Portolan-compliant STAC catalog from finished per-tile COGs."""
    from landsat_lst.catalog import build_catalog
    from landsat_lst.catalog.spec import DEFAULT_SPEC, spec_for_window

    spec = DEFAULT_SPEC if window is None else spec_for_window(window)
    wanted = tuple(name.strip() for name in tiles.split(",")) if tiles else None
    console.print(f"[bold]Building catalog for {spec.window}[/bold] from {source}")
    root = build_catalog(
        source,
        out,
        spec,
        tiles=wanted,
        thumbnail=thumbnail,
        place_assets=not metadata_only,
    )
    console.print(f"  Wrote [green]{root}[/green]")
    if metadata_only:
        console.print(
            "  [yellow]Metadata-only tree: validator byte checks will skip "
            "the COGs. Sample real tiles locally for byte coverage.[/yellow]"
        )


def _print_report(report: Report, unaccepted: set[str]) -> None:
    """Render a validation report for a person reading a terminal."""
    for finding in report.errors:
        console.print(f"[red]{finding.rule_id}[/red] {finding.path}: {finding.message}")
    for summary in report.by_rule():
        colour = "red" if summary.severity.value == "error" else "yellow"
        console.print(
            f"  [{colour}]{summary.rule_id}[/{colour}] x{summary.count}  {summary.description}"
        )
    if unaccepted:
        console.print(f"[red]Warnings outside the accepted baseline: {sorted(unaccepted)}[/red]")


@catalog.command("validate")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit the full report as JSON")
@click.option("--live", is_flag=True, help="Also probe the hosting server (requires network)")
@click.option("--live-base-url", default=None, help="https URL the catalog root is published under")
def catalog_validate(path: Path, as_json: bool, live: bool, live_base_url: str | None) -> None:
    """Validate a built catalog against the Portolan profile.

    The default run is offline. --live adds the hosting pass, which probes the
    published server for range support, CORS, and Content-Length. The catalog's
    hrefs are relative, so that pass needs --live-base-url to know what to probe.
    """
    import json as json_module

    from landsat_lst.catalog.validation import unaccepted_warnings, validate_catalog

    if live and live_base_url is None:
        msg = "--live needs --live-base-url: the catalog's asset hrefs are relative"
        raise click.UsageError(msg)
    # --live-base-url alone turns the pass on too, so naming a URL is never a
    # no-op that quietly reports an offline verdict.
    report = validate_catalog(path, live_base_url=live_base_url)
    unaccepted = unaccepted_warnings(report)
    if as_json:
        payload = report.to_dict()
        payload["unaccepted_warnings"] = sorted(unaccepted)
        click.echo(json_module.dumps(payload, indent=2))
    else:
        _print_report(report, unaccepted)
    if report.errors or unaccepted:
        raise SystemExit(1)


@catalog.command("publish")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--remote", required=True, help="s3://bucket/prefix/ to publish the tree under")
@click.option("--dry-run", is_flag=True, help="Print the upload plan without uploading")
@click.option("--profile", default=None, help="AWS named profile to authenticate with")
def catalog_publish(path: Path, remote: str, dry_run: bool, profile: str | None) -> None:
    """Sync a built catalog tree to the S3 prefix it is published under.

    Each object is uploaded with the media type its extension declares. An
    asset whose remote size already matches is skipped; JSON and markdown are
    always re-sent, because an equal-sized metadata edit is realistic and they
    cost kilobytes.
    """
    from landsat_lst.catalog.publish import publish_catalog

    console.print(f"[bold]Publishing {path}[/bold] to {remote}")
    summary = publish_catalog(path, remote, dry_run=dry_run, profile=profile)

    if dry_run:
        for upload in summary.planned:
            console.print(f"    upload  {upload}")
        console.print(
            f"[yellow]Dry run - nothing uploaded.[/yellow] Would upload "
            f"{len(summary.planned)} objects ({summary.planned_bytes} bytes), "
            f"skipping {summary.skipped_count} unchanged."
        )
        return

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Uploaded: [green]{summary.uploaded_count}[/green]")
    console.print(f"  Skipped:  [yellow]{summary.skipped_count}[/yellow]")
    console.print(f"  Bytes:    {summary.uploaded_bytes}")


@main.group("shard")
def shard() -> None:
    """One tile across many VMs: the local driver, and the shard tasks it starts.

    ``process`` and ``resume`` are the two a person runs. The five stage
    subcommands are what a Coiled Batch task runs on a VM; they take a shard
    index and are not meant to be typed, though running one by hand is a
    legitimate way to reproduce a failed shard.
    """


def _shard_job(tile: str, year: int | None, end_year: int | None, max_scenes: int | None):
    """One job, through the same window defaulting every other command uses."""
    jobs = _build_jobs(year, end_year, (tile,), max_scenes)
    return jobs[0]


@shard.command("process")
@click.option("-t", "--tile", required=True, help="Tile to build, e.g. N40W075")
@click.option("-y", "--year", type=int, default=None, help="Start year; omit for the window")
@click.option("--end-year", type=int, default=None, help="End year (inclusive)")
@click.option("--max-scenes", type=int, default=None, help="Sample at most N scenes")
@click.option("--run-id", default=None, help="Run token; generated when omitted")
@click.option(
    "--ack-quota",
    is_flag=True,
    help="Proceed when the Coiled credit balance cannot be read, on your own check",
)
def shard_process(
    *,
    tile: str,
    year: int | None,
    end_year: int | None,
    max_scenes: int | None,
    run_id: str | None,
    ack_quota: bool,
) -> None:
    """Build one tile as a fleet of shards, driven from this shell.

    This shell has to stay open, unlike ``process --distributed``: it is the
    thing sequencing the stages, because Coiled Batch has no dependency
    mechanism. It holds no state, though -- print the run id and
    ``landsat-lst shard resume <run-id> <tile>`` picks up wherever the bucket
    says the run got to.
    """
    from landsat_lst import quota
    from landsat_lst.config import settings
    from landsat_lst.shard_driver import (
        ShardBackendMismatch,
        ShardStageFailed,
        drive_tile,
        require_shared_storage,
        shard_run_id,
    )
    from landsat_lst.storage import get_storage

    job = _shard_job(tile, year, end_year, max_scenes)
    # Before the run id is printed: a driver that cannot see its shards' output,
    # or that cannot pay for them, has not started a run -- and printing a
    # resume hint for it would be a lie.
    if ack_quota:
        settings.ack_quota = True
    try:
        require_shared_storage(get_storage(), None)
        # Identity before credits: a session that cannot call STS cannot read a
        # Coiled balance either, and "log in again" beats "balance unreadable".
        quota.preflight_identity()
        estimate = quota.estimate_run_credits()
        balance = quota.preflight_credits(estimate)
    except (ShardBackendMismatch, quota.IdentityRefused, quota.QuotaRefused) as e:
        raise click.ClickException(str(e)) from e
    console.print(
        f"  credits: ~{estimate:.0f} needed, "
        f"{'unknown' if balance.remaining is None else f'{balance.remaining:.0f}'} "
        f"remaining ({balance.source})"
    )
    run_id = run_id or shard_run_id(job)
    console.print(f"[bold]Sharding {tile}[/bold] {job.window_label}  run-id [cyan]{run_id}[/cyan]")
    console.print(f"  resume with: landsat-lst shard resume {run_id} {tile}")

    try:
        summary = drive_tile(job, run_id=run_id)
    except (ShardStageFailed, ShardBackendMismatch) as e:
        raise click.ClickException(str(e)) from e

    _print_shard_summary(summary)


@shard.command("resume")
@click.argument("run_id")
@click.argument("tile")
@click.option(
    "--ack-quota",
    is_flag=True,
    help="Proceed when the Coiled credit balance cannot be read, on your own check",
)
def shard_resume(run_id: str, tile: str, ack_quota: bool) -> None:
    """Continue a killed driver's run, reading its position out of the bucket."""
    from landsat_lst import quota
    from landsat_lst.config import settings
    from landsat_lst.shard_driver import ShardBackendMismatch, ShardStageFailed, resume_tile

    if ack_quota:
        settings.ack_quota = True
    console.print(f"[bold]Resuming {tile}[/bold] in run [cyan]{run_id}[/cyan]")
    try:
        summary = resume_tile(run_id, tile)
    except (
        ShardStageFailed,
        ShardBackendMismatch,
        quota.IdentityRefused,
        quota.QuotaRefused,
    ) as e:
        raise click.ClickException(str(e)) from e

    _print_shard_summary(summary)


def _print_shard_summary(summary) -> None:
    from rich.table import Table

    table = Table(title=f"{summary.tile} {summary.window}")
    for column in ("stage", "shards", "skipped", "submits", "wall (s)"):
        table.add_column(column, justify="right" if column != "stage" else "left")
    for stage in summary.stages:
        table.add_row(
            stage.stage,
            str(stage.shards),
            str(stage.already_done),
            str(stage.submissions),
            f"{stage.wall_s:.0f}",
        )
    console.print(table)
    verdict = "[green]complete[/green]" if summary.completed else "[red]incomplete[/red]"
    console.print(
        f"  {verdict}  {summary.wall_s / 60:.0f} min, {summary.resubmissions} resubmission(s)"
    )


def _shard_stage_command(stage: str, needs_job: bool = False):
    """Register one stage subcommand: the thing a batch task actually runs."""

    def decorator(func):
        func = click.option("--index", type=int, default=0, help="Which shard of this stage")(func)
        if needs_job:
            func = click.option(
                "--units",
                type=int,
                default=None,
                help="Fused offsets fleet width the plan is cut to",
            )(func)
            func = click.option("--max-scenes", type=int, default=None)(func)
            func = click.option("--end-year", type=int, default=None)(func)
            func = click.option("-y", "--year", type=int, default=None)(func)
        func = click.option("-t", "--tile", required=True, help="Tile this shard belongs to")(func)
        func = click.option("--run-id", required=True, help="Run token")(func)
        return shard.command(stage)(func)

    return decorator


@_shard_stage_command("resolve", needs_job=True)
def shard_resolve(
    *,
    run_id: str,
    tile: str,
    year: int | None,
    end_year: int | None,
    max_scenes: int | None,
    units: int | None,
    index: int,
) -> None:
    """Query the catalog once and freeze where this tile is cut."""
    from landsat_lst.shard_tasks import run_shard

    job = _shard_job(tile, year, end_year, max_scenes)
    plan = run_shard("resolve", run_id, tile, index, job=job, units=units)
    console.print(
        f"planned {tile}: {len(plan.scene_ids)} scenes, {len(plan.blocks)} blocks, "
        f"{plan.ref_shards}/{plan.scene_shards}/{len(plan.bands)} shards, digest {plan.digest}"
    )


@_shard_stage_command("climatology")
def shard_climatology(*, run_id: str, tile: str, index: int) -> None:
    """Reduce this shard's blocks of the 12-month climatology."""
    from landsat_lst.shard_tasks import run_shard

    written = run_shard("climatology", run_id, tile, index)
    console.print(f"{tile} climatology shard {index}: {len(written)} block(s)")


@_shard_stage_command("offsets", needs_job=True)
def shard_offsets(
    *,
    run_id: str,
    tile: str,
    year: int | None,
    end_year: int | None,
    max_scenes: int | None,
    units: int | None,
    index: int,
) -> None:
    """The whole offsets side of a tile: resolve, climatology, barrier, offsets.

    One fleet, one boot. Shard 0 resolves; every shard then waits for that plan,
    reduces its climatology blocks, waits at the in-process phase-A barrier, and
    estimates its scenes' offsets. The window arguments are needed only by shard
    0, and reach every shard because one command shape is cheaper than two.
    """
    from landsat_lst.shard_tasks import run_shard

    job = _shard_job(tile, year, end_year, max_scenes)
    key = run_shard("offsets", run_id, tile, index, job=job, units=units)
    console.print(f"{tile} offsets shard {index}: {key or 'nothing to publish'}")


@_shard_stage_command("composite")
def shard_composite(*, run_id: str, tile: str, index: int) -> None:
    """Composite one row band and publish both products' slabs."""
    from landsat_lst.shard_tasks import run_shard

    written = run_shard("composite", run_id, tile, index)
    console.print(f"{tile} composite band {index}: {len(written)} slab(s)")


@_shard_stage_command("export")
def shard_export(*, run_id: str, tile: str, index: int) -> None:
    """Stitch the row bands into the tile's two COGs."""
    from landsat_lst.shard_tasks import run_shard

    written = run_shard("export", run_id, tile, index)
    console.print(f"{tile} export: {len(written)} COG(s)")


if __name__ == "__main__":
    main()
