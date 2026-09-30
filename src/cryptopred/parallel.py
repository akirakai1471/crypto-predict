"""Spread independent training jobs across CPU cores.

Walk-forward folds do not depend on each other, and neither do symbols in a
validation run, so both can train at the same time. Doing it with processes
rather than LightGBM's own threads is measurably faster here: on a 4-core
machine one fit took 12.3s on one thread and 5.4s on four, so four one-thread
processes finish four fits in the time four-thread fits finish two and a bit.

Results do not change. LightGBM produced bit-identical predictions at one, two
and four threads on this project's features (max difference 0.0), so a parallel
run reports exactly the numbers a serial one does.

Processes are started with "spawn" on every platform. Forking a parent that has
already run LightGBM copies its OpenMP thread pool in a broken state and can hang
the child - and Windows, where this project runs, only has spawn anyway.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def plan(jobs: int, tasks: int) -> tuple[int, int]:
    """How many worker processes, and how many LightGBM threads each.

    `jobs` <= 0 means "use every core". Never more workers than tasks, and the
    cores are divided between workers so the machine is not oversubscribed:
    eight processes each asking LightGBM for every core run slower than one.
    """
    cpus = os.cpu_count() or 1
    workers = cpus if jobs <= 0 else jobs
    workers = max(1, min(workers, tasks, cpus))
    threads = max(1, cpus // workers)
    return workers, threads


def run[T](
    fn: Callable[..., T],
    arguments: Sequence[tuple[Any, ...]],
    workers: int,
    on_done: Callable[[int, T], None] | None = None,
) -> list[T]:
    """Call fn(*args) for each tuple, returning results in input order.

    With one worker nothing is spawned, so the serial path stays the one the
    tests and the debugger see. `on_done(i, result)` fires as each job finishes,
    in completion order, for progress output.
    """
    if workers <= 1 or len(arguments) <= 1:
        results = []
        for i, args in enumerate(arguments):
            result = fn(*args)
            if on_done is not None:
                on_done(i, result)
            results.append(result)
        return results

    context = multiprocessing.get_context("spawn")
    results: list[T | None] = [None] * len(arguments)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(fn, *args): i for i, args in enumerate(arguments)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            if on_done is not None:
                on_done(i, results[i])
    return results  # type: ignore[return-value]
