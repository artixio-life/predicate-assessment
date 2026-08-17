"""
Runs N copies of a stage's worker loop concurrently and sums their results.

Each worker function manages its own DB connection (psycopg2 connections
aren't thread-safe to share) and loops claiming + processing rows until
there's nothing left to claim, returning a dict of outcome -> count for
what it did. This module just fans that out and merges the totals.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def run_worker_pool(worker_fn, num_workers, label):
    """
    `worker_fn` is called as worker_fn(worker_index, num_workers) so a stage
    can shard its own work deterministically (see promote.py) instead of
    relying on lock timing to keep workers off each other's rows.
    """
    totals = {}
    count = max(1, num_workers)
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(worker_fn, i, count) for i in range(count)]
        for future in as_completed(futures):
            try:
                result = future.result() or {}
            except Exception:
                logger.exception(f"[{label}] worker thread crashed")
                continue
            for key, value in result.items():
                totals[key] = totals.get(key, 0) + value
    return totals
