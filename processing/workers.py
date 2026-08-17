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
    totals = {}
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = [pool.submit(worker_fn) for _ in range(max(1, num_workers))]
        for future in as_completed(futures):
            try:
                result = future.result() or {}
            except Exception:
                logger.exception(f"[{label}] worker thread crashed")
                continue
            for key, value in result.items():
                totals[key] = totals.get(key, 0) + value
    return totals
