"""
Progress + ETA reporting for the long-running stages.

Each stage knows how many rows it is about to work on (the same WHERE clause
its claim query uses, run once as a COUNT before the workers start), so it can
report "N/M done, X% , ETA hh:mm:ss" instead of only per-row lines that give no
sense of how much is left.

The rate is the run's overall average (items / elapsed since the stage
started), not a short rolling window: per-item cost here swings wildly (a
text-based PDF is ~1s, a 40-page scan through OCR is minutes), so a window
short enough to react produces an ETA that jumps around. The average settles
after the first few dozen rows and stays readable.

Thread-safe: every worker thread in a stage advances the same tracker.
"""
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Minimum seconds between progress lines, so a fast stage (promote) doesn't
# drown the log in ETA updates.
LOG_INTERVAL_SECONDS = float(os.getenv("PROGRESS_LOG_SECONDS", "15"))


def format_duration(seconds):
    """Human duration: '45s', '12m 03s', '2h 07m'."""
    if seconds is None:
        return "unknown"
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class Progress:
    """
    Counts completed items against a known total and logs an ETA.

    `total` may be None (unknown) — then it still reports the count, the
    elapsed time and the throughput, just without a percentage or ETA.
    """

    def __init__(self, label, total=None, workers=1):
        self.label = label
        self.total = total
        self.workers = workers
        self.done = 0
        self.started_at = time.monotonic()
        self._last_log_at = self.started_at
        self._lock = threading.Lock()
        if total:
            logger.info(f"[{label}] {total} item(s) to process with {workers} worker(s)")
        else:
            logger.info(f"[{label}] starting with {workers} worker(s) (total unknown)")

    def advance(self, count=1):
        """Record `count` finished items; log a progress line if due."""
        with self._lock:
            self.done += count
            now = time.monotonic()
            is_last = self.total is not None and self.done >= self.total
            if not is_last and (now - self._last_log_at) < LOG_INTERVAL_SECONDS:
                return
            self._last_log_at = now
            message = self._render(now)
        logger.info(message)

    def finish(self):
        """Log a final line with the actual elapsed time."""
        elapsed = time.monotonic() - self.started_at
        logger.info(
            f"[{self.label}] finished {self.done} item(s) in {format_duration(elapsed)}"
            f"{self._rate_suffix(elapsed)}"
        )

    def _rate_suffix(self, elapsed):
        if self.done <= 0 or elapsed <= 0:
            return ""
        per_minute = self.done / elapsed * 60
        return f" ({per_minute:.1f} items/min)"

    def _render(self, now):
        """Caller holds the lock."""
        elapsed = now - self.started_at
        parts = [f"[{self.label}] progress {self.done}"]
        if self.total:
            pct = self.done / self.total * 100
            parts[0] += f"/{self.total} ({pct:.1f}%)"
        parts.append(f"elapsed {format_duration(elapsed)}")
        if self.total and self.done > 0:
            remaining = max(0, self.total - self.done)
            eta = remaining * (elapsed / self.done)
            parts.append(f"ETA {format_duration(eta)}")
        rate = self._rate_suffix(elapsed).strip(" ()")
        if rate:
            parts.append(rate)
        return " | ".join(parts)


class NullProgress:
    """No-op stand-in so callers never need to branch on `progress is None`."""

    def advance(self, count=1):
        pass

    def finish(self):
        pass
