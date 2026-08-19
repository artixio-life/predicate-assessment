import time
import logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 2.0


class RetriesExhausted(Exception):
    """
    Raised when a retried call fails on every attempt. Carries how many
    attempts were made and the last underlying exception, so callers can
    persist them straight into a row's `*_attempts` / `*_error` columns.
    """

    def __init__(self, attempts, last_exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Failed after {attempts} attempt(s): {last_exception}")


def run_with_retries(fn, *args, max_attempts=DEFAULT_MAX_ATTEMPTS,
                      backoff_seconds=DEFAULT_BACKOFF_SECONDS, label=None,
                      no_retry_on=(), **kwargs):
    """
    Call fn(*args, **kwargs), retrying on any exception up to `max_attempts`
    total attempts (initial call + retries), with linear backoff
    (backoff_seconds * attempt number) between tries.

    Every pipeline step (promote / text extraction / AI extraction) uses this
    with the default 3 attempts, matching issue #1's "3 retries per step"
    requirement. Returns fn's result on the first success; raises
    RetriesExhausted if every attempt fails.

    `no_retry_on` is a tuple of exception types that propagate immediately
    instead of being retried, for failures a retry provably cannot fix. The
    case this exists for is a context-length overflow in the AI-extraction
    fold: the same oversized payload fails identically every time, so retrying
    it three times with backoff just delays the caller's real remedy (sending a
    smaller batch) by the full backoff.
    """
    name = label or getattr(fn, '__name__', 'call')
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except no_retry_on:
            raise
        except Exception as e:
            last_exc = e
            if attempt < max_attempts:
                delay = backoff_seconds * attempt
                logger.warning(
                    f"[retry] {name} attempt {attempt}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"[retry] {name} attempt {attempt}/{max_attempts} failed: {e}. Giving up.")
    raise RetriesExhausted(max_attempts, last_exc)
