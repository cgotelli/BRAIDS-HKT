"""Bounded, ordered archive I/O; numerical execution never uses this pool.

Every file is still copied and authenticated. Threads overlap independent
filesystem waits, not solver work. A live runtime lease must remain on its
owning thread: its dependency observer and reentrant lock cannot cross threads.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os

from .river_runtime_lease import runtime_lease_active

# The matched 788-file archive sweep qualifies 16 over 1, 4 and 8 workers.
# This is filesystem concurrency, not a hydraulic CPU/GPU or thread selector.
IO_WORKERS = 16


def map_files(function, items, *, cancelled=lambda: None, workers=None):
    """Return ordered results; join all running tasks before any exit.

    Cancellation callbacks execute only on the calling thread. At most two
    tasks per worker are queued, and unpublished copies remain private after
    cancellation or an exception. No background writer outlives this call.
    """
    requested = min(IO_WORKERS, os.cpu_count() or 1) if workers is None else workers
    if type(requested) is not int or requested < 1:
        raise ValueError('Source I/O workers must be a positive integer')
    cancelled()
    if requested == 1 or runtime_lease_active():
        results = []
        for item in items:
            cancelled()
            results.append(function(item))
        cancelled()
        return results
    iterator = iter(items)
    pending = deque()
    results = []
    pool = ThreadPoolExecutor(max_workers=requested, thread_name_prefix='braids-source')
    try:
        for _ in range(2*requested):
            cancelled()
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(pool.submit(function, item))
        while pending:
            cancelled()
            results.append(pending.popleft().result())
            cancelled()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(pool.submit(function, item))
        cancelled()
        return results
    finally:
        # In particular, do not leave copies racing publication or a retry.
        pool.shutdown(wait=True, cancel_futures=True)
