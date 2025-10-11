import time
import tracemalloc
from functools import wraps

import psutil

proc = psutil.Process()  # current process


def loop_timer(func):
    """Decorator that prints wall‑clock time, CPU % and memory usage."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        time_start = time.perf_counter()

        cpu_start = proc.cpu_times()  # (user, system, ...)
        rss_start = proc.memory_info().rss  # bytes

        tracemalloc.start()

        # Execute the loop
        result = func(*args, **kwargs)

        # End, get results
        _, peak = tracemalloc.get_traced_memory()
        time_end = time.perf_counter()

        cpu_end = proc.cpu_times()
        cpu_user = cpu_end.user - cpu_start.user
        cpu_sys = cpu_end.system - cpu_start.system
        rss_end = proc.memory_info().rss

        tracemalloc.stop()

        # ---- Print ---------------------------------------------------------
        print(f"\n--- {func.__name__}, solver: {str(args[0])} ---")
        print(f"Wall‑clock : {time_end - time_start:.3f}s")
        print(f"CPU user   : {cpu_user:.3f}s")
        print(f"CPU sys    : {cpu_sys:.3f}s")
        print(f"RSS change : {(rss_end - rss_start) / 1024**2:+.3f} MB")
        print(f"Peak alloc : {peak / 1024**2:.3f} MB")
        print(f"Peak RSS   : {rss_end / 1024**2:.3f} MB")

        return result

    return wrapper
