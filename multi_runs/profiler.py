import csv
import logging
import os
import time
from functools import wraps
from pathlib import Path
from threading import Thread
from typing import Callable

import psutil

MIB = 1024**2


def profile_usage(interval: float = 0.5):
    """
    Decorator to get metrics
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            proc = psutil.Process()

            cpu_percent = []
            memory_usage = []
            running = True

            def sampler():
                proc.cpu_percent()
                while running:
                    try:
                        cpu_p = proc.cpu_percent()
                        mem = proc.memory_full_info().uss / MIB
                        cpu_percent.append(cpu_p)
                        memory_usage.append(mem)
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        print(f"Program was terminated unexpectedly (exception: {str(e)})")
                        return
                    time.sleep(interval)

            sampler_thread = Thread(target=sampler)
            sampler_thread.start()

            time_start = time.time()
            # CPU times: init
            cpu_times_start = proc.cpu_times()

            # Execute function
            func(*args, **kwargs)

            cpu_times_end = proc.cpu_times()
            time_end = time.time()

            running = False
            sampler_thread.join()

            # Calculate
            cpu_times_system = cpu_times_end.system - cpu_times_start.system
            cpu_times_user = cpu_times_end.user - cpu_times_start.user

            max_cpu_perc = 0 if not cpu_percent else max(cpu_percent)
            avg_cpu_perc = 0 if not cpu_percent else sum(cpu_percent) / len(cpu_percent)

            max_mem_usage = 0 if not memory_usage else max(memory_usage)
            avg_mem_usage = 0 if not memory_usage else sum(memory_usage) / len(memory_usage)

            runtime = time_end - time_start

            # Log
            logging.info(f"\n--- {func.__name__} ---")
            logging.info(f"Wall‑clock: {runtime:.3f}s")
            logging.info(f"CPU times user: {cpu_times_user:.3f}s")
            logging.info(f"CPU times sys: {cpu_times_system:.3f}s")
            logging.info(f"Max CPU usage: {max_cpu_perc:.3f}%")
            logging.info(f"Mean CPU usage: {avg_cpu_perc:.3f}%")
            logging.info(f"Max USS: {max_mem_usage:.3f} MB")
            logging.info(f"Mean USS: {avg_mem_usage:.3f} MB")

            results_folder = Path(os.path.dirname(__file__)).parent / "results" / "metrics"
            results_folder.mkdir(exist_ok=True)
            out_file = results_folder / "metrics.csv"

            header = [
                "TIMESTAMP",
                "SOLVER",
                "DATASET",
                "RUN",
                "RUNTIME",
                "CPU_TIME_USER",
                "CPU_TIME_SYS",
                "MAX_CPU_USAGE",
                "AVG_CPU_USAGE",
                "MAX_RAM_USAGE",
                "AVG_RAM_USAGE",
            ]

            f_exists = out_file.exists()
            with open(out_file, "a") as f:
                writer = csv.writer(f)
                if not f_exists:
                    writer.writerow(header)
                writer.writerow(
                    [
                        time.time(),
                        str(args[0]),
                        str(args[1]),
                        str(args[2]),
                        f"{runtime:.3f}",
                        f"{cpu_times_user:.3f}",
                        f"{cpu_times_system:.3f}",
                        f"{max_cpu_perc:.3f}",
                        f"{avg_cpu_perc:.3f}",
                        f"{max_mem_usage:.3f}",
                        f"{avg_mem_usage:.3f}",
                    ]
                )

        return wrapper

    return decorator
