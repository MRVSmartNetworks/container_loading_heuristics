import time
import signal

"""
This script contains the settings for setting the time limit in the execution of the solver.

- TIMEOUT_VALUE: timeout duration in seconds
- TimeoutException: exception raised when timeout is triggered (by the `signal.SIGALRM` signal)
- timeout_handler: function used for raising TimeoutException
"""

TIMEOUT_VALUE = 1 * 60  # Timeout duration in seconds


class TimeoutException(Exception):
    """Exception type used for interrupting the execution of the program"""

    pass


def timeout_handler(signum, frame):
    raise TimeoutException


if __name__ == "__main__":
    # Test program for raising Timeout Exception - infinite loop interrupted by timer

    time_start = time.time()

    # Initialize signal to be sent
    signal.signal(signal.SIGALRM, timeout_handler)

    # Start timer
    signal.alarm(TIMEOUT_VALUE)

    sym = ["|", "/", "-", "\\"]
    i = 0
    try:
        while True:
            """
            Place here long operation (e.g., solver)
            """

            print(f"Waiting {sym[i]}", end="\r")
            i = (i + 1) % len(sym)
            time.sleep(0.1)
    except TimeoutException:
        print("Timer finished!")

    # Stop the alarm (needed if there are still operations that need to be executed,
    # such as solution checks, or the 'long operation' in the loop finishes before
    # the timer is triggered)
    signal.alarm(0)
