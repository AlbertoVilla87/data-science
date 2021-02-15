"""
This module includes functions used to measure the performance of the code.
"""
import functools
import time


def timer(logger=None):
    """
    Decorator to measure the time a function takes to execute.

    If a logger is passed, the information will be added to the logger.
    Otherwise, information will printed on the console.

    Example:
        Using a logger:

        .. code-block:: python

            @timer(logger)
            def lazy_add_numbers(x, y):
                return x + y

        Without a logger:

        .. code-block:: python

            @timer()
            def lazy_add_numbers(x, y):
                return x + y

    Args:
        logger (DsaaLogger, optional): Logger. Defaults to None.

    Returns:
        object: What the function being decorated returns
    """   
    def decorator_timer(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            # Start time
            start_time = time.perf_counter()

            # Execute function
            value = func(*args, **kwargs)

            # Calculate delta
            end_time = time.perf_counter()
            run_time = end_time - start_time

            msg = f"Finished {func.__name__!r} in {run_time:.4f} secs"

            if logger:
                logger.info(msg)
            else:
                print(msg)

            return value

        return wrapper_timer

    return decorator_timer