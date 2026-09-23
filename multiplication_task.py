"""The single unit of work required by the assignment: ONE multiplication.

Each call to `multiply_scalar` represents exactly one scalar multiplication
A[i][k] * B[k][j]. This is the function that gets submitted to the
ThreadPoolExecutor -- one task per multiplication, as the assignment
requires ("Every multiplication operation must be on a thread"). We do NOT
create one Python Thread object per multiplication (that would mean one
million Thread objects for a 100x100 case) -- instead a small, fixed pool
of worker threads pulls these tasks off an internal work queue and executes
them, one task at a time per worker.
"""

import threading


def multiply_scalar(a, b, i, j, k):
    """Perform ONE scalar multiplication A[i][k] * B[k][j].

    Runs on whichever worker thread the ThreadPoolExecutor assigns it to.
    Returns a plain tuple (cheap to construct, no shared state touched)
    so the caller can build display/accumulation data without extra
    synchronization:

        (i, j, k, a_val, b_val, product, thread_name)
    """
    a_val = int(a[i, k])
    b_val = int(b[k, j])
    product = a_val * b_val
    thread_name = threading.current_thread().name
    return i, j, k, a_val, b_val, product, thread_name
