"""Core multithreaded matrix multiplication engine.

Design summary
--------------
* A single background "driver" thread owns the ThreadPoolExecutor and walks
  the matrix row by row (i = 0 .. n-1).
* For each row i, the driver submits n*n tasks -- one per (j, k) pair -- to
  the ThreadPoolExecutor. Each task is exactly ONE scalar multiplication
  A[i][k] * B[k][j], satisfying "every multiplication operation must be on
  a thread" without ever creating n^3 physical Thread objects.
* The driver thread collects that row's futures with
  concurrent.futures.as_completed(...) and accumulates the partial products
  into C[i][j] itself. Because only the single driver thread ever writes to
  C, there is no race condition on the result matrix and no lock is needed
  for it.
* Every finished operation is also pushed onto a bounded queue.Queue that
  the Tkinter GUI drains on the main thread via root.after(). The queue is
  bounded (maxsize) so that if the GUI can't keep up, the driver thread's
  queue.put() blocks -- this naturally paces the animation to something a
  human can actually follow instead of flooding the GUI with a million
  updates per second. The *real* computation (and the completed-operation
  counter) is never gated by the GUI's drawing speed.
* Pause/Stop use threading.Event objects checked at row boundaries (Stop is
  additionally checked inside the per-row completion loop) -- fine-grained
  enough for a live demo, simple enough to explain in a viva.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

from models import (
    OperationResult,
    STATUS_IDLE,
    STATUS_RUNNING,
    STATUS_PAUSED,
    STATUS_STOPPED,
    STATUS_DONE,
)
from multiplication_task import multiply_scalar


class MultiplicationEngine:
    def __init__(self, a: np.ndarray, b: np.ndarray, thread_count: int = 8, queue_maxsize: int = 500):
        self.a = a
        self.b = b
        self.n = a.shape[0]
        self.thread_count = max(1, thread_count)

        self.c = np.zeros((self.n, self.n), dtype=np.int64)

        self.total_ops = self.n ** 3
        self.completed_ops = 0
        self.thread_counts: Dict[str, int] = defaultdict(int)
        self.thread_last_seen: Dict[str, float] = {}

        self.result_queue: "queue.Queue[OperationResult]" = queue.Queue(maxsize=queue_maxsize)

        self.status = STATUS_IDLE
        self.pause_event = threading.Event()
        self.pause_event.set()  # not paused
        self.stop_flag = threading.Event()

        self._driver_thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        self.verification_passed: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # Public controls (safe to call from the GUI/main thread)
    # ------------------------------------------------------------------ #
    def start(self):
        if self._driver_thread is not None:
            return
        self.status = STATUS_RUNNING
        self.start_time = time.perf_counter()
        self._executor = ThreadPoolExecutor(max_workers=self.thread_count, thread_name_prefix="Worker")
        self._driver_thread = threading.Thread(target=self._run, name="Driver", daemon=True)
        self._driver_thread.start()

    def pause(self):
        if self.status == STATUS_RUNNING:
            self.pause_event.clear()
            self.status = STATUS_PAUSED

    def resume(self):
        if self.status == STATUS_PAUSED:
            self.pause_event.set()
            self.status = STATUS_RUNNING

    def stop(self):
        self.stop_flag.set()
        self.pause_event.set()  # unblock a paused driver so it can see the stop flag

    # ------------------------------------------------------------------ #
    # Driver thread body -- NEVER touches Tkinter directly
    # ------------------------------------------------------------------ #
    def _run(self):
        n = self.n
        try:
            for i in range(n):
                if self.stop_flag.is_set():
                    break
                self.pause_event.wait()
                if self.stop_flag.is_set():
                    break

                futures = [
                    self._executor.submit(multiply_scalar, self.a, self.b, i, j, k)
                    for j in range(n)
                    for k in range(n)
                ]

                row_partial: Dict[int, int] = defaultdict(int)
                for fut in as_completed(futures):
                    if self.stop_flag.is_set():
                        break
                    i_, j_, k_, a_val, b_val, product, tname = fut.result()
                    row_partial[j_] += product
                    self.completed_ops += 1
                    self.thread_counts[tname] += 1
                    self.thread_last_seen[tname] = time.perf_counter()

                    op = OperationResult(i_, j_, k_, a_val, b_val, product, tname)
                    try:
                        self.result_queue.put(op, timeout=0.5)
                    except queue.Full:
                        # GUI is far behind -- drop this one for display purposes only.
                        # The real computation above already happened and was counted.
                        pass

                for j in range(n):
                    self.c[i, j] = row_partial[j]

            self.end_time = time.perf_counter()
            if self.stop_flag.is_set():
                self.status = STATUS_STOPPED
            else:
                self.status = STATUS_DONE
                self._verify()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)

    def _verify(self):
        """Cross-check the threaded result against NumPy's matmul.
        NumPy is used HERE ONLY for verification, never for the actual
        computation above."""
        expected = self.a.astype(np.int64) @ self.b.astype(np.int64)
        self.verification_passed = bool(np.array_equal(expected, self.c))

    # ------------------------------------------------------------------ #
    # Helpers for the GUI (read-only, called from the main thread)
    # ------------------------------------------------------------------ #
    @property
    def progress_fraction(self) -> float:
        if self.total_ops == 0:
            return 0.0
        return self.completed_ops / self.total_ops

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time

    def drain_queue(self, max_items: int) -> List[OperationResult]:
        """Pull up to max_items operations off the result queue.
        Must only be called from the Tkinter main thread (e.g. inside an
        after() callback) -- queue.Queue itself is thread-safe, but this
        keeps the calling convention explicit."""
        items: List[OperationResult] = []
        for _ in range(max_items):
            try:
                items.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return items
