"""Data models shared across the Multithreaded Matrix Multiplication Visualizer."""

from dataclasses import dataclass


@dataclass
class OperationResult:
    """Result of ONE scalar multiplication A[i][k] * B[k][j] executed on a worker thread."""
    i: int
    j: int
    k: int
    a_val: int
    b_val: int
    product: int
    thread_name: str


# Engine status values
STATUS_IDLE = "IDLE"
STATUS_RUNNING = "RUNNING"
STATUS_PAUSED = "PAUSED"
STATUS_STOPPED = "STOPPED"
STATUS_DONE = "DONE"
