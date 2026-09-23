# Multithreaded Matrix Multiplication Visualizer

A Tkinter desktop app that multiplies two matrices (default 100×100) where
**every individual scalar multiplication `A[i][k] * B[k][j]` runs as a task
on a `ThreadPoolExecutor` worker thread**, animates the process live, and
verifies the result against NumPy.

---

## A. Problem statement
Multiply two n×n matrices (n ≥ 100) so that every scalar multiplication is
executed on a thread, and animate the process so the computation is visible
and explainable.

## B. Proposed solution
A background **driver thread** walks matrix `A` row by row. For each row it
submits `n²` tasks (one per `(j, k)` pair) to a `ThreadPoolExecutor`; each
task performs exactly one multiplication `A[i][k] * B[k][j]`. The driver
collects that row's results, sums the partial products into `C[i][j]`
itself (no locks needed — it's the only writer of `C`), and also pushes
each result onto a bounded `queue.Queue` for the GUI. The Tkinter main
thread drains that queue on a timer (`after()`) and animates the highlighted
row/column/cell, the equation, the thread name, and progress.

## C. Architecture (text diagram)
```
                 ┌───────────────────────────┐
                 │   Driver thread            │
                 │  (walks i = 0..n-1)         │
                 └──────────────┬──────────────┘
                                │ submits n² tasks per row
                                ▼
                 ┌───────────────────────────┐
                 │  ThreadPoolExecutor         │
                 │  Worker Worker Worker ...   │  <- fixed pool (e.g. 8)
                 │   each task = ONE A[i][k]*B[k][j]
                 └──────────────┬──────────────┘
                                │ future.result() per task
                                ▼
                 ┌───────────────────────────┐
                 │ Driver accumulates C[i][j]  │  (single writer, no lock)
                 │ AND pushes OperationResult   │
                 │ onto a bounded queue.Queue   │
                 └──────────────┬──────────────┘
                                │ (backpressure paces the run)
                                ▼
                 ┌───────────────────────────┐
                 │ Tkinter main thread         │
                 │ root.after() polls queue    │
                 │ → updates canvases, labels, │
                 │   progress bar, thread bars │
                 └───────────────────────────┘
```

## D. How this satisfies the assignment
- **"Every multiplication operation must be on a thread"** → `multiply_scalar()`
  in `multiplication_task.py` is the task body; one call = one multiplication;
  every one of the n³ calls is submitted to `ThreadPoolExecutor.submit()`.
- **Doesn't create n³ physical threads** → a fixed-size pool (default 8)
  reuses worker threads across all n³ tasks.
- **Animation** → live-updating heatmaps for A/B/C, highlighted row/column/
  cell, current equation with values, thread name, thread activity bars,
  recent-operations log, progress bar/percentage/elapsed time.
- **Doesn't display all 10,000 cells as numbers** → heatmap coloring for any
  size; numeric text only auto-enabled when n ≤ 12 (use 5×5/10×10 test modes
  to see numbers).
- **GUI never touches Tkinter from worker threads** → workers only return
  data; the driver thread only touches the queue; only the main thread
  touches widgets, via `after()`.
- **Verification** → after a full run, `MultiplicationEngine._verify()`
  computes `A @ B` with NumPy purely to check correctness — NumPy is never
  used to *produce* the actual result.

## E. Threading strategy
`ThreadPoolExecutor(max_workers=thread_count)` with one task per
`(i, j, k)`. Futures for a whole row (`n²` of them, e.g. 10,000 for n=100)
are collected with `as_completed()` before moving to the next row — this
bounds memory (never more than one row's worth of futures alive) while
still literally putting every multiplication on a thread.

## F. Animation strategy
Correctness and animation are decoupled: the driver thread computes and
counts progress immediately and independently of the GUI. Each finished
operation is *also* offered to a bounded `queue.Queue`; if the GUI can't
keep up, `put()` blocks the driver, which naturally throttles the whole run
to a human-watchable pace. `Animation Speed` (Slow/Medium/Fast) controls the
GUI's polling interval and batch size, i.e. how fast the queue drains.

## G. Project structure
```
multithreaded_matrix_visualizer/
├── main.py                   entry point
├── matrix_generator.py       reproducible random matrix generation
├── multiplication_task.py    the ONE-multiplication task (multiply_scalar)
├── multiplication_engine.py  ThreadPoolExecutor driver + queue + verification
├── visualizer.py             Tkinter GUI, canvases, animation loop
├── models.py                 OperationResult dataclass, status constants
├── requirements.txt
└── README.md
```

---

## Setup (Windows, beginner-friendly)

**1. Install Python 3.11+**
Download from https://www.python.org/downloads/ and run the installer.
✅ Tick **"Add python.exe to PATH"** during install.

**2. Check the install**
Open **Command Prompt** (`Win + R` → type `cmd` → Enter):
```
python --version
```
You should see `Python 3.11.x` or newer.

**3. Create a project folder and put these files in it**
```
cd Desktop
mkdir matrix_visualizer
cd matrix_visualizer
```
Copy `main.py`, `matrix_generator.py`, `multiplication_task.py`,
`multiplication_engine.py`, `visualizer.py`, `models.py`,
`requirements.txt` into this folder.

**4. Create a virtual environment**
```
python -m venv venv
```

**5. Activate it (Windows)**
```
venv\Scripts\activate
```
Your prompt should now start with `(venv)`.

**6. Install dependencies**
```
pip install -r requirements.txt
```
(Just NumPy — that's the only third-party package used.)

**7. About Tkinter**
Tkinter ships **built into** the official Windows Python installer — you do
**not** need `pip install tkinter` (that package doesn't exist on PyPI). If
`import tkinter` ever fails on Windows, re-run the Python installer and make
sure "tcl/tk and IDLE" is checked under optional features.

**8. Run it**
```
python main.py
```
A window titled "Multithreaded Matrix Multiplication Visualizer" opens.
Click **Start**.

**9. Common errors**
| Error | Fix |
|---|---|
| `'python' is not recognized...` | Python wasn't added to PATH — reinstall and tick that box, or use `py` instead of `python`. |
| `ModuleNotFoundError: No module named 'numpy'` | You forgot step 6, or the venv isn't activated — re-run `venv\Scripts\activate` then `pip install -r requirements.txt`. |
| `ModuleNotFoundError: No module named 'tkinter'` | Reinstall Python with the "tcl/tk and IDLE" option checked. |
| Window opens but freezes at 100×100 | It isn't frozen — a full 100×100 run is ~1,000,000 multiplications; give it a few seconds, or lower **Animation Speed** to Fast, or reduce thread/queue load by closing other apps. |
| Nothing happens on Start | Check the **Matrix Size** box has a number ≥ 2 and **Thread Count** ≥ 1. |

---

## Testing modes

| Mode | Size | Purpose | What to show faculty |
|---|---|---|---|
| Test 1 | 5×5 | See individual numbers and operations clearly | Set size=5, Speed=Slow, point at the equation panel updating one multiplication at a time, and the recent-operations log. |
| Test 2 | 10×10 | See animation + multithreading clearly | Set size=10, Speed=Medium, point at the Thread Activity panel — different `Worker-N` threads lighting up ACTIVE, and the highlighted row/column/cell moving across A, B, C. |
| Test 3 | 100×100 | Full assignment requirement | Set size=100, Thread Count=8, Speed=Fast (or Medium). Let it run to completion. Point at **Total operations = 1,000,000**, **Completed**, and the final **VERIFICATION PASSED** banner. |

---

## How it works (plain-language explanation)

1. **Matrix multiplication**: `C[i][j]` is the sum, over every `k`, of
   `A[i][k] * B[k][j]` — row `i` of A "dotted" with column `j` of B.
2. **Three indices**: `i` picks the row of A / row of the result, `j` picks
   the column of B / column of the result, `k` walks along that row/column
   pairing up the terms that get multiplied and summed.
3. **`C[i][j]`** is one number in the result matrix — the dot product of
   A's row `i` and B's column `j`.
4. **One multiplication operation** = one `A[i][k] * B[k][j]` — the
   smallest unit of work in this assignment, and the thing we put on
   threads.
5. **Why threads**: the assignment requires concurrency — instead of doing
   the n³ multiplications one after another, many run in parallel across
   worker threads.
6. **`ThreadPoolExecutor`**: a fixed pool of reusable worker threads; you
   `submit()` callables (tasks) and it hands each one to a free worker,
   queuing the rest internally.
7. **Why not 1,000,000 real threads**: OS threads are expensive (memory,
   context-switch overhead); creating a million would crash or badly stall
   most machines. A small pool (e.g. 8) is reused instead.
8. **Every multiplication → a task**: for each `(i, j, k)` the driver calls
   `executor.submit(multiply_scalar, A, B, i, j, k)` — one task per
   multiplication, literally.
9. **Worker threads execute tasks**: each free worker thread picks the next
   queued task, runs `multiply_scalar`, and returns `(i, j, k, a, b,
   product, thread_name)`.
10. **Combining partial products**: for a fixed `(i, j)`, the driver sums
    every `product` for `k = 0..n-1` into `row_partial[j]`.
11. **Final `C[i][j]`**: once all `k` for that row are done,
    `C[i][j] = row_partial[j]` is written once into the result matrix.
12. **Thread safety**: only the single driver thread ever writes to `C` —
    worker threads only compute and return a value, they never share mutable
    state with each other, so there's nothing to lock.
13. **Why `queue.Queue`**: it's the standard thread-safe hand-off — one
    thread `put()`s, another `get()`s, with the locking already handled
    internally.
14. **Why Tkinter updates only on the main thread**: Tkinter (like most GUI
    toolkits) is not thread-safe — calling widget methods from a worker
    thread can corrupt internal state or crash the app.
15. **`after()`**: schedules a callback to run once on the main thread after
    a delay, without blocking it — used here to repeatedly "poll" the queue
    on a timer, which is how the animation stays smooth and non-blocking.
16. **Animation reflects the real computation**: the GUI never fakes
    numbers — every value it displays came out of an actual
    `multiply_scalar()` call that really ran on a worker thread; the queue
    only controls *how fast* those real results are shown, never *what*
    they are.
17. **Progress**: `completed_ops / total_ops` (`total_ops = n³`), updated by
    the driver thread as futures complete — independent of GUI drawing
    speed.
18. **Verification**: after the threaded run finishes, `A.astype(int64) @
    B.astype(int64)` (NumPy) is compared element-by-element
    (`np.array_equal`) against the threaded result `C`.

---

## Viva questions & answers

1. **What is matrix multiplication?** Combining two matrices A (m×n) and B
   (n×p) into C (m×p) where each `C[i][j]` is the dot product of A's row
   `i` and B's column `j`.
2. **Why are three loops/indices required?** `i` and `j` index the output
   cell, `k` indexes the terms being summed to produce that cell.
3. **What is a thread?** A separate, lighter-weight unit of execution within
   a process that can run concurrently with other threads.
4. **What is multithreading?** Running multiple threads at once so
   independent work happens concurrently instead of strictly sequentially.
5. **What is `ThreadPoolExecutor`?** A Python `concurrent.futures` class
   that manages a fixed pool of reusable worker threads and a queue of
   pending callables.
6. **Why use a thread pool instead of raw threads?** Creating/destroying OS
   threads is expensive; a pool reuses a small, bounded number of them
   across a huge number of small tasks.
7. **What is a worker thread?** One of the pool's threads that actually
   executes submitted tasks.
8. **What is a task?** A callable (here, `multiply_scalar(A, B, i, j, k)`)
   submitted to the executor for eventual execution.
9. **Thread vs. task?** A thread is the execution resource; a task is the
   unit of work handed to whichever thread is free.
10. **Why not create 1,000,000 threads?** Massive memory/OS overhead and
    context-switching cost — the machine would likely hang or crash;
    reusing ~8 threads for 1,000,000 tasks is far more efficient.
11. **What is a race condition?** A bug where multiple threads read/write
    shared data at the same time without coordination, producing
    unpredictable results.
12. **How is thread safety maintained here?** Worker threads never share
    mutable state with each other; only the single driver thread writes to
    the result matrix `C`, so there's no concurrent write to guard.
13. **Why `queue.Queue`?** It's Python's built-in thread-safe FIFO for
    passing data between threads without manual locking.
14. **Why can't worker threads update Tkinter directly?** Tkinter isn't
    thread-safe; only the main thread may touch widgets safely.
15. **What does `after()` do?** Schedules a function to run once, after a
    delay, on the Tkinter main event loop — used to repeatedly poll the
    result queue without blocking the GUI.
16. **How is the animation implemented?** `after()` periodically drains a
    batch of real results from the queue and updates the canvases/labels
    to match.
17. **How are partial products calculated?** Each worker computes one
    `A[i][k] * B[k][j]`; the driver sums these for fixed `(i, j)` across all
    `k`.
18. **How is the final result obtained?** Once all `k` for a row are summed,
    each `C[i][j]` is written into the result matrix.
19. **How is correctness verified?** By comparing the threaded `C` against
    `A @ B` computed with NumPy, using `np.array_equal`.
20. **Why use NumPy at all?** For fast, convenient storage and indexing of
    matrices, and as an independent, trusted reference for verification.
21. **Why isn't NumPy used for the actual multiplication?** The assignment
    requires every scalar multiplication to run as its own threaded task —
    `A @ B` would compute everything internally in optimized C code,
    hiding the required per-operation threading.
22. **What happens if the thread count is increased?** More multiplications
    can run truly in parallel (up to the CPU's core count), which can speed
    up the run — but past a point, thread-switching overhead outweighs the
    benefit.
