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

