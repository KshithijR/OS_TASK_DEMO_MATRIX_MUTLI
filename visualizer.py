"""Tkinter GUI: renders matrices as heatmaps, animates the current
multiplication, shows thread activity, and drives the MultiplicationEngine.

Threading rule enforced throughout this file: worker threads NEVER touch a
Tkinter widget. All engine state is read here, on the main thread, inside
periodic root.after() callbacks that drain the engine's thread-safe queue.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional

from matrix_generator import generate_matrices
from multiplication_engine import MultiplicationEngine
from models import STATUS_IDLE, STATUS_RUNNING, STATUS_PAUSED, STATUS_STOPPED, STATUS_DONE

BG = "#0f172a"
PANEL_BG = "#1e293b"
PANEL_BG2 = "#243044"
FG = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
GREEN = "#4ade80"
AMBER = "#fbbf24"
RED = "#f87171"

MATRIX_PX = 260  # canvas size (square) for each matrix panel
TEXT_THRESHOLD = 12  # show numeric labels only when n <= this


def _lerp_color(v: float, vmax: float, base=(30, 41, 59), top=(56, 189, 248)) -> str:
    ratio = 0.0 if vmax <= 0 else max(0.0, min(1.0, v / vmax))
    r = int(base[0] + ratio * (top[0] - base[0]))
    g = int(base[1] + ratio * (top[1] - base[1]))
    b = int(base[2] + ratio * (top[2] - base[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


class MatrixPanel(ttk.Frame):
    """One heatmap canvas for a single matrix (A, B, or C), with cheap
    row/column/cell highlight overlays that are repositioned (not
    recreated) on every animation tick."""

    def __init__(self, parent, title: str, vmax_fn):
        super().__init__(parent, style="Panel.TFrame")
        self.vmax_fn = vmax_fn
        self.n = 0
        self.cell = 1.0
        self.rects: List[List[int]] = []
        self.texts: List[List[Optional[int]]] = []

        ttk.Label(self, text=title, style="PanelTitle.TLabel").pack(anchor="w", padx=8, pady=(8, 2))
        self.canvas = tk.Canvas(self, width=MATRIX_PX, height=MATRIX_PX, bg=PANEL_BG2,
                                 highlightthickness=0)
        self.canvas.pack(padx=8, pady=(0, 8))

        self.row_hl = self.canvas.create_rectangle(0, 0, 0, 0, outline=AMBER, width=2, state="hidden")
        self.col_hl = self.canvas.create_rectangle(0, 0, 0, 0, outline=GREEN, width=2, state="hidden")
        self.cell_hl = self.canvas.create_rectangle(0, 0, 0, 0, outline=RED, width=2, state="hidden")

    def build(self, matrix):
        self.canvas.delete("cell")
        n = matrix.shape[0]
        self.n = n
        self.cell = MATRIX_PX / n
        self.rects = [[0] * n for _ in range(n)]
        self.texts = [[None] * n for _ in range(n)]
        vmax = self.vmax_fn()
        show_text = n <= TEXT_THRESHOLD
        for r in range(n):
            for c in range(n):
                x0, y0 = c * self.cell, r * self.cell
                x1, y1 = x0 + self.cell, y0 + self.cell
                color = _lerp_color(float(matrix[r, c]), vmax)
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=PANEL_BG2,
                                                      width=1, tags="cell")
                self.rects[r][c] = rect
                if show_text:
                    txt = self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(int(matrix[r, c])),
                                                    fill=FG, font=("Consolas", 9), tags="cell")
                    self.texts[r][c] = txt
        for hl in (self.row_hl, self.col_hl, self.cell_hl):
            self.canvas.tag_raise(hl)

    def update_row(self, matrix, row: int):
        if self.n == 0:
            return
        vmax = self.vmax_fn()
        show_text = self.n <= TEXT_THRESHOLD
        for c in range(self.n):
            val = float(matrix[row, c])
            self.canvas.itemconfig(self.rects[row][c], fill=_lerp_color(val, vmax))
            if show_text and self.texts[row][c] is not None:
                self.canvas.itemconfig(self.texts[row][c], text=str(int(val)))

    def highlight_row(self, row: Optional[int]):
        if row is None or self.n == 0:
            self.canvas.itemconfig(self.row_hl, state="hidden")
            return
        y0, y1 = row * self.cell, (row + 1) * self.cell
        self.canvas.coords(self.row_hl, 0, y0, MATRIX_PX, y1)
        self.canvas.itemconfig(self.row_hl, state="normal")
        self.canvas.tag_raise(self.row_hl)

    def highlight_col(self, col: Optional[int]):
        if col is None or self.n == 0:
            self.canvas.itemconfig(self.col_hl, state="hidden")
            return
        x0, x1 = col * self.cell, (col + 1) * self.cell
        self.canvas.coords(self.col_hl, x0, 0, x1, MATRIX_PX)
        self.canvas.itemconfig(self.col_hl, state="normal")
        self.canvas.tag_raise(self.col_hl)

    def highlight_cell(self, row: Optional[int], col: Optional[int]):
        if row is None or col is None or self.n == 0:
            self.canvas.itemconfig(self.cell_hl, state="hidden")
            return
        x0, y0 = col * self.cell, row * self.cell
        x1, y1 = x0 + self.cell, y0 + self.cell
        self.canvas.coords(self.cell_hl, x0, y0, x1, y1)
        self.canvas.itemconfig(self.cell_hl, state="normal")
        self.canvas.tag_raise(self.cell_hl)


class MatrixVisualizerApp(ttk.Frame):
    SPEED_MS = {"Slow": 250, "Medium": 90, "Fast": 30}
    SPEED_BATCH = {"Slow": 15, "Medium": 60, "Fast": 250}
    MAX_LOG_LINES = 200

    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.root = root
        self.root.title("Multithreaded Matrix Multiplication Visualizer")
        self.root.configure(bg=BG)
        self.root.minsize(1180, 780)

        self._setup_style()

        self.engine: Optional[MultiplicationEngine] = None
        self.a = None
        self.b = None
        self.seed = 42
        self._poll_job = None

        self._build_layout()
        self._new_matrices(initial=True)

    # ------------------------------------------------------------------ #
    # Style
    # ------------------------------------------------------------------ #
    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("Panel2.TFrame", background=PANEL_BG2)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=PANEL_BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=PANEL_BG, foreground=ACCENT,
                         font=("Segoe UI Semibold", 11))
        style.configure("Title.TLabel", background=BG, foreground=FG, font=("Segoe UI Semibold", 18))
        style.configure("Muted.TLabel", background=PANEL_BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=PANEL_BG, foreground=GREEN, font=("Segoe UI Semibold", 12))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TEntry", fieldbackground=PANEL_BG2, foreground=FG)
        style.configure("TRadiobutton", background=PANEL_BG, foreground=FG)
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL_BG2, background=ACCENT)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(14, 6))
        ttk.Label(header, text="⛓  Multithreaded Matrix Multiplication Visualizer", style="Title.TLabel").pack(side="left")
        self.status_label = ttk.Label(header, text="STATUS: IDLE", style="Title.TLabel", foreground=MUTED)
        self.status_label.pack(side="right")

        self._build_controls()

        matrices_row = ttk.Frame(self)
        matrices_row.pack(fill="x", padx=16, pady=8)

        self.panel_a = MatrixPanel(matrices_row, "Matrix A", lambda: 9.0)
        self.panel_a.pack(side="left", padx=6)
        ttk.Label(matrices_row, text="×", style="Title.TLabel").pack(side="left", padx=4)
        self.panel_b = MatrixPanel(matrices_row, "Matrix B", lambda: 9.0)
        self.panel_b.pack(side="left", padx=6)
        ttk.Label(matrices_row, text="=", style="Title.TLabel").pack(side="left", padx=4)
        self.panel_c = MatrixPanel(matrices_row, "Matrix C (result)", self._c_vmax)
        self.panel_c.pack(side="left", padx=6)

        info_row = ttk.Frame(self)
        info_row.pack(fill="both", expand=True, padx=16, pady=8)

        self._build_operation_panel(info_row)
        self._build_thread_panel(info_row)
        self._build_log_panel(info_row)

        self._build_progress_bar()

    def _build_controls(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=16, pady=6)

        self.btn_start = ttk.Button(bar, text="▶ Start", command=self.on_start)
        self.btn_start.pack(side="left", padx=3)
        self.btn_pause = ttk.Button(bar, text="⏸ Pause", command=self.on_pause, state="disabled")
        self.btn_pause.pack(side="left", padx=3)
        self.btn_resume = ttk.Button(bar, text="⏵ Resume", command=self.on_resume, state="disabled")
        self.btn_resume.pack(side="left", padx=3)
        self.btn_stop = ttk.Button(bar, text="⏹ Stop", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=3)
        self.btn_reset = ttk.Button(bar, text="⟲ Reset", command=self.on_reset)
        self.btn_reset.pack(side="left", padx=3)

        ttk.Label(bar, text="   Matrix Size:").pack(side="left", padx=(16, 2))
        self.size_var = tk.StringVar(value="100")
        ttk.Entry(bar, textvariable=self.size_var, width=6).pack(side="left")

        ttk.Label(bar, text="   Thread Count:").pack(side="left", padx=(16, 2))
        self.threads_var = tk.StringVar(value="8")
        ttk.Entry(bar, textvariable=self.threads_var, width=4).pack(side="left")

        ttk.Label(bar, text="   Speed:").pack(side="left", padx=(16, 2))
        self.speed_var = tk.StringVar(value="Medium")
        for s in ("Slow", "Medium", "Fast"):
            ttk.Radiobutton(bar, text=s, value=s, variable=self.speed_var).pack(side="left")

        self.btn_new = ttk.Button(bar, text="⟳ Generate New Matrices", command=self.on_new_matrices)
        self.btn_new.pack(side="right", padx=3)

    def _build_operation_panel(self, parent):
        f = ttk.Frame(parent, style="Panel.TFrame")
        f.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ttk.Label(f, text="Current Operation", style="PanelTitle.TLabel").pack(anchor="w", padx=10, pady=(10, 4))

        self.op_eq = ttk.Label(f, text="A[-][-] × B[-][-]", style="Panel.TLabel", font=("Consolas", 13, "bold"))
        self.op_eq.pack(anchor="w", padx=10, pady=2)
        self.op_values = ttk.Label(f, text="— × — = —", style="Panel.TLabel", font=("Consolas", 12))
        self.op_values.pack(anchor="w", padx=10, pady=2)
        self.op_thread = ttk.Label(f, text="Executed by: —", style="Muted.TLabel")
        self.op_thread.pack(anchor="w", padx=10, pady=(2, 10))

        ttk.Separator(f).pack(fill="x", padx=10, pady=4)

        grid = ttk.Frame(f, style="Panel.TFrame")
        grid.pack(fill="x", padx=10, pady=6)
        self.stat_labels = {}
        rows = [
            ("Total operations", "total"),
            ("Completed", "completed"),
            ("Progress", "progress"),
            ("Elapsed time", "elapsed"),
            ("Active threads", "active"),
            ("Verification", "verify"),
        ]
        for idx, (label, key) in enumerate(rows):
            ttk.Label(grid, text=label + ":", style="Muted.TLabel").grid(row=idx, column=0, sticky="w", pady=2)
            val = ttk.Label(grid, text="—", style="Panel.TLabel", font=("Consolas", 10, "bold"))
            val.grid(row=idx, column=1, sticky="w", padx=(8, 0), pady=2)
            self.stat_labels[key] = val

    def _build_thread_panel(self, parent):
        f = ttk.Frame(parent, style="Panel.TFrame")
        f.pack(side="left", fill="both", expand=True, padx=6)
        ttk.Label(f, text="Thread Activity", style="PanelTitle.TLabel").pack(anchor="w", padx=10, pady=(10, 4))
        self.thread_canvas = tk.Canvas(f, bg=PANEL_BG2, highlightthickness=0, height=300)
        self.thread_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _build_log_panel(self, parent):
        f = ttk.Frame(parent, style="Panel.TFrame")
        f.pack(side="left", fill="both", expand=True, padx=(6, 0))
        ttk.Label(f, text="Recent Operations", style="PanelTitle.TLabel").pack(anchor="w", padx=10, pady=(10, 4))
        self.log_text = tk.Text(f, bg=PANEL_BG2, fg=FG, insertbackground=FG, font=("Consolas", 9),
                                 height=16, borderwidth=0, highlightthickness=0, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _build_progress_bar(self):
        f = ttk.Frame(self)
        f.pack(fill="x", padx=16, pady=(0, 14))
        self.progress = ttk.Progressbar(f, style="Horizontal.TProgressbar", maximum=1000)
        self.progress.pack(fill="x", side="left", expand=True)
        self.progress_pct = ttk.Label(f, text="0.00%", width=9)
        self.progress_pct.pack(side="left", padx=(8, 0))

    # ------------------------------------------------------------------ #
    # Matrix (re)generation
    # ------------------------------------------------------------------ #
    def _read_size(self) -> int:
        try:
            n = int(self.size_var.get())
        except ValueError:
            n = 10
        return max(2, min(n, 200))

    def _read_threads(self) -> int:
        try:
            t = int(self.threads_var.get())
        except ValueError:
            t = 8
        return max(1, min(t, 64))

    def _c_vmax(self) -> float:
        n = self.panel_c.n or 1
        return float(n * 9 * 9)

    def _new_matrices(self, initial: bool = False):
        n = self._read_size()
        self.seed = self.seed + 1 if not initial else 42
        self.a, self.b = generate_matrices(n, seed=self.seed)
        self.panel_a.build(self.a)
        self.panel_b.build(self.b)
        import numpy as np
        self.panel_c.build(np.zeros((n, n), dtype="int64"))
        self._reset_stats(n)

    def _reset_stats(self, n: int):
        self.stat_labels["total"].configure(text=f"{n**3:,}")
        self.stat_labels["completed"].configure(text="0")
        self.stat_labels["progress"].configure(text="0.00%")
        self.stat_labels["elapsed"].configure(text="0.00 s")
        self.stat_labels["active"].configure(text="0")
        self.stat_labels["verify"].configure(text="—")
        self.progress["value"] = 0
        self.progress_pct.configure(text="0.00%")
        self.op_eq.configure(text="A[-][-] × B[-][-]")
        self.op_values.configure(text="— × — = —")
        self.op_thread.configure(text="Executed by: —")
        self.thread_canvas.delete("all")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.status_label.configure(text="STATUS: IDLE", foreground=MUTED)
        for p in (self.panel_a, self.panel_b, self.panel_c):
            p.highlight_row(None)
            p.highlight_col(None)
            p.highlight_cell(None, None)

    # ------------------------------------------------------------------ #
    # Button handlers
    # ------------------------------------------------------------------ #
    def on_new_matrices(self):
        if self.engine is not None and self.engine.status in (STATUS_RUNNING, STATUS_PAUSED):
            messagebox.showinfo("Busy", "Stop or wait for the current run to finish first.")
            return
        self._new_matrices()

    def on_start(self):
        if self.engine is not None and self.engine.status in (STATUS_RUNNING, STATUS_PAUSED):
            return
        n = self._read_size()
        if self.a is None or self.a.shape[0] != n:
            self._new_matrices()
        threads = self._read_threads()
        self.engine = MultiplicationEngine(self.a, self.b, thread_count=threads, queue_maxsize=800)
        self.engine.start()
        self.status_label.configure(text="STATUS: RUNNING", foreground=GREEN)
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal")
        self.btn_resume.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_new.configure(state="disabled")
        self._schedule_poll()

    def on_pause(self):
        if self.engine:
            self.engine.pause()
            self.status_label.configure(text="STATUS: PAUSED", foreground=AMBER)
            self.btn_pause.configure(state="disabled")
            self.btn_resume.configure(state="normal")

    def on_resume(self):
        if self.engine:
            self.engine.resume()
            self.status_label.configure(text="STATUS: RUNNING", foreground=GREEN)
            self.btn_pause.configure(state="normal")
            self.btn_resume.configure(state="disabled")

    def on_stop(self):
        if self.engine:
            self.engine.stop()
        self.btn_pause.configure(state="disabled")
        self.btn_resume.configure(state="disabled")
        self.btn_stop.configure(state="disabled")

    def on_reset(self):
        if self.engine is not None and self.engine.status in (STATUS_RUNNING, STATUS_PAUSED):
            self.engine.stop()
        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
            self._poll_job = None
        self.engine = None
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        self.btn_resume.configure(state="disabled")
        self.btn_stop.configure(state="disabled")
        self.btn_new.configure(state="normal")
        self._new_matrices()

    # ------------------------------------------------------------------ #
    # Animation loop (main thread only)
    # ------------------------------------------------------------------ #
    def _schedule_poll(self):
        interval = self.SPEED_MS[self.speed_var.get()]
        self._poll_job = self.root.after(interval, self._poll)

    def _poll(self):
        eng = self.engine
        if eng is None:
            return
        batch = self.SPEED_BATCH[self.speed_var.get()]
        items = eng.drain_queue(batch)

        if items:
            last = items[-1]
            self.panel_a.highlight_row(last.i)
            self.panel_b.highlight_col(last.j)
            self.panel_c.highlight_cell(last.i, last.j)
            self.op_eq.configure(text=f"A[{last.i}][{last.k}] × B[{last.k}][{last.j}]")
            self.op_values.configure(text=f"{last.a_val} × {last.b_val} = {last.product}")
            self.op_thread.configure(text=f"Executed by: {last.thread_name}")

            self.panel_c.update_row(eng.c, last.i)

            self._append_log(items)

        self._update_stats(eng)
        self._update_thread_activity(eng)

        if eng.status in (STATUS_DONE, STATUS_STOPPED) and eng.result_queue.empty():
            self._finish(eng)
            return

        self._schedule_poll()

    def _append_log(self, items):
        self.log_text.configure(state="normal")
        for op in items[-8:]:
            self.log_text.insert("end", f"{op.thread_name:>10}  A[{op.i}][{op.k}] × B[{op.k}][{op.j}] = {op.product}\n")
        # trim
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > self.MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{line_count - self.MAX_LOG_LINES}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_stats(self, eng: MultiplicationEngine):
        pct = eng.progress_fraction * 100
        self.stat_labels["total"].configure(text=f"{eng.total_ops:,}")
        self.stat_labels["completed"].configure(text=f"{eng.completed_ops:,}")
        self.stat_labels["progress"].configure(text=f"{pct:.2f}%")
        self.stat_labels["elapsed"].configure(text=f"{eng.elapsed:.2f} s")
        self.progress["value"] = eng.progress_fraction * 1000
        self.progress_pct.configure(text=f"{pct:.2f}%")

    def _update_thread_activity(self, eng: MultiplicationEngine):
        c = self.thread_canvas
        c.delete("all")
        names = sorted(eng.thread_counts.keys())
        if not names:
            return
        maxc = max(eng.thread_counts.values()) or 1
        now = time.perf_counter()
        h = 26
        pad = 6
        width = c.winfo_width() or 300
        self.stat_labels["active"].configure(
            text=str(sum(1 for t in names if now - eng.thread_last_seen.get(t, 0) < 0.4))
        )
        for idx, name in enumerate(names):
            y = pad + idx * (h + 6)
            count = eng.thread_counts[name]
            active = (now - eng.thread_last_seen.get(name, 0)) < 0.4
            bar_w = max(4, (count / maxc) * (width - 140))
            color = GREEN if active else MUTED
            c.create_text(6, y + h / 2, text=name, anchor="w", fill=FG, font=("Consolas", 9))
            c.create_rectangle(90, y + 4, 90 + bar_w, y + h - 4, fill=color, outline="")
            state_text = "ACTIVE" if active else "WAITING"
            c.create_text(width - 8, y + h / 2, text=f"{state_text} ({count})", anchor="e",
                           fill=color, font=("Consolas", 8))

    def _finish(self, eng: MultiplicationEngine):
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        self.btn_resume.configure(state="disabled")
        self.btn_stop.configure(state="disabled")
        self.btn_new.configure(state="normal")
        for p in (self.panel_a, self.panel_b, self.panel_c):
            p.highlight_row(None)
            p.highlight_col(None)
        self.panel_c.highlight_cell(None, None)

        if eng.status == STATUS_DONE:
            passed = eng.verification_passed
            self.status_label.configure(
                text=f"STATUS: DONE — {'VERIFICATION PASSED' if passed else 'VERIFICATION FAILED'}",
                foreground=GREEN if passed else RED,
            )
            self.stat_labels["verify"].configure(text="PASSED ✓" if passed else "FAILED ✗")
        else:
            self.status_label.configure(text="STATUS: STOPPED", foreground=RED)
            self.stat_labels["verify"].configure(text="—")
