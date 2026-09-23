"""Entry point for the Multithreaded Matrix Multiplication Visualizer.

Run with:
    python main.py
"""

import tkinter as tk

from visualizer import MatrixVisualizerApp


def main():
    root = tk.Tk()
    app = MatrixVisualizerApp(root)
    app.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
