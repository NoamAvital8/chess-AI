"""Double-click on Windows to play with the project's shared model."""

import os
from pathlib import Path


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    try:
        from gui import main
        main()
    except Exception as error:
        from tkinter import messagebox
        messagebox.showerror("Chess AI could not open", str(error))
