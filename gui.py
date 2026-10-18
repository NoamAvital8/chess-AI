"""Native click-to-move board backed by the existing shared-model play pipeline."""

from __future__ import annotations

import argparse
import queue
import threading
from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class BoardLayout:
    left: float
    top: float
    cell: float
    flipped: bool = False

    @classmethod
    def fit(cls, width: float, height: float, flipped: bool = False):
        cell = max(1., min(width - 48, height - 48) / 8)
        return cls((width - 8 * cell) / 2, (height - 8 * cell) / 2, cell, flipped)

    def square_at(self, x: float, y: float) -> chess.Square | None:
        column, row = int((x - self.left) // self.cell), int((y - self.top) // self.cell)
        if not (0 <= column < 8 and 0 <= row < 8):
            return None
        return chess.square(7 - column if self.flipped else column,
                            row if self.flipped else 7 - row)

    def center(self, square: chess.Square) -> tuple[float, float]:
        file, rank = chess.square_file(square), chess.square_rank(square)
        column, row = (7 - file, rank) if self.flipped else (file, 7 - rank)
        return self.left + (column + .5) * self.cell, self.top + (row + .5) * self.cell


class BoardInteraction:
    def __init__(self, board: chess.Board, human_color: chess.Color):
        self.board, self.human_color, self.selected = board, human_color, None

    @property
    def destinations(self) -> set[chess.Square]:
        return {move.to_square for move in self.board.legal_moves if move.from_square == self.selected}

    def click(self, square: chess.Square | None) -> list[chess.Move]:
        if square is None:
            self.selected = None
            return []
        if self.board.turn != self.human_color:
            return []
        piece = self.board.piece_at(square)
        if piece is not None and piece.color == self.human_color:
            self.selected = None if self.selected == square else square
            return []
        moves = [move for move in self.board.legal_moves
                 if move.from_square == self.selected and move.to_square == square]
        if moves:
            self.selected = None
        return moves


class WindowBridge:
    """Worker/UI communication; every Tk call stays on the main thread."""
    def __init__(self):
        self.events = queue.Queue()
        self.responses = queue.Queue()
        self.cancelled = threading.Event()

    def position(self, board, human_color, status):
        self.events.put(("position", (board.copy(stack=True), human_color, status)))

    def output(self, text):
        self.events.put(("message", str(text)))

    def input(self, prompt):
        if self.cancelled.is_set():
            return "quit"
        self.events.put(("awaiting", None))
        while not self.cancelled.is_set():
            try:
                return self.responses.get(timeout=.1)
            except queue.Empty:
                pass
        return "quit"

    def submit(self, text):
        self.responses.put(text)

    def cancel(self):
        self.cancelled.set()


class ChessWindow:
    BG, PANEL, TEXT, MUTED, ACCENT = "#0d1522", "#172333", "#edf2f6", "#a5b6c8", "#55d6bd"
    LIGHT, DARK = "#e8dfcd", "#749587"
    GLYPHS = {chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝",
              chess.ROOK: "♜", chess.QUEEN: "♛", chess.KING: "♚"}

    def __init__(self, root, args):
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk, self.ttk, self.messagebox = tk, ttk, messagebox
        self.root, self.args = root, args
        self.worker, self.bridge, self.awaiting = None, None, False
        self.closing, self.ending, self.learning = False, False, False
        self.flipped = args.color == "black"
        self.interaction = BoardInteraction(chess.Board(), not self.flipped)
        self.layout = BoardLayout.fit(640, 640, self.flipped)
        root.title("Chess AI — play & learn")
        screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
        width, height = min(1060, screen_width - 50), min(820, screen_height - 80)
        root.geometry(f"{width}x{height}+{max(0, (screen_width - width) // 2)}+20")
        root.minsize(820, 580)
        root.configure(bg=self.BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", self.deselect)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Chess.Treeview", background=self.PANEL, fieldbackground=self.PANEL,
                        foreground=self.TEXT, rowheight=29, borderwidth=0, font=("Segoe UI", 11))
        style.configure("Chess.Treeview.Heading", background=self.PANEL, foreground=self.MUTED,
                        font=("Segoe UI", 10, "bold"), relief="flat")
        style.map("Chess.Treeview", background=[("selected", "#2d4b55")])
        style.configure("Chess.Horizontal.TProgressbar", troughcolor=self.PANEL,
                        background=self.ACCENT, borderwidth=0)

        header = tk.Frame(root, bg=self.BG)
        header.pack(fill="x", padx=28, pady=(20, 6))
        tk.Label(header, text="CHESS AI", bg=self.BG, fg=self.TEXT,
                 font=("Segoe UI", 23, "bold")).pack(side="left")
        tk.Label(header, text="One shared model · learns after completed games", bg=self.BG,
                 fg=self.MUTED, font=("Segoe UI", 11)).pack(side="right")
        self.status = tk.StringVar(value="Choose your color, then start a game.")
        tk.Label(root, textvariable=self.status, bg=self.BG, fg=self.ACCENT,
                 font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", padx=28, pady=(0, 10))
        self.result = tk.StringVar(value="")

        content = tk.Frame(root, bg=self.BG)
        content.pack(fill="both", expand=True, padx=16)
        board_panel = tk.Frame(content, bg=self.BG)
        board_panel.pack(side="left", fill="both", expand=True)
        self.player_label = tk.StringVar(value="You: White   ·   Model: Black")
        tk.Label(board_panel, textvariable=self.player_label, bg=self.BG, fg=self.MUTED,
                 font=("Segoe UI", 11), anchor="w").pack(fill="x", padx=16)
        tk.Label(board_panel, textvariable=self.result, bg=self.BG, fg=self.TEXT,
                 font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", padx=16)
        self.canvas = tk.Canvas(board_panel, width=650, height=650, bg=self.BG,
                                highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", self.deselect)

        sidebar = tk.Frame(content, width=278, bg=self.PANEL, padx=18, pady=18)
        sidebar.pack(side="right", fill="y", padx=(12, 10), pady=(4, 20))
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="PLAY A GAME", bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x")
        color_row = tk.Frame(sidebar, bg=self.PANEL)
        color_row.pack(fill="x", pady=(8, 8))
        tk.Label(color_row, text="Your color", bg=self.PANEL, fg=self.TEXT,
                 font=("Segoe UI", 11)).pack(side="left")
        self.color = tk.StringVar(value=args.color.title())
        self.color_picker = ttk.Combobox(color_row, textvariable=self.color,
                                        values=("White", "Black", "Alternate"), width=10, state="readonly")
        self.color_picker.pack(side="right")
        self.color_picker.bind("<<ComboboxSelected>>", self.preview_color)
        action_row = tk.Frame(sidebar, bg=self.PANEL)
        action_row.pack(fill="x", pady=(0, 16))
        self.new_button = self.button(action_row, "Start game", self.start, accent=True)
        self.new_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.button(action_row, "Flip board", self.flip).pack(side="left", fill="x", expand=True)
        tk.Label(sidebar, text="MOVES", bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x", pady=(0, 8))
        history_frame = tk.Frame(sidebar, bg=self.PANEL)
        history_frame.pack(fill="both", expand=True)
        self.history = ttk.Treeview(history_frame, columns=("number", "white", "black"),
                                   show="headings", style="Chess.Treeview", selectmode="none", height=4)
        for column, text, width in (("number", "#", 34), ("white", "White", 88), ("black", "Black", 88)):
            self.history.heading(column, text=text)
            self.history.column(column, width=width, stretch=column != "number", anchor="center")
        scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history.yview)
        self.history.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.history.pack(side="left", fill="both", expand=True)
        self.progress = ttk.Progressbar(sidebar, mode="indeterminate", style="Chess.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(12, 10))
        end_row = tk.Frame(sidebar, bg=self.PANEL)
        end_row.pack(fill="x")
        self.resign_button = self.button(end_row, "Resign", self.resign, danger=True)
        self.resign_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.end_button = self.button(end_row, "End game", self.end_game)
        self.end_button.pack(side="left", fill="x", expand=True)
        tk.Label(sidebar, text="Completed games train your model.\nEnding early saves without training.",
                 bg=self.PANEL, fg=self.MUTED, justify="left", wraplength=235,
                 font=("Segoe UI", 9)).pack(fill="x", pady=(14, 0))
        tk.Label(root, text="Click a piece, then a highlighted square. Right-click or Esc clears selection.",
                 bg=self.BG, fg=self.MUTED, font=("Segoe UI", 10)).pack(
                     side="bottom", before=content, fill="x", pady=(0, 12))
        self.controls()
        self.preview_color()
        self.root.after(40, self.poll)

    def button(self, parent, text, command, accent=False, danger=False):
        return self.tk.Button(parent, text=text, command=command, relief="flat", bd=0,
                              bg=self.ACCENT if accent else "#26384b",
                              fg=self.BG if accent else "#ffb8b8" if danger else self.TEXT,
                              activebackground="#88e4cf" if accent else "#365169",
                              activeforeground=self.BG if accent else self.TEXT,
                              disabledforeground="#61748a", font=("Segoe UI", 11, "bold"),
                              padx=12, pady=8, cursor="hand2")

    @property
    def active(self):
        return self.worker is not None

    def controls(self):
        ready = self.awaiting and not self.closing and not self.ending
        self.new_button.configure(state="disabled" if self.active else "normal",
                                  text="New game" if self.interaction.board.move_stack else "Start game")
        self.color_picker.configure(state="disabled" if self.active else "readonly")
        self.resign_button.configure(state="normal" if ready else "disabled")
        self.end_button.configure(state="normal" if self.active and not self.learning and not self.ending else "disabled")

    def preview_color(self, event=None):
        if self.active:
            return
        self.flipped = self.color.get() == "Black"
        self.interaction = BoardInteraction(chess.Board(), not self.flipped)
        self.result.set("")
        self.status.set("Choose your color, then start a game.")
        self.player_label.set(f"You: {'Black' if self.flipped else 'White'}   ·   Model: {'White' if self.flipped else 'Black'}")
        self.update_history()
        self.controls()
        self.draw()

    def flip(self):
        self.flipped = not self.flipped
        self.draw()

    def deselect(self, event=None):
        self.interaction.selected = None
        self.draw()

    def draw(self):
        canvas, board = self.canvas, self.interaction.board
        canvas.delete("all")
        self.layout = BoardLayout.fit(max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100), self.flipped)
        layout = self.layout
        last = board.peek() if board.move_stack else None
        destinations = self.interaction.destinations if self.awaiting else set()
        check_king = board.king(board.turn) if board.is_check() else None
        size = max(16, int(layout.cell * .69))
        for square in chess.SQUARES:
            x, y = layout.center(square)
            half = layout.cell / 2
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            color = self.LIGHT if light else self.DARK
            if last is not None and square in (last.from_square, last.to_square):
                color = "#dfcc7f" if light else "#aaa969"
            if square == check_king:
                color = "#d47f76"
            canvas.create_rectangle(x - half, y - half, x + half, y + half, fill=color, outline=color)
            if square == self.interaction.selected:
                canvas.create_rectangle(x - half + 2, y - half + 2, x + half - 2, y + half - 2,
                                        outline="#0b6258", width=4)
            piece = board.piece_at(square)
            if piece is not None:
                glyph = self.GLYPHS[piece.piece_type]
                # Filled silhouettes with an outline keep White distinct on light squares.
                if piece.color == chess.WHITE:
                    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                        canvas.create_text(x + dx, y + dy - 1, text=glyph,
                                           font=("Segoe UI Symbol", size), fill="#26333a")
                canvas.create_text(x, y - 1, text=glyph, font=("Segoe UI Symbol", size),
                                   fill="#fff8e9" if piece.color else "#1b2933")
            if square in destinations:
                radius = layout.cell * (.41 if piece else .095)
                canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                   fill="" if piece else "#285b50", outline="#285b50", width=3)
        for index in range(8):
            file = 7 - index if self.flipped else index
            rank = index if self.flipped else 7 - index
            canvas.create_text(layout.left + (index + .5) * layout.cell,
                               layout.top + 8 * layout.cell + 15, text=chess.FILE_NAMES[file],
                               fill=self.MUTED, font=("Segoe UI", 10))
            canvas.create_text(layout.left - 15, layout.top + (index + .5) * layout.cell,
                               text=str(rank + 1), fill=self.MUTED, font=("Segoe UI", 10))

    def click(self, event):
        if not self.awaiting or self.closing or self.ending:
            return
        moves = self.interaction.click(self.layout.square_at(event.x, event.y))
        self.draw()
        if not moves:
            return
        move = moves[0] if len(moves) == 1 else self.promotion(moves)
        if move is not None:
            self.awaiting = False
            self.status.set("Confirming your move...")
            self.progress.start(12)
            self.controls()
            self.bridge.submit(move.uci())

    def promotion(self, moves):
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Promote pawn")
        dialog.configure(bg=self.PANEL, padx=20, pady=20)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        selected = []
        self.tk.Label(dialog, text="Choose a promotion", bg=self.PANEL, fg=self.TEXT,
                      font=("Segoe UI", 13, "bold")).pack(pady=(0, 14))
        row = self.tk.Frame(dialog, bg=self.PANEL)
        row.pack()
        names = {chess.QUEEN: "Queen", chess.ROOK: "Rook", chess.BISHOP: "Bishop", chess.KNIGHT: "Knight"}
        for kind in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            move = next(move for move in moves if move.promotion == kind)
            def choose(chosen=move):
                selected.append(chosen)
                dialog.destroy()
            self.button(row, names[kind], choose).pack(side="left", padx=4)
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        dialog.grab_set()
        self.root.wait_window(dialog)
        return selected[0] if selected else None

    def start(self):
        if self.active or self.closing:
            return
        self.awaiting, self.ending, self.learning = False, False, False
        self.result.set("")
        self.bridge = WindowBridge()
        self.interaction = BoardInteraction(chess.Board(), self.color.get() != "Black")
        self.flipped = self.color.get() == "Black"
        self.update_history()
        args = argparse.Namespace(**{**vars(self.args), "color": self.color.get().lower(), "games": 1})
        self.status.set("Loading the shared model...")
        self.progress.start(12)
        bridge = self.bridge
        def work():
            try:
                from play import run
                run(args, input_fn=bridge.input, output=bridge.output, on_position=bridge.position,
                    cancel_requested=bridge.cancelled.is_set)
            except Exception as error:
                bridge.events.put(("error", str(error)))
            finally:
                bridge.events.put(("finished", None))
        self.worker = threading.Thread(target=work, name="chess-game", daemon=False)
        self.worker.start()
        self.controls()
        self.draw()

    def resign(self):
        if self.awaiting:
            self.awaiting = False
            self.status.set("Saving the result...")
            self.bridge.submit("resign")
            self.controls()

    def end_game(self):
        if self.active and not self.learning:
            self.awaiting, self.ending = False, True
            self.status.set("Saving unfinished game...")
            self.bridge.cancel()
            self.controls()

    def close(self):
        if not self.active:
            self.root.destroy()
            return
        self.closing, self.awaiting = True, False
        self.bridge.cancel()
        self.status.set("Finishing learning and saving before closing..." if self.learning else "Saving before closing...")
        self.controls()

    def update_history(self):
        self.history.delete(*self.history.get_children())
        board, cells = chess.Board(), []
        for move in self.interaction.board.move_stack:
            cells.append(board.san(move))
            board.push(move)
        last = None
        for offset in range(0, len(cells), 2):
            last = self.history.insert("", "end", values=(offset // 2 + 1, cells[offset],
                                       cells[offset + 1] if offset + 1 < len(cells) else ""))
        if last:
            self.history.see(last)

    def poll(self):
        finished, error = False, None
        if self.bridge is not None:
            while True:
                try:
                    event, value = self.bridge.events.get_nowait()
                except queue.Empty:
                    break
                if event == "position":
                    board, human, status = value
                    if human != self.interaction.human_color:
                        self.flipped = human == chess.BLACK
                    self.interaction = BoardInteraction(board, human)
                    self.player_label.set(f"You: {'White' if human else 'Black'}   ·   Model: {'Black' if human else 'White'}")
                    self.awaiting = False
                    if status.startswith("Result:"):
                        result = status.split()[1]
                        self.result.set("Game unfinished" if result == "*" else "Draw" if result == "1/2-1/2"
                                        else "You win" if (result == "1-0") == human else "Model wins")
                        if result != "*":
                            self.learning = True
                    if not self.closing and not self.ending:
                        self.status.set("Saving the game..." if status.startswith("Result:") else status)
                    self.update_history()
                    self.draw()
                elif event == "awaiting" and not self.closing and not self.ending:
                    self.awaiting = True
                    self.progress.stop()
                    self.status.set("Your move — click a piece, then its destination.")
                elif event == "message":
                    if value.startswith("Reanalyzing"):
                        self.learning = True
                        self.status.set("Learning from your completed game...")
                        self.progress.start(12)
                    elif value.startswith("Learned from"):
                        self.status.set("Saving the updated model...")
                    elif value.startswith("Updated model saved"):
                        self.status.set("Game saved. Your model is ready for another game.")
                    elif value.startswith("Unfinished game"):
                        self.status.set("Unfinished game saved. Learning skipped.")
                    elif value.startswith("No moves were played"):
                        self.status.set("No moves played; model unchanged.")
                    elif value.startswith("Model is thinking"):
                        self.progress.start(12)
                elif event == "error":
                    error = value
                    self.status.set("Could not finish this game. See the error message.")
                elif event == "finished":
                    finished = True
        if finished:
            self.worker, self.awaiting, self.learning, self.ending = None, False, False, False
            self.progress.stop()
            self.controls()
            self.draw()
            if self.closing:
                self.root.destroy()
                return
        if error is not None and not self.closing:
            self.messagebox.showerror("Chess AI", error, parent=self.root)
        self.controls()
        self.root.after(40, self.poll)


def run_window(args):
    try:
        import tkinter as tk
    except ImportError as error:
        raise RuntimeError("Graphical play requires Tkinter. Install Python with Tcl/Tk support.") from error
    root = tk.Tk()
    ChessWindow(root, args)
    root.mainloop()


def main():
    from play import build_parser
    args = build_parser().parse_args()
    args.gui = True
    run_window(args)


if __name__ == "__main__":
    main()
