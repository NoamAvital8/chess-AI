"""A weakened Stockfish sparring partner: random legal moves mixed with shallow engine moves.

The trainer only learns from its own searches and from game results; the engine's
moves are never used as training targets.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import chess
import chess.engine
import numpy as np

# Each finished game nudges the randomness so the model scores about 50%: a win makes
# the opponent stronger, a loss weaker (a stochastic-approximation update).
RANDOMNESS_STEP = 0.05
START_RANDOMNESS = 0.9


def find_stockfish(setting: str = "auto") -> str | None:
    """Resolve ``--stockfish``: a path, ``auto`` (PATH, then a winget install) or ``none``."""
    if setting.lower() == "none":
        return None
    if setting.lower() != "auto":
        if not Path(setting).is_file():
            raise FileNotFoundError(f"No Stockfish executable at {setting}")
        return setting
    found = shutil.which("stockfish")
    if found:
        return found
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for candidate in sorted(packages.glob("Stockfish.Stockfish*/**/stockfish*.exe")):
        return str(candidate)
    return None


def adapt_randomness(randomness: float, model_scores: list[float]) -> float:
    """Move towards the randomness at which the model's expected score is 0.5."""
    for score in model_scores:
        randomness += RANDOMNESS_STEP * (0.5 - score)
    return float(min(1.0, max(0.0, randomness)))


class Stockfish:
    """One engine process shared by every game; full strength, limited by search depth."""

    def __init__(self, path: str, depth: int = 1):
        if depth < 1:
            raise ValueError("stockfish depth must be >= 1")
        self.path, self.depth = path, depth
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        # One thread and a fresh hash per game keep depth-limited moves reproducible.
        self.engine.configure({"Threads": 1, "Hash": 16})

    def player(self, randomness: float) -> "StockfishPlayer":
        return StockfishPlayer(self, randomness)

    def best_move(self, board: chess.Board, game: object) -> chess.Move:
        return self.engine.play(board, chess.engine.Limit(depth=self.depth), game=game).move

    def close(self) -> None:
        self.engine.quit()


class StockfishPlayer:
    """Plays a uniformly random legal move with probability ``randomness``."""

    def __init__(self, engine: Stockfish, randomness: float):
        if not 0 <= randomness <= 1:
            raise ValueError("stockfish randomness must be between 0 and 1")
        self.engine, self.randomness = engine, randomness
        self.label = f"stockfish@{randomness:.2f}"

    def choose_move(self, board: chess.Board, rng: np.random.Generator,
                    game: object) -> chess.Move:
        if rng.random() < self.randomness:
            legal = list(board.legal_moves)
            return legal[int(rng.integers(len(legal)))]
        return self.engine.best_move(board, game)
