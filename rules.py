"""Game-ending rules shared by training, search, evaluation, play and recordings."""

from __future__ import annotations

import chess


def game_outcome(board: chess.Board) -> chess.Outcome | None:
    """The outcome, with threefold repetition and the 50-move rule applied automatically.

    These draws count only once they have actually happened. python-chess's
    ``claim_draw=True`` also treats a position as drawn when the side to move merely
    *could* repeat with its next move, which ends games (and search lines) before
    anything repeats, even for a side that is winning.
    """
    # Checkmate, stalemate, insufficient material, 75-move rule, fivefold repetition.
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        return outcome
    if board.halfmove_clock >= 100:
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    if board.is_repetition(3):
        return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    return None
