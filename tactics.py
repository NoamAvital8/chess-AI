"""Bounded proofs of mates made entirely with checks; no material evaluation."""

from __future__ import annotations

import chess

from rules import game_outcome


class _BudgetExceeded(Exception):
    pass


def _proves_mate(board: chess.Board, attacker: chess.Color, checks: int,
                 budget: list[int]) -> bool:
    budget[0] -= 1
    if budget[0] < 0:
        raise _BudgetExceeded
    outcome = game_outcome(board)
    if outcome is not None:
        return outcome.winner == attacker
    if checks <= 0:
        return False
    if board.turn == attacker:
        # Any forcing checking continuation is enough to prove an attack.
        for move in list(board.legal_moves):
            if not board.gives_check(move):
                continue
            board.push(move)
            try:
                if _proves_mate(board, attacker, checks - 1, budget):
                    return True
            finally:
                board.pop()
        return False
    # A proof must cover EVERY legal defense, not just the played continuation.
    for move in list(board.legal_moves):
        board.push(move)
        try:
            if not _proves_mate(board, attacker, checks, budget):
                return False
        finally:
            board.pop()
    return True


def tactical_moves(board: chess.Board, checks: int = 3,
                   nodes_per_move: int = 300) -> set[chess.Move]:
    """Prefer proved wins and exclude proved losses when alternatives exist.

    Failure to prove a mate (including exhaustion of the budget) means unknown,
    never a certified draw or safe position. Board/history are preserved exactly.
    """
    legal = list(board.legal_moves)
    if checks <= 0:
        return set(legal)
    if nodes_per_move < 1:
        raise ValueError("tactical node budget must be positive")
    player = board.turn
    wins = set()
    for move in legal:
        if not board.gives_check(move):
            continue
        board.push(move)
        try:
            if _proves_mate(board, player, checks - 1, [nodes_per_move]):
                wins.add(move)
        except _BudgetExceeded:
            pass
        finally:
            board.pop()
    if wins:
        return wins
    losing = set()
    for move in legal:
        board.push(move)
        try:
            if _proves_mate(board, not player, checks, [nodes_per_move]):
                losing.add(move)
        except _BudgetExceeded:
            pass
        finally:
            board.pop()
    return set(legal) - losing if len(losing) < len(legal) else set(legal)
