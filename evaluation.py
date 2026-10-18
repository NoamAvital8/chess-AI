"""Fixed-opening, color-balanced evaluation, isolated from training replay and RNGs."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch

from chess_ai import PolicyValueNet, check_format, run_batched, search_steps
from rules import game_outcome


# Each opening is played twice, with candidate colors swapped. The same suite and
# seeds are used at every iteration, so evaluation settings remain comparable.
OPENINGS = (
    (),
    ("e2e4", "e7e5", "g1f3", "b8c6"),
    ("d2d4", "d7d5", "c2c4"),
    ("c2c4", "e7e5", "b1c3"),
    ("e2e4", "c7c5", "g1f3"),
    ("d2d4", "g8f6", "c2c4", "e7e6"),
)


def load_opponent(path: Path, device: torch.device) -> PolicyValueNet:
    # Support both lightweight snapshots and the user's trusted full checkpoints.
    data = torch.load(path, map_location="cpu", weights_only=False)
    check_format(data, path)
    # Constructing an evaluation model must not consume the trainer's Torch RNG.
    with torch.random.fork_rng(devices=[]):
        network = PolicyValueNet(data["width"], data["blocks"])
    network.load_state_dict(data["network"])
    return network.to(device).eval()


def play_game_steps(candidate: PolicyValueNet, opponent: PolicyValueNet | None,
                    candidate_color: chess.Color, simulations: int, max_plies: int,
                    opening: tuple[str, ...], seed: int):
    """One evaluation game as a generator of search requests (see ``run_batched``)."""
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    rng = np.random.default_rng(seed)
    played = 0
    # max_plies counts moves after the fixed opening, equally for both colors.
    while played < max_plies and game_outcome(board) is None:
        player = candidate if board.turn == candidate_color else opponent
        if player is None:
            legal = list(board.legal_moves)
            move = legal[int(rng.integers(len(legal)))]
        else:
            move = (yield from search_steps(board, player, simulations, noise=False)).move
        board.push(move)
        played += 1
    outcome = game_outcome(board)
    if outcome is None:
        result = "truncated"
    elif outcome.winner is None:
        result = "draw"
    else:
        result = "win" if outcome.winner == candidate_color else "loss"
    return {"candidate_color": "white" if candidate_color else "black",
            "opening": list(opening), "seed": seed, "plies": played,
            "result": result, "pgn_result": outcome.result() if outcome else "*",
            "termination": outcome.termination.name if outcome else "MOVE_CAP"}


def play_game(candidate, opponent, candidate_color, device, simulations, max_plies,
              opening, seed) -> dict:
    return run_batched([play_game_steps(candidate, opponent, candidate_color, simulations,
                                        max_plies, opening, seed)], device)[0]


def evaluate(candidate: PolicyValueNet, opponent: PolicyValueNet | None,
             device: torch.device, games: int, simulations: int,
             max_plies: int, seed: int) -> dict:
    if games < 2 or games % 2:
        raise ValueError("evaluation games must be a positive even number (>= 2)")
    if simulations < 1 or max_plies < 1:
        raise ValueError("evaluation simulations and max-plies must be positive")
    models = [candidate] + ([opponent] if opponent is not None else [])
    modes = [model.training for model in models]
    try:
        for model in models:
            model.eval()
        # All games run together; leaves of the same model are evaluated in one batch.
        tasks = [play_game_steps(candidate, opponent, color, simulations, max_plies,
                                 OPENINGS[pair % len(OPENINGS)], seed + pair)
                 for pair in range(games // 2) for color in (chess.WHITE, chess.BLACK)]
        details = run_batched(tasks, device)
    finally:
        for model, training in zip(models, modes):
            model.train(training)
    wins = sum(game["result"] == "win" for game in details)
    draws = sum(game["result"] == "draw" for game in details)
    losses = sum(game["result"] == "loss" for game in details)
    truncated = games - wins - draws - losses
    completed = games - truncated
    points = wins + 0.5 * draws
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "truncated": truncated, "completed": completed,
            "completion_rate": completed / games,
            # A completed-only score can be biased when many games are truncated.
            # Bounds show the possible score over ALL games if they were finished.
            "completed_score": points / completed if completed else None,
            "score_lower_bound": points / games,
            "score_upper_bound": (points + truncated) / games,
            "simulations": simulations, "max_plies": max_plies, "seed": seed,
            "opening_suite": "fixed-v1", "game_details": details}
