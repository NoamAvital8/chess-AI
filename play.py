"""Play terminal chess against a frozen model, then learn after each completed game."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch

from chess_ai import (IncompatibleCheckpointError, PolicyValueNet, batch_split, batch_tensors,
                      check_format, encode, learn, load_checkpoint, make_optimizer, move_indices,
                      policy_cross_entropy, save_checkpoint, save_snapshot, search)
from recording import write_pgn
from rules import game_outcome
from training_metrics import MetricsWriter, replay_summary
from checkpoint_access import single_writer
from tactics import tactical_moves


@dataclass
class PlayedGame:
    moves: list[chess.Move]
    examples: list[tuple]
    result: str
    termination: str
    human_color: chess.Color

    @property
    def completed(self) -> bool:
        return self.result != "*"


def board_text(board: chess.Board, color: chess.Color) -> str:
    ranks = range(7, -1, -1) if color == chess.WHITE else range(8)
    files = list(range(8)) if color == chess.WHITE else list(range(7, -1, -1))
    rows = []
    for rank in ranks:
        pieces = [board.piece_at(chess.square(file, rank)) for file in files]
        rows.append(f"{rank + 1}  " + " ".join(piece.symbol() if piece else "." for piece in pieces))
    rows.append("   " + " ".join(chess.FILE_NAMES[file] for file in files))
    rows.append(f"{'White' if board.turn else 'Black'} to move" + (" (check)" if board.is_check() else ""))
    return "\n".join(rows)


def analyze_position(board, network, device, simulations, tactical_checks, tactical_nodes):
    allowed = tactical_moves(board, tactical_checks, tactical_nodes) if tactical_checks else None
    return search(board, network, device, simulations, noise=False, root_moves=allowed)


def reanalyze_game(game: PlayedGame, network, device, simulations: int,
                   tactical_checks: int, tactical_nodes: int) -> list[tuple]:
    """Replace pre-loss search targets using stronger, tactically checked search."""
    board, refreshed = chess.Board(), []
    for move, (_, _, _, value) in zip(game.moves, game.examples):
        result = analyze_position(board, network, device, simulations,
                                  tactical_checks, tactical_nodes)
        refreshed.append((encode(board), move_indices(board, result.moves),
                          result.policy.astype(np.float32), value))
        board.push(move)
    return refreshed


def refresh_matching_policies(replay: deque, examples: list[tuple]) -> int:
    """Refresh stale policy labels on identical inputs, preserving observed outcomes."""
    def key(example):
        return hashlib.sha256(example[0].tobytes() + example[1].tobytes()).digest()

    policies = {key(ex): ex[2] for ex in examples}
    count = 0
    for i, ex in enumerate(replay):
        policy = policies.get(key(ex))
        if policy is not None:
            replay[i] = (ex[0], ex[1], policy.copy(), ex[3])
            count += 1
    return count


@torch.no_grad()
def game_fit(network, examples, device) -> dict:
    """Fit on this game's refreshed targets; not a playing-strength measurement."""
    network.eval()
    policy_losses, squared_errors = [], []
    for start in range(0, len(examples), 64):
        batch = examples[start:start + 64]
        states, indices, targets, valid = batch_tensors(batch, device)
        logits, values = network(states)
        policy_losses += policy_cross_entropy(logits, indices, targets, valid).tolist()
        squared_errors += [(value - ex[3]) ** 2 for value, ex in zip(values.tolist(), batch)]
    return {"policy_loss": float(np.mean(policy_losses)), "value_loss": float(np.mean(squared_errors))}


def human_game(network: PolicyValueNet, device: torch.device, human_color: chess.Color,
               simulations: int, max_plies: int, *, input_fn=None, output=print,
               tactical_checks: int = 3, tactical_nodes: int = 300,
               on_position=None, cancel_requested=None) -> PlayedGame:
    """Search targets are generated before each move; weights stay fixed for the game."""
    if simulations < 1 or max_plies < 1:
        raise ValueError("simulations and max-plies must be positive")
    input_fn = input if input_fn is None else input_fn
    board = chess.Board()
    positions, moves = [], []
    result, termination, winner = "*", "MOVE_CAP", None
    network.eval()
    output("Enter SAN (e4, Nf3, O-O) or UCI (e2e4, e7e8q). Commands: legal, board, help, resign, quit.")
    try:
        while len(moves) < max_plies and game_outcome(board) is None:
            if cancel_requested is not None and cancel_requested():
                termination = "ABANDONED"
                break
            output(board_text(board, human_color))
            human_turn = board.turn == human_color
            if on_position is not None:
                on_position(board.copy(stack=True), human_color,
                            "Your move" if human_turn else "Model is thinking...")
            if human_turn:
                while True:
                    try:
                        text = input_fn("Your move: ").strip()
                    except EOFError:
                        text = "quit"
                    command = text.lower()
                    if command in ("quit", "exit"):
                        termination = "ABANDONED"
                        break
                    if command == "resign":
                        winner = not human_color
                        result = "1-0" if winner == chess.WHITE else "0-1"
                        termination = "RESIGNATION"
                        break
                    if command in ("help", "?"):
                        output("Moves: e4, Nf3, O-O, e2e4, e7e8q. Commands: legal, board, resign, quit.")
                        continue
                    if command == "board":
                        output(board_text(board, human_color))
                        continue
                    if command == "legal":
                        output("Legal moves: " + " ".join(board.san(move) for move in board.legal_moves))
                        continue
                    try:
                        # parse_san accepts coordinate moves as well; parse_uci is a
                        # fallback for explicit UCI forms. A null move is never legal.
                        try:
                            selected = board.parse_san(text)
                        except ValueError:
                            selected = board.parse_uci(text)
                        if selected not in board.legal_moves:
                            raise ValueError("Move is not legal")
                    except ValueError:
                        output("Illegal or unclear move. Enter 'legal' to see available moves.")
                        continue
                    break
                if termination in ("ABANDONED", "RESIGNATION"):
                    break
            else:
                output("Model is thinking...")
            searched = analyze_position(board, network, device, simulations,
                                        tactical_checks, tactical_nodes)
            if cancel_requested is not None and cancel_requested():
                termination = "ABANDONED"
                break
            positions.append((encode(board), move_indices(board, searched.moves),
                              searched.policy.astype(np.float32), board.turn))
            if not human_turn:
                selected = searched.move
                output(f"Model plays {board.san(selected)} ({selected.uci()})")
            board.push(selected)
            moves.append(selected)
    except KeyboardInterrupt:
        termination = "ABANDONED"
        output("\nGame interrupted; saving it as unfinished.")
        # A Ctrl+C between push() and recording a move must not create a PGN/result
        # mismatch. Only fully recorded moves belong to the saved game.
        board = chess.Board()
        for move in moves:
            board.push(move)
        positions = positions[:len(moves)]

    outcome = game_outcome(board)
    if outcome is not None:
        result, winner, termination = outcome.result(), outcome.winner, outcome.termination.name
    examples = [(x, indices, pi, None if result == "*" else float(
        0 if winner is None else (1 if color == winner else -1)))
        for x, indices, pi, color in positions]
    output(board_text(board, human_color))
    output(f"Result: {result} ({termination})")
    if on_position is not None:
        on_position(board.copy(stack=True), human_color, f"Result: {result} ({termination})")
    return PlayedGame(moves, examples, result, termination, human_color)


@single_writer
def run(args: argparse.Namespace, *, input_fn=None, output=print,
        on_position=None, cancel_requested=None) -> None:
    reanalysis_simulations = getattr(args, "reanalyze_simulations", 512)
    tactical_checks = getattr(args, "tactical_checks", 3)
    tactical_nodes = getattr(args, "tactical_nodes", 300)
    if reanalysis_simulations < 1 or not 0 <= tactical_checks <= 3 or tactical_nodes < 1:
        raise ValueError("reanalyze-simulations and tactical-nodes must be positive; tactical-checks must be 0..3")
    steps, game_passes = getattr(args, "steps", None), getattr(args, "game_passes", 1.0)
    if min(args.games, args.simulations, args.max_plies, args.batch_size,
           args.replay_size, args.human_replay_size, args.threads) < 1 or (
               steps is not None and steps < 1):
        raise ValueError("games, simulations, max-plies, steps, batch sizes and threads must be positive")
    if not np.isfinite(game_passes) or game_passes <= 0:
        raise ValueError("game-passes must be finite and positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not 0 < args.human_fraction <= 1:
        raise ValueError("human-fraction must be greater than 0 and at most 1 in play mode")
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    destination = Path(args.checkpoint_dir)
    checkpoint = destination / "latest.pt"
    source = checkpoint if checkpoint.is_file() else Path(args.checkpoint)
    if not source.is_file():
        raise FileNotFoundError(f"No model at {source}. Train a model first or pass --checkpoint.")
    # As elsewhere in this project, only load checkpoints you trust.
    data = torch.load(source, map_location="cpu", weights_only=False)
    check_format(data, source)
    width, blocks = data["width"], data["blocks"]
    network = PolicyValueNet(width, blocks).to(device)
    optimizer = make_optimizer(network, args.lr)
    rng, sampler = np.random.default_rng(args.seed), random.Random(args.seed)
    human_replay = deque(maxlen=args.human_replay_size)
    metadata = {"human_games": 0, "origin_snapshot": None}
    if "optimizer" in data:
        del data
        iteration, replay = load_checkpoint(source, network, optimizer, device, args.replay_size,
                                             rng, sampler, human_replay=human_replay, metadata=metadata)
    else:
        # Lightweight evaluation snapshots can also seed a human-play session.
        network.load_state_dict(data["network"])
        iteration, replay = data["iteration"], deque(maxlen=args.replay_size)
        del data
    for group in optimizer.param_groups:
        group["lr"] = args.lr
    metrics = MetricsWriter(destination / "human_metrics.jsonl")
    snapshots = destination / "snapshots" / metrics.run_id
    config = {**vars(args), "mode": "human_play", "loaded_checkpoint": str(source.resolve()),
              "device": str(device), "width": width, "blocks": blocks,
              "reanalyze_simulations": reanalysis_simulations,
              "tactical_checks": tactical_checks, "tactical_nodes": tactical_nodes,
              "game_passes": game_passes}
    save_snapshot(snapshots / "before_session.pt", network, iteration, width, blocks)
    if source != checkpoint or metadata["origin_snapshot"] is None:
        # A newly seeded experiment is anchored to the model it started from.
        metadata["origin_snapshot"] = os.path.relpath(snapshots / "before_session.pt", destination)
    metrics.write("human_session_start", config=config, training_iteration=iteration,
                  human_games=metadata["human_games"], replay=replay_summary(replay),
                  human_replay=replay_summary(human_replay))
    output(f"Loaded model: {source.resolve()}")
    output(f"Learning after completed games; model will be saved to {checkpoint.resolve()}")
    for game_number in range(1, args.games + 1):
        if args.color == "alternate":
            color = metadata["human_games"] % 2 == 0
        else:
            color = chess.WHITE if args.color == "white" else chess.BLACK
        output(f"\nGame {game_number}/{args.games}. You play {'White' if color else 'Black'}.")
        started = time.perf_counter()
        game = human_game(network, device, color, args.simulations, args.max_plies,
                          input_fn=input_fn, output=output, tactical_checks=tactical_checks,
                          tactical_nodes=tactical_nodes, on_position=on_position,
                          cancel_requested=cancel_requested)
        archive = destination / "games" / f"{metrics.run_id}_game_{game_number:04d}.pgn"
        write_pgn(game.moves, archive, result_override=game.result, termination=game.termination,
                  headers={"Event": "Human vs Chess AI", "White": "Human" if color else "Chess AI",
                           "Black": "Chess AI" if color else "Human"})
        metrics.write("human_game", game=game_number, result=game.result,
                      completed=game.completed, termination=game.termination,
                      human_color="white" if color else "black", plies=len(game.moves),
                      seconds=time.perf_counter() - started, pgn=str(archive.resolve()))
        output(f"Saved game: {archive.resolve()}")
        if not game.completed:
            output("Unfinished game: learning skipped.")
            break
        metadata["human_games"] += 1
        if game.examples:
            output(f"Reanalyzing {len(game.examples)} positions at {reanalysis_simulations} simulations...")
            reanalysis_started = time.perf_counter()
            refreshed = reanalyze_game(game, network, device, reanalysis_simulations,
                                       tactical_checks, tactical_nodes)
            changed = sum(old[1][int(np.argmax(old[2]))] != new[1][int(np.argmax(new[2]))]
                          for old, new in zip(game.examples, refreshed))
            revised = refresh_matching_policies(human_replay, refreshed)
            game.examples = refreshed
            metrics.write("human_reanalysis", human_games=metadata["human_games"],
                          seconds=time.perf_counter() - reanalysis_started,
                          positions=len(refreshed), simulations=reanalysis_simulations,
                          changed_best_moves=int(changed), revised_older_positions=revised,
                          tactical_checks=tactical_checks, tactical_nodes=tactical_nodes)
            human_replay.extend(game.examples)
            # By default, train each position of this game about once (--game-passes):
            # many repeated passes over a single game memorise its outcome.
            _, human_per_batch = batch_split(args.batch_size, len(replay), len(human_replay),
                                             args.human_fraction)
            game_steps = steps or max(1, math.ceil(game_passes * len(game.examples)
                                                   / human_per_batch))
            stats = {}
            fit_before = game_fit(network, game.examples, device)
            training_started = time.perf_counter()
            learn(network, optimizer, replay, device, game_steps, args.batch_size, sampler,
                  stats=stats, human_replay=human_replay, human_fraction=args.human_fraction,
                  recent_human=game.examples)
            fit_after = game_fit(network, game.examples, device)
            metrics.write("human_training", human_games=metadata["human_games"],
                          seconds=time.perf_counter() - training_started,
                          replay=replay_summary(replay), human_replay=replay_summary(human_replay),
                          game_fit_before=fit_before, game_fit_after=fit_after,
                          **stats)
            output(f"Learned from {stats['recent_human_samples']} samples of this game "
                   f"({stats['human_samples']} human / {stats['self_play_samples']} self-play samples total).")
            output(f"Covered {stats['recent_unique_positions']}/{len(game.examples)} game positions; "
                   f"policy target loss {fit_before['policy_loss']:.4f} -> {fit_after['policy_loss']:.4f}.")
        else:
            output("No moves were played; no positions to train on.")
        save_checkpoint(checkpoint, network, optimizer, iteration, replay, rng, sampler,
                        width, blocks, config=config, human_replay=human_replay,
                        human_games=metadata["human_games"],
                        origin_snapshot=metadata["origin_snapshot"])
        snapshot = snapshots / f"after_game_{metadata['human_games']:06d}.pt"
        save_snapshot(snapshot, network, iteration, width, blocks)
        metrics.write("human_checkpoint", human_games=metadata["human_games"],
                      checkpoint=str(checkpoint.resolve()), snapshot=str(snapshot.resolve()))
        output(f"Updated model saved: {checkpoint.resolve()}")
    metrics.write("human_session_end", human_games=metadata["human_games"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="open a click-to-move chess window")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt",
                        help="trusted source model for the first session (full checkpoint or snapshot)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="shared model directory used by self-play too (default: checkpoints)")
    parser.add_argument("--color", choices=("white", "black", "alternate"), default="white")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=600)
    parser.add_argument("--steps", type=int, default=None,
                        help="gradient steps after each completed game (default: from --game-passes)")
    parser.add_argument("--game-passes", type=float, default=1.0,
                        help="average times each position of a finished game is trained on")
    parser.add_argument("--reanalyze-simulations", type=int, default=512,
                        help="stronger post-game search used to refresh policy targets")
    parser.add_argument("--tactical-checks", type=int, choices=range(4), default=3,
                        help="bounded forcing-mate checking depth; 0 disables tactical proofs")
    parser.add_argument("--tactical-nodes", type=int, default=300,
                        help="maximum proof nodes per root move")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--human-fraction", type=float, default=0.25)
    parser.add_argument("--human-replay-size", type=int, default=5000)
    parser.add_argument("--replay-size", type=int, default=50000,
                        help="self-play replay kept in the shared checkpoint (match chess_ai.py)")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gui:
        from gui import run_window
        run_window(args)
    else:
        try:
            run(args)
        except IncompatibleCheckpointError as error:
            raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
