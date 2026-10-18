"""Small AlphaZero-style chess self-play trainer. No handcrafted chess evaluation."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import chess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from checkpoint_access import single_writer
from rules import game_outcome

PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
PROMOTIONS = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
ACTION_SIZE = 64 * 64 * len(PROMOTIONS)
INPUT_PLANES = 21
HALFMOVE_PLANE = 18
POLICY_DIM = 32
WEIGHT_DECAY = 0.01
# Format 2: side-to-move board orientation, uint8 planes, a batch-normalised body and a
# from/to policy head. Format 1 models and replay buffers cannot be converted.
FORMAT_VERSION = 2
# Gumbel MuZero's completed-Q transform (defaults of DeepMind's mctx library).
C_VISIT, C_SCALE = 50.0, 0.1
MAX_CONSIDERED = 16
# Material of the stronger side; the weaker side has a lone king.
ENDGAMES = {"KQvK": "Q", "KRvK": "R", "KRRvK": "RR", "KQRvK": "QR", "KPvK": "P"}


class IncompatibleCheckpointError(ValueError):
    pass


def check_format(data: dict, path) -> None:
    version = data.get("format_version", 1)
    if version != FORMAT_VERSION:
        raise IncompatibleCheckpointError(
            f"{path} was saved in model format {version}, but this code uses format "
            f"{FORMAT_VERSION} (side-to-move board encoding and a new network). It cannot "
            "be resumed or loaded: train a new model in a fresh --checkpoint-dir.")


def orient(square: chess.Square, turn: chess.Color) -> chess.Square:
    """Squares as the side to move sees them: Black's view is mirrored rank-wise."""
    return square if turn == chess.WHITE else chess.square_mirror(square)


def move_index(move: chess.Move, turn: chess.Color) -> int:
    """Encode from/to/promotion from the mover's view, matching ``encode``."""
    return ((orient(move.from_square, turn) * 64 + orient(move.to_square, turn)) * 5
            + PROMOTIONS[move.promotion])


def move_indices(board: chess.Board, moves: list[chess.Move]) -> np.ndarray:
    return np.array([move_index(move, board.turn) for move in moves], dtype=np.int64)


def encode(board: chess.Board) -> np.ndarray:
    """uint8 planes from the side to move's view; Black's positions are mirrored.

    0-5 own pieces, 6-11 opponent pieces, 12 all ones (marks the board edge for padded
    convolutions), 13-16 own then opponent castling rights, 17 en-passant square,
    18 halfmove clock (0-100, scaled inside the network), 19-20 repetition flags.
    There is no colour plane, so mirror-image positions are identical inputs and the
    value head cannot learn "White wins" as a shortcut.
    """
    us = board.turn
    masks = [board.pieces_mask(piece_type, color) for color in (us, not us)
             for piece_type in PIECES]
    if us == chess.BLACK:
        masks = [chess.flip_vertical(mask) for mask in masks]
    x = np.zeros((INPUT_PLANES, 8, 8), dtype=np.uint8)
    x[:12] = np.unpackbits(np.array(masks, dtype="<u8").view(np.uint8),
                           bitorder="little").reshape(12, 8, 8)
    x[12] = 1
    for i, color in enumerate((us, not us)):
        x[13 + 2 * i] = board.has_kingside_castling_rights(color)
        x[14 + 2 * i] = board.has_queenside_castling_rights(color)
    if board.ep_square is not None:
        square = orient(board.ep_square, us)
        x[17, square // 8, square % 8] = 1
    x[HALFMOVE_PLANE] = min(board.halfmove_clock, 100)
    x[19] = board.is_repetition(2)
    x[20] = board.is_repetition(3)
    return x


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        nn.init.zeros_(self.bn2.weight)  # each block starts as the identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


class PolicyValueNet(nn.Module):
    def __init__(self, width: int = 64, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(INPUT_PLANES, width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU())
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        # Move logits score a (from, to) square pair with a dot product of per-square
        # embeddings, plus a from-square promotion term: no 20,480-wide dense layer.
        self.policy_from = nn.Conv2d(width, POLICY_DIM, 1)
        self.policy_to = nn.Conv2d(width, POLICY_DIM, 1)
        self.policy_promotion = nn.Conv2d(width, len(PROMOTIONS), 1)
        self.value_conv = nn.Conv2d(width, 4, 1)
        self.value_fc1 = nn.Linear(4 * 64, 128)
        self.value_fc2 = nn.Linear(128, 1)
        scale = torch.ones(INPUT_PLANES, 1, 1)
        scale[HALFMOVE_PLANE] = 0.01
        self.register_buffer("plane_scale", scale, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(self.stem(x.float() * self.plane_scale))
        source = self.policy_from(features).flatten(2)
        target = self.policy_to(features).flatten(2)
        pairs = torch.einsum("bdf,bdt->bft", source, target) * POLICY_DIM ** -0.5
        promotion = self.policy_promotion(features).flatten(2).transpose(1, 2)
        policy = (pairs.unsqueeze(-1) + promotion.unsqueeze(2)).flatten(1)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(
            F.relu(self.value_conv(features)).flatten(1))))).squeeze(-1)
        return policy, value


def make_optimizer(network: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)


def terminal_value(board: chess.Board) -> float | None:
    outcome = game_outcome(board)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _leaf(board: chess.Board) -> tuple[float | None, list[chess.Move]]:
    """``terminal_value`` and the legal moves, generating moves only once."""
    moves = list(board.legal_moves)
    if not moves:
        return (-1.0 if board.is_check() else 0.0), moves
    if (board.halfmove_clock >= 100 or board.is_insufficient_material()
            or board.is_repetition(3)):
        return 0.0, moves
    return None, moves


class SearchResult(NamedTuple):
    moves: list[chess.Move]  # every legal root move
    policy: np.ndarray  # improved-policy training target over ``moves``
    move: chess.Move  # the move sequential halving selected


class Node:
    """Expanded position; child statistics are from this node's side-to-move view."""
    __slots__ = ("moves", "logits", "priors", "raw_value", "visits", "value_sums", "children")

    def __init__(self, moves: list[chess.Move], logits: np.ndarray, raw_value: float):
        self.moves = moves
        self.logits = np.array(logits, dtype=np.float64)
        self.logits -= self.logits.max()
        self.priors = np.exp(self.logits)
        self.priors /= self.priors.sum()
        self.raw_value = raw_value
        self.visits = np.zeros(len(moves))
        self.value_sums = np.zeros(len(moves))
        # None: unexplored; float: terminal value for the child's side to move.
        self.children: list[Node | float | None] = [None] * len(moves)


def _sigma(node: Node) -> np.ndarray:
    """Monotone transform of completed Q-values (Danihelka et al., 2022).

    Unvisited moves take the node's mixed value estimate instead of an arbitrary Q.
    """
    visited = node.visits > 0
    total = node.visits.sum()
    q = np.zeros(len(node.moves))
    if total:
        q[visited] = node.value_sums[visited] / node.visits[visited]
        weights = np.maximum(node.priors[visited], 1e-30)
        mixed = (node.raw_value + total * (weights @ q[visited]) / weights.sum()) / (total + 1)
    else:
        mixed = node.raw_value
    completed = np.where(visited, q, mixed)
    low, high = completed.min(), completed.max()
    completed = (completed - low) / max(high - low, 1e-8)
    return (C_VISIT + node.visits.max()) * C_SCALE * completed


def _softmax(scores: np.ndarray) -> np.ndarray:
    weights = np.exp(scores - scores.max())
    return weights / weights.sum()


def _interior_action(node: Node) -> int:
    improved = _softmax(node.logits + _sigma(node))
    return int(np.argmax(improved - node.visits / (1 + node.visits.sum())))


def considered_visits(considered: int, simulations: int) -> list[int]:
    """Sequential-halving schedule: the root visit count each simulation tops up."""
    if considered <= 1:
        return list(range(simulations))
    rounds = math.ceil(math.log2(considered))
    sequence, visits = [], 0
    while len(sequence) < simulations:
        for _ in range(max(1, simulations // (rounds * considered))):
            sequence.extend([visits] * considered)
            visits += 1
        considered = max(2, considered // 2)
    return sequence[:simulations]


def _rounds(schedule: list[int]) -> list[tuple[int, int]]:
    """Group the schedule into rounds: (visit count to top up, number of root moves)."""
    rounds = []
    for target in schedule:
        if rounds and rounds[-1][0] == target:
            rounds[-1][1] += 1
        else:
            rounds.append([target, 1])
    return [(target, count) for target, count in rounds]


def _descend(board: chess.Board, root: Node, action: int):
    """Follow the tree from a root move to a leaf, restoring ``board`` afterwards.

    Returns the path and either the leaf's terminal value or the leaf to expand.
    """
    path, node = [(root, action)], root
    board.push(root.moves[action])
    depth = 1
    try:
        while True:
            child = node.children[action]
            if child is None:
                value, legal = _leaf(board)
                if value is None:
                    return path, None, (legal, (encode(board), move_indices(board, legal)))
                node.children[action] = value
                return path, value, None
            if not isinstance(child, Node):
                return path, child, None
            node = child
            action = _interior_action(node)
            path.append((node, action))
            board.push(node.moves[action])
            depth += 1
    finally:
        for _ in range(depth):
            board.pop()


def _backup(path: list[tuple[Node, int]], value: float) -> None:
    for parent, step in reversed(path):
        value = -value
        parent.value_sums[step] += value
        parent.visits[step] += 1


def _round(board: chess.Board, root: Node, actions: list[int], network):
    """Simulate distinct root moves together, evaluating their new leaves in one batch.

    Their subtrees are disjoint, so expanding the leaves after all descents gives the
    same tree as simulating them one after another.
    """
    leaves, requests = [], []
    for action in actions:
        path, value, leaf = _descend(board, root, action)
        if leaf is None:
            _backup(path, value)
        else:
            leaves.append((path, leaf[0]))
            requests.append(leaf[1])
    if requests:
        replies = yield network, requests
        for (path, legal), (logits, value) in zip(leaves, replies):
            parent, step = path[-1]
            parent.children[step] = Node(legal, logits, value)
            _backup(path, value)


def search_steps(board: chess.Board, network, simulations: int, *, noise: bool = False,
                 rng: np.random.Generator | None = None,
                 root_moves: set[chess.Move] | None = None):
    """Gumbel AlphaZero search, written as a generator so callers can batch leaves.

    Yields ``(network, [(planes, legal move indices), ...])`` and expects a list of
    ``(legal logits, value)`` back; returns a ``SearchResult``. The root samples moves
    with Gumbel noise (``noise=True``) or deterministically, then spends simulations by
    sequential halving; each halving round is one batch. The policy target uses every
    legal move's completed Q-value, so it is informative even with few simulations.
    ``board`` is restored before returning.
    """
    if simulations < 1:
        raise ValueError("simulations must be >= 1")
    value, legal = _leaf(board)
    if value is not None:
        raise ValueError("cannot search a terminal position")
    (logits, root_value), = yield network, [(encode(board), move_indices(board, legal))]
    keep = list(range(len(legal)))
    if root_moves is not None:
        keep = [i for i, move in enumerate(legal) if move in root_moves]
        if not keep:
            raise ValueError("root move restriction must include at least one legal move")
    root = Node([legal[i] for i in keep], np.asarray(logits)[keep], root_value)
    if noise:
        gumbel = (rng or np.random.default_rng()).gumbel(size=len(keep))
    else:
        gumbel = np.zeros(len(keep))
    for target, count in _rounds(considered_visits(min(MAX_CONSIDERED, len(keep)), simulations)):
        scores = np.where(root.visits == target, gumbel + root.logits + _sigma(root), -np.inf)
        actions = np.argsort(-scores, kind="stable")[:count]
        yield from _round(board, root, actions.tolist(), network)
    sigma = _sigma(root)
    scores = np.where(root.visits == root.visits.max(), gumbel + root.logits + sigma, -np.inf)
    policy = np.zeros(len(legal))
    policy[keep] = _softmax(root.logits + sigma)
    return SearchResult(legal, policy, root.moves[int(np.argmax(scores))])


@torch.no_grad()
def _evaluate(network, device: torch.device, requests: list) -> list[tuple[np.ndarray, float]]:
    states = torch.from_numpy(np.stack([planes for planes, _ in requests])).to(device)
    logits, values = network(states)
    logits, values = logits.float().cpu().numpy(), values.float().cpu().numpy()
    return [(logits[row, indices], float(values[row]))
            for row, (_, indices) in enumerate(requests)]


def run_batched(tasks: list, device: torch.device) -> list:
    """Advance search generators in lockstep, evaluating their leaves in batches.

    Requests for the same network are evaluated together, so several games cost
    little more per step than one. Returns each task's return value, in order.
    """
    results, pending, prepared = [None] * len(tasks), {}, set()

    def advance(index, replies=None):
        try:
            pending[index] = tasks[index].send(replies)
        except StopIteration as stop:
            results[index] = stop.value

    for index in range(len(tasks)):
        advance(index)
    while pending:
        groups = {}
        for index, (network, requests) in pending.items():
            groups.setdefault(id(network), (network, []))[1].append((index, requests))
        pending = {}
        for key, (network, items) in groups.items():
            if key not in prepared:
                network.eval()
                prepared.add(key)
            replies = _evaluate(network, device, [request for _, requests in items
                                                  for request in requests])
            offset = 0
            for index, requests in items:
                advance(index, replies[offset:offset + len(requests)])
                offset += len(requests)
    return results


def search(board: chess.Board, network, device: torch.device, simulations: int,
           noise: bool = False, rng: np.random.Generator | None = None,
           root_moves: set[chess.Move] | None = None) -> SearchResult:
    return run_batched([search_steps(board, network, simulations, noise=noise, rng=rng,
                                     root_moves=root_moves)], device)[0]


def endgame_start(rng: np.random.Generator) -> tuple[chess.Board, str]:
    """A random legal, unfinished position from a simple endgame, either colour stronger.

    Mates arrive quickly here, giving the value head decisive outcomes to learn from.
    """
    name = list(ENDGAMES)[int(rng.integers(len(ENDGAMES)))]
    strong = bool(rng.integers(2))
    pieces = [chess.Piece(chess.KING, strong), chess.Piece(chess.KING, not strong)]
    pieces += [chess.Piece(chess.Piece.from_symbol(symbol).piece_type, strong)
               for symbol in ENDGAMES[name]]
    while True:
        board = chess.Board(None)
        for piece, square in zip(pieces, rng.choice(64, size=len(pieces), replace=False)):
            board.set_piece_at(int(square), piece)
        board.turn = bool(rng.integers(2))
        if board.is_valid() and terminal_value(board) is None:
            return board, name


def is_endgame_game(number: int, fraction: float) -> bool:
    """Spread a fraction of games evenly over the run's global game numbers."""
    return math.floor((number + 1) * fraction) > math.floor(number * fraction)


def self_play_steps(network, simulations: int, max_plies: int, rng: np.random.Generator,
                    *, board: chess.Board | None = None,
                    recorded_moves: list[chess.Move] | None = None,
                    game_info: dict | None = None):
    """One self-play game as a generator of search requests; see ``run_batched``."""
    board = chess.Board() if board is None else board.copy()
    positions = []
    while len(positions) < max_plies and terminal_value(board) is None:
        result = yield from search_steps(board, network, simulations, noise=True, rng=rng)
        positions.append((encode(board), move_indices(board, result.moves),
                          result.policy.astype(np.float32), board.turn))
        board.push(result.move)
        if recorded_moves is not None:
            recorded_moves.append(result.move)

    outcome = game_outcome(board)
    # An unfinished game has no observed value target. Keep its search-policy targets.
    winner = outcome.winner if outcome is not None else None
    examples = [(x, idx, probs, None if outcome is None else float(
        0 if winner is None else (1 if color == winner else -1)))
        for x, idx, probs, color in positions]
    result = outcome.result() if outcome is not None else "* (move cap)"
    if game_info is not None:
        game_info.update(result=result, truncated=outcome is None, plies=len(positions),
                         termination=outcome.termination.name if outcome else "MOVE_CAP",
                         finished=time.perf_counter())
    return examples, result


def self_play(network, device: torch.device, simulations: int, max_plies: int,
              rng: np.random.Generator, *, board: chess.Board | None = None,
              recorded_moves: list[chess.Move] | None = None,
              game_info: dict | None = None) -> tuple[list[tuple], str]:
    return run_batched([self_play_steps(network, simulations, max_plies, rng, board=board,
                                        recorded_moves=recorded_moves,
                                        game_info=game_info)], device)[0]


def batch_tensors(batch: list[tuple], device: torch.device):
    """Stack examples, padding each position's legal-move list to the longest one."""
    states = torch.from_numpy(np.stack([ex[0] for ex in batch])).to(device)
    width = max(len(ex[1]) for ex in batch)
    indices = np.zeros((len(batch), width), dtype=np.int64)
    targets = np.zeros((len(batch), width), dtype=np.float32)
    valid = np.zeros((len(batch), width), dtype=bool)
    for row, (_, idx, pi, _) in enumerate(batch):
        indices[row, :len(idx)], targets[row, :len(idx)], valid[row, :len(idx)] = idx, pi, True
    return (states, torch.from_numpy(indices).to(device), torch.from_numpy(targets).to(device),
            torch.from_numpy(valid).to(device))


def policy_cross_entropy(logits: torch.Tensor, indices: torch.Tensor,
                         targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-example cross-entropy against search targets, normalised over legal moves."""
    legal = logits.gather(1, indices).masked_fill(~valid, float("-inf"))
    log_probs = F.log_softmax(legal, dim=1).masked_fill(~valid, 0.0)
    return -(targets * log_probs).sum(1)


def batch_split(batch_size: int, self_available: int, human_available: int,
                human_fraction: float) -> tuple[int, int]:
    """(self-play, human) samples per update, keeping the human quota when possible."""
    size = min(batch_size, self_available + human_available)
    human = min(human_available, max(1, int(size * human_fraction)))
    self_count = min(size - human, self_available)
    return self_count, size - self_count


def learn(network: PolicyValueNet, optimizer: torch.optim.Optimizer, replay: deque,
          device: torch.device, steps: int, batch_size: int, sampler: random.Random,
          stats: dict | None = None, human_replay: deque | None = None,
          human_fraction: float = 0.25,
          recent_human: list[tuple] | None = None) -> tuple[float, float]:
    if not 0 <= human_fraction <= 1:
        raise ValueError("human-fraction must be between 0 and 1")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    examples = list(replay)
    human_examples = list(human_replay or []) if human_fraction > 0 else []
    if not examples and not human_examples:
        raise ValueError("cannot train with empty replay")
    network.train()
    policy_losses, value_losses = [], []
    policy_samples, value_samples, decisive_samples = 0, 0, 0
    human_samples, recent_samples = 0, 0
    recent_ids = {id(ex) for ex in recent_human or []}
    recent = [ex for ex in human_examples if id(ex) in recent_ids]
    recent_order = recent.copy()
    if recent_order:
        sampler.shuffle(recent_order)
    cursor, recent_seen = 0, set()
    for _ in range(steps):
        self_count, human_count = batch_split(batch_size, len(examples), len(human_examples),
                                              human_fraction)
        if recent and human_count:
            # Cover the current game before repeating positions, while keeping the
            # human/self-play quota. Every update still includes a recent position.
            forced = recent_order[cursor:cursor + human_count]
            cursor += len(forced)
            if not forced:
                forced = [sampler.choice(recent)]
            remaining = human_examples.copy()
            for selected in forced:
                remaining.pop(next(i for i, ex in enumerate(remaining) if ex is selected))
            human_batch = forced + sampler.sample(remaining, k=human_count - len(forced))
        else:
            human_batch = sampler.sample(human_examples, k=human_count)
        batch = sampler.sample(examples, k=self_count) + human_batch
        human_samples += human_count
        recent_samples += sum(id(ex) in recent_ids for ex in human_batch)
        recent_seen.update(id(ex) for ex in human_batch if id(ex) in recent_ids)
        states, indices, targets, valid = batch_tensors(batch, device)
        logits, values = network(states)
        policy_loss = policy_cross_entropy(logits, indices, targets, valid).mean()
        labeled = [i for i, ex in enumerate(batch) if ex[3] is not None]
        # A disconnected zero leaves value-head gradients at None when no labels exist.
        # This also prevents weight decay from updating an unsupervised value head.
        value_loss = logits.new_zeros(())
        if labeled:
            value_targets = torch.tensor([batch[i][3] for i in labeled],
                                         dtype=torch.float32, device=device)
            value_loss = F.mse_loss(values[labeled], value_targets)
            value_losses.append(value_loss.item())
        policy_samples += len(batch)
        value_samples += len(labeled)
        decisive_samples += sum(batch[i][3] != 0 for i in labeled)
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()
        policy_losses.append(policy_loss.item())
    p_loss = float(np.mean(policy_losses))
    v_loss = float(np.mean(value_losses)) if value_losses else 0.0
    if stats is not None:
        stats.update(policy_loss=p_loss, value_loss=v_loss if value_losses else None,
                     policy_samples=policy_samples, value_samples=value_samples,
                     decisive_value_samples=decisive_samples,
                     draw_value_samples=value_samples - decisive_samples,
                     human_samples=human_samples, self_play_samples=policy_samples - human_samples,
                     recent_human_samples=recent_samples,
                     recent_unique_positions=len(recent_seen),
                     recent_game_positions=len(recent_human or []),
                     value_label_fraction=value_samples / policy_samples,
                     learning_rate=optimizer.param_groups[0]["lr"])
    return p_loss, v_loss


def save_checkpoint(path: Path, network: PolicyValueNet, optimizer: torch.optim.Optimizer,
                    iteration: int, replay: deque, rng: np.random.Generator,
                    sampler: random.Random, width: int, blocks: int,
                    config: dict | None = None, human_replay: deque | None = None,
                    human_games: int = 0, origin_snapshot: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(),
                "iteration": iteration, "replay": list(replay), "rng": rng.bit_generator.state,
                "sampler": sampler.getstate(), "torch_rng": torch.get_rng_state(),
                "width": width, "blocks": blocks, "format_version": FORMAT_VERSION,
                "config": config, "human_replay": list(human_replay or []),
                "human_games": human_games, "origin_snapshot": origin_snapshot}, temp)
    os.replace(temp, path)


def save_snapshot(path: Path, network: PolicyValueNet, iteration: int,
                  width: int, blocks: int) -> None:
    """Keep a lightweight, immutable reference without copying the replay buffer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({"network": network.state_dict(), "iteration": iteration,
                "width": width, "blocks": blocks, "format_version": FORMAT_VERSION}, temp)
    os.replace(temp, path)


def load_checkpoint(path: Path, network: PolicyValueNet, optimizer: torch.optim.Optimizer,
                    device: torch.device, replay_size: int,
                    rng: np.random.Generator, sampler: random.Random,
                    human_replay: deque | None = None,
                    metadata: dict | None = None) -> tuple[int, deque]:
    # Only load checkpoints you trust: PyTorch checkpoint files can execute code on load.
    data = torch.load(path, map_location=device, weights_only=False)
    check_format(data, path)
    network.load_state_dict(data["network"])
    optimizer.load_state_dict(data["optimizer"])
    rng.bit_generator.state = data["rng"]
    sampler.setstate(data["sampler"])
    torch.set_rng_state(data["torch_rng"].cpu())
    if human_replay is not None:
        human_replay.clear()
        human_replay.extend(data.get("human_replay", []))
    if metadata is not None:
        metadata.update(human_games=data.get("human_games", 0),
                        origin_snapshot=data.get("origin_snapshot"))
    return data["iteration"], deque(data["replay"], maxlen=replay_size)


@single_writer
def run(args: argparse.Namespace) -> None:
    from training_metrics import MetricsWriter, replay_summary

    if args.simulations < 1 or args.max_plies < 1 or args.batch_size < 1 or args.steps < 1:
        raise ValueError("simulations, max-plies, batch-size and steps must be positive")
    if args.games < 1 or args.iterations < 1 or args.replay_size < 1:
        raise ValueError("games, iterations and replay-size must be positive")
    eval_every = getattr(args, "eval_every", 10)
    eval_games = getattr(args, "eval_games", 12)
    eval_simulations = getattr(args, "eval_simulations", 16)
    eval_max_plies = getattr(args, "eval_max_plies", 400)
    eval_seed = getattr(args, "eval_seed", 2026)
    eval_checkpoints = [Path(p) for p in getattr(args, "eval_checkpoint", [])]
    human_fraction = getattr(args, "human_fraction", 0.25)
    human_replay_size = getattr(args, "human_replay_size", 5000)
    endgame_fraction = getattr(args, "endgame_fraction", 0.25)
    if not 0 <= human_fraction <= 1 or human_replay_size < 1:
        raise ValueError("human-fraction must be between 0 and 1; human-replay-size must be positive")
    if not 0 <= endgame_fraction <= 1:
        raise ValueError("endgame-fraction must be between 0 and 1")
    if eval_every < 0 or eval_games < 2 or eval_games % 2:
        raise ValueError("eval-every must be >= 0 and eval-games must be even and >= 2")
    if eval_simulations < 1 or eval_max_plies < 1:
        raise ValueError("eval-simulations and eval-max-plies must be positive")
    for reference in eval_checkpoints:
        if not reference.is_file():
            raise FileNotFoundError(f"No evaluation checkpoint at {reference}")
    if args.width < 1 or args.blocks < 0 or args.threads < 1:
        raise ValueError("width and threads must be positive; blocks must be >= 0")
    if args.lr is not None and (not np.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("lr must be finite and positive")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sampler = random.Random(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    checkpoint_dir = Path(args.checkpoint_dir)
    path = checkpoint_dir / "latest.pt"
    if path.exists() and not args.resume:
        raise FileExistsError(f"Checkpoint already exists at {path}; use --resume or a new --checkpoint-dir")
    width, blocks = args.width, args.blocks
    if args.resume:
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint at {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        check_format(saved, path)
        width, blocks = saved["width"], saved["blocks"]
        del saved
    network = PolicyValueNet(width, blocks).to(device)
    optimizer = make_optimizer(network, args.lr or 0.001)
    replay = deque(maxlen=args.replay_size)
    human_replay = deque(maxlen=human_replay_size)
    metadata = {"human_games": 0, "origin_snapshot": None}
    start = 0
    if args.resume:
        start, replay = load_checkpoint(path, network, optimizer, device,
                                        args.replay_size, rng, sampler,
                                        human_replay=human_replay, metadata=metadata)
        if args.lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        print(f"Resumed after iteration {start} ({len(replay)} stored positions)", flush=True)
    config = {**vars(args), "width": width, "blocks": blocks, "device": str(device),
              "lr": optimizer.param_groups[0]["lr"], "eval_every": eval_every,
              "eval_games": eval_games, "eval_simulations": eval_simulations,
              "eval_max_plies": eval_max_plies, "eval_seed": eval_seed,
              "human_fraction": human_fraction, "human_replay_size": human_replay_size,
              "endgame_fraction": endgame_fraction, "format_version": FORMAT_VERSION}
    metrics_path = Path(getattr(args, "metrics_file", None) or checkpoint_dir / "metrics.jsonl")
    metrics = MetricsWriter(metrics_path)
    snapshot_dir = checkpoint_dir / "snapshots" / metrics.run_id
    initial_snapshot = snapshot_dir / f"iteration_{start:06d}.pt"
    save_snapshot(initial_snapshot, network, start, width, blocks)
    # The experiment's first model is a fixed evaluation anchor across resumed sessions.
    if metadata["origin_snapshot"] is None:
        metadata["origin_snapshot"] = os.path.relpath(initial_snapshot, checkpoint_dir)
    origin = checkpoint_dir / metadata["origin_snapshot"]
    previous_snapshot = initial_snapshot
    metrics.write("run_start", start_iteration=start, config=config,
                  torch_version=torch.__version__, replay=replay_summary(replay),
                  human_replay=replay_summary(human_replay), human_games=metadata["human_games"],
                  initial_snapshot=str(initial_snapshot.resolve()),
                  origin_snapshot=str(origin.resolve()) if origin.is_file() else None)
    print(f"Training on {device}, width={width}, blocks={blocks}", flush=True)
    print(f"Metrics: {metrics_path.resolve()}", flush=True)
    recorded_once = False
    for iteration in range(start + 1, start + args.iterations + 1):
        iteration_started = time.perf_counter()
        selfplay_started = iteration_started
        tasks, game_infos, record, record_game = [], [], None, None
        for game in range(args.games):
            board, start_name = None, "standard"
            if is_endgame_game((iteration - 1) * args.games + game, endgame_fraction):
                board, start_name = endgame_start(rng)
            moves = None
            # Record the command's first game from the standard starting position.
            if ((args.record or args.record_video) and not recorded_once
                    and record_game is None and board is None):
                record = moves = []
                record_game = game
            game_info = {"start": start_name}
            if board is not None:
                game_info["start_fen"] = board.fen()
            game_infos.append(game_info)
            tasks.append(self_play_steps(network, args.simulations, args.max_plies, rng,
                                         board=board, recorded_moves=moves,
                                         game_info=game_info))
        games_played = run_batched(tasks, device)
        selfplay_seconds = time.perf_counter() - selfplay_started
        results = {}
        for game, ((samples, result), game_info) in enumerate(zip(games_played, game_infos)):
            game_seconds = game_info.pop("finished") - selfplay_started
            metrics.write("self_play_game", iteration=iteration, game=game + 1,
                          **game_info, seconds=game_seconds,
                          positions_per_second=len(samples) / max(game_seconds, 1e-9))
            if game == record_game:
                from recording import lichess_url, write_pgn
                output_dir = Path(args.record_dir)
                stem = f"selfplay_iteration_{iteration}_game_{game + 1}"
                pgn_path = output_dir / (stem + ".pgn")
                write_pgn(record, pgn_path)
                print(f"Recorded first game ({len(record)} plies). PGN: {pgn_path.resolve()}", flush=True)
                if record:
                    url = lichess_url(record)
                    if len(url) <= 2000:
                        print(f"Watch/replay on Lichess: {url}", flush=True)
                    else:
                        print("Game is too long for a reliable replay URL. Import the "
                              f"PGN above at https://www.chess.com/analysis", flush=True)
                if args.record_video:
                    from recording import write_mp4
                    video_path = output_dir / (stem + ".mp4")
                    write_mp4(record, video_path, fps=args.video_fps)
                    print(f"Video saved: {video_path.resolve()}", flush=True)
                recorded_once = True
            replay.extend(samples)
            results[result] = results.get(result, 0) + 1
            start_text = "" if game_info["start"] == "standard" else f" from {game_info['start']}"
            print(f"Iteration {iteration}, game {game + 1}/{args.games}{start_text}: {result}, "
                  f"{len(samples)} plies", flush=True)
        train_started = time.perf_counter()
        train_stats = {}
        p_loss, _ = learn(network, optimizer, replay, device,
                          args.steps, args.batch_size, sampler, stats=train_stats,
                          human_replay=human_replay, human_fraction=human_fraction)
        train_seconds = time.perf_counter() - train_started
        save_started = time.perf_counter()
        save_checkpoint(path, network, optimizer, iteration, replay, rng, sampler,
                        width, blocks, config=config, human_replay=human_replay,
                        human_games=metadata["human_games"],
                        origin_snapshot=metadata["origin_snapshot"])
        snapshot = snapshot_dir / f"iteration_{iteration:06d}.pt"
        save_snapshot(snapshot, network, iteration, width, blocks)
        save_seconds = time.perf_counter() - save_started
        total_plies = sum(info["plies"] for info in game_infos)
        capped = sum(info["truncated"] for info in game_infos)
        metrics.write("training_iteration", iteration=iteration, games=results,
                      capped_games=capped, capped_game_rate=capped / args.games,
                      completed_games=args.games - capped, generated_positions=total_plies,
                      endgame_games=sum(info["start"] != "standard" for info in game_infos),
                      average_game_plies=total_plies / args.games,
                      positions_per_second=total_plies / max(selfplay_seconds, 1e-9),
                      self_play_seconds=selfplay_seconds, training_seconds=train_seconds,
                      checkpoint_seconds=save_seconds, replay=replay_summary(replay),
                      human_replay=replay_summary(human_replay),
                      checkpoint=str(path.resolve()), snapshot=str(snapshot.resolve()),
                      **train_stats)
        value_text = f"{train_stats['value_loss']:.4f}" if train_stats["value_loss"] is not None else "n/a"
        print(f"Iteration {iteration}: games={results}, replay={len(replay)}, "
              f"policy_loss={p_loss:.4f}, value_loss={value_text}, "
              f"capped={capped}/{args.games}, {total_plies / max(selfplay_seconds, 1e-9):.1f} "
              f"positions/s; saved {path}", flush=True)
        if eval_every and iteration % eval_every == 0:
            from evaluation import evaluate, load_opponent

            references = {}
            if origin.is_file():
                references[origin.resolve()] = "origin"
            references.setdefault(initial_snapshot.resolve(), "session_start")
            references.setdefault(previous_snapshot.resolve(), "previous")
            for reference in eval_checkpoints:
                references.setdefault(reference.resolve(), f"checkpoint:{reference.name}")
            for reference, label in [(None, "random"), *references.items()]:
                eval_started = time.perf_counter()
                opponent = load_opponent(reference, device) if reference else None
                scores = evaluate(network, opponent, device, eval_games, eval_simulations,
                                  eval_max_plies, eval_seed)
                metrics.write("evaluation", iteration=iteration, opponent=label,
                              opponent_checkpoint=str(reference) if reference else None,
                              candidate_checkpoint=str(snapshot.resolve()),
                              seconds=time.perf_counter() - eval_started, **scores)
                score = scores["completed_score"]
                score_text = f"{score:.1%}" if score is not None else "n/a"
                print(f"Evaluation vs {label}: W/D/L={scores['wins']}/{scores['draws']}/"
                      f"{scores['losses']}, unfinished={scores['truncated']}/{eval_games}, "
                      f"completed_score={score_text}", flush=True)
                del opponent
        previous_snapshot = snapshot
        metrics.write("iteration_end", iteration=iteration,
                      seconds=time.perf_counter() - iteration_started)
    metrics.write("run_end", final_iteration=start + args.iterations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play chess NN + Gumbel MCTS trainer")
    parser.add_argument("--iterations", type=int, default=10, help="additional iterations to run")
    parser.add_argument("--games", type=int, default=8,
                        help="self-play games per iteration (played concurrently, batched)")
    parser.add_argument("--simulations", type=int, default=16, help="MCTS simulations per move")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--endgame-fraction", type=float, default=0.25,
                        help="share of self-play games started from random simple endgames")
    parser.add_argument("--steps", type=int, default=32, help="gradient updates per iteration")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--human-replay-size", type=int, default=5000)
    parser.add_argument("--human-fraction", type=float, default=0.25,
                        help="target fraction of each batch from human games when available (0 disables)")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None,
                        help="learning rate (new run: 0.001; resume: keep saved unless specified)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--metrics-file", default=None,
                        help="append-only JSONL metrics (default: checkpoint-dir/metrics.jsonl)")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="evaluate every N iterations; 0 disables evaluation")
    parser.add_argument("--eval-games", type=int, default=12,
                        help="even number of games per opponent, colors swapped")
    parser.add_argument("--eval-simulations", type=int, default=16)
    parser.add_argument("--eval-max-plies", type=int, default=400,
                        help="evaluation move cap after each fixed opening")
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--eval-checkpoint", action="append", default=[],
                        help="trusted older .pt file to evaluate too; may be repeated")
    parser.add_argument("--record", action="store_true",
                        help="save first standard-start self-play game as PGN and print a Lichess link")
    parser.add_argument("--record-video", action="store_true",
                        help="also save first game as MP4 (requires optional video dependencies)")
    parser.add_argument("--record-dir", "--video-dir", dest="record_dir", default="recordings",
                        help="directory for recorded PGN/MP4 files")
    parser.add_argument("--video-fps", type=float, default=2.0,
                        help="video frames per second (one frame per half-move)")
    try:
        run(parser.parse_args())
    except IncompatibleCheckpointError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
