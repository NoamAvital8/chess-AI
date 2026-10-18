"""Small AlphaZero-style chess self-play trainer. No handcrafted chess evaluation."""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
PROMOTIONS = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
ACTION_SIZE = 64 * 64 * 5
INPUT_PLANES = 21


def move_index(move: chess.Move) -> int:
    """Encode from/to/promotion without confusing underpromotions with ordinary moves."""
    return (move.from_square * 64 + move.to_square) * 5 + PROMOTIONS[move.promotion]


def encode(board: chess.Board) -> np.ndarray:
    """Planes include pieces, turn, castling, en-passant, draw clocks and repetition.

    Repetition flags summarize history; exact fivefold/threefold adjudication still uses
    python-chess's full move history. We never change the rules using these features.
    """
    x = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    for color in (chess.WHITE, chess.BLACK):
        for p, piece_type in enumerate(PIECES):
            for square in board.pieces(piece_type, color):
                x[(0 if color else 6) + p, square // 8, square % 8] = 1.0
    x[12] = float(board.turn == chess.WHITE)
    for i, (color, side) in enumerate(((chess.WHITE, True), (chess.WHITE, False),
                                       (chess.BLACK, True), (chess.BLACK, False))):
        x[13 + i] = float(board.has_kingside_castling_rights(color) if side
                          else board.has_queenside_castling_rights(color))
    if board.ep_square is not None:
        x[17, board.ep_square // 8, board.ep_square % 8] = 1.0
    x[18] = min(board.halfmove_clock, 100) / 100.0
    x[19] = float(board.is_repetition(2))
    x[20] = float(board.is_repetition(3))
    return x


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class PolicyValueNet(nn.Module):
    def __init__(self, width: int = 32, blocks: int = 2):
        super().__init__()
        self.stem = nn.Conv2d(INPUT_PLANES, width, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.policy_conv = nn.Conv2d(width, 2, 1)
        self.policy_fc = nn.Linear(2 * 8 * 8, ACTION_SIZE)
        self.value_conv = nn.Conv2d(width, 1, 1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(F.relu(self.stem(x)))
        policy = self.policy_fc(F.relu(self.policy_conv(features)).flatten(1))
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(
            F.relu(self.value_conv(features)).flatten(1))))).squeeze(-1)
        return policy, value


@dataclass
class Node:
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0  # perspective of the player to move at this node
    children: dict[chess.Move, Node] = field(default_factory=dict)

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def terminal_value(board: chess.Board) -> float | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


@torch.no_grad()
def expand(node: Node, board: chess.Board, network: PolicyValueNet,
           device: torch.device) -> float:
    logits, value = network(torch.from_numpy(encode(board)).unsqueeze(0).to(device))
    legal = list(board.legal_moves)
    indices = torch.tensor([move_index(m) for m in legal], dtype=torch.long, device=device)
    priors = torch.softmax(logits[0, indices], dim=0).cpu().numpy()
    node.children = {move: Node(prior=float(p)) for move, p in zip(legal, priors)}
    return float(value.item())


def search(board: chess.Board, network: PolicyValueNet, device: torch.device,
           simulations: int, c_puct: float = 1.5, noise: bool = False,
           rng: np.random.Generator | None = None) -> tuple[list[chess.Move], np.ndarray]:
    """PUCT MCTS; returns legal moves and *normalized visit-count* training target."""
    if simulations < 1:
        raise ValueError("simulations must be >= 1")
    if terminal_value(board) is not None:
        raise ValueError("cannot search a terminal position")
    network.eval()
    root = Node()
    expand(root, board, network, device)
    if noise:
        rng = rng or np.random.default_rng()
        children = list(root.children.values())
        eta = rng.dirichlet(np.full(len(children), 0.3))
        for child, n in zip(children, eta):
            child.prior = 0.75 * child.prior + 0.25 * float(n)

    for _ in range(simulations):
        state = board.copy(stack=True)
        node = root
        path = [root]
        while node.children:
            scale = c_puct * np.sqrt(node.visits + 1)
            move, child = max(node.children.items(), key=lambda item:
                -item[1].q + scale * item[1].prior / (1 + item[1].visits))
            state.push(move)
            node = child
            path.append(node)
            if not node.children:
                break
        value = terminal_value(state)
        if value is None:
            value = expand(node, state, network, device)
        # Alternate point of view for each step towards the root.
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value
    moves = list(root.children)
    counts = np.array([root.children[m].visits for m in moves], dtype=np.float64)
    return moves, counts / counts.sum()


def choose_move(moves: list[chess.Move], pi: np.ndarray, temperature: float,
                rng: np.random.Generator) -> chess.Move:
    if temperature <= 0:
        return moves[int(np.argmax(pi))]
    weights = np.power(pi, 1.0 / temperature)
    weights /= weights.sum()
    return moves[int(rng.choice(len(moves), p=weights))]


def self_play(network: PolicyValueNet, device: torch.device, simulations: int,
              max_plies: int, temperature_moves: int,
              rng: np.random.Generator,
              recorded_moves: list[chess.Move] | None = None) -> tuple[list[tuple], str]:
    board = chess.Board()
    positions = []
    for ply in range(max_plies):
        result = terminal_value(board)
        if result is not None:
            break
        moves, pi = search(board, network, device, simulations, noise=True, rng=rng)
        positions.append((encode(board), np.array([move_index(m) for m in moves],
                                                   dtype=np.int64), pi.astype(np.float32), board.turn))
        selected = choose_move(moves, pi, 1.0 if ply < temperature_moves else 0.0, rng)
        board.push(selected)
        if recorded_moves is not None:
            recorded_moves.append(selected)

    outcome = board.outcome(claim_draw=True)
    # Artificial move cap is a draw; increase max_plies to reduce this bias.
    winner = outcome.winner if outcome is not None else None
    examples = [(x, idx, probs, float(0 if winner is None else
                 (1 if color == winner else -1))) for x, idx, probs, color in positions]
    result = board.result(claim_draw=True) if outcome is not None else "1/2-1/2 (move cap)"
    return examples, result


def learn(network: PolicyValueNet, optimizer: torch.optim.Optimizer, replay: deque,
          device: torch.device, steps: int, batch_size: int, sampler: random.Random) -> tuple[float, float]:
    if not replay:
        raise ValueError("cannot train with empty replay")
    network.train()
    policy_losses, value_losses = [], []
    for _ in range(steps):
        batch = sampler.sample(list(replay), k=min(batch_size, len(replay)))
        states = torch.from_numpy(np.stack([ex[0] for ex in batch])).to(device)
        targets = torch.tensor([ex[3] for ex in batch], dtype=torch.float32, device=device)
        logits, values = network(states)
        losses = []
        for i, (_, indices, pi, _) in enumerate(batch):
            legal_logits = logits[i, torch.as_tensor(indices, device=device)]
            probs = torch.as_tensor(pi, device=device)
            losses.append(-(probs * F.log_softmax(legal_logits, dim=0)).sum())
        policy_loss = torch.stack(losses).mean()
        value_loss = F.mse_loss(values, targets)
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def save_checkpoint(path: Path, network: PolicyValueNet, optimizer: torch.optim.Optimizer,
                    iteration: int, replay: deque, rng: np.random.Generator,
                    sampler: random.Random, width: int, blocks: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(),
                "iteration": iteration, "replay": list(replay), "rng": rng.bit_generator.state,
                "sampler": sampler.getstate(), "torch_rng": torch.get_rng_state(),
                "width": width, "blocks": blocks}, temp)
    os.replace(temp, path)


def load_checkpoint(path: Path, network: PolicyValueNet, optimizer: torch.optim.Optimizer,
                    device: torch.device, replay_size: int,
                    rng: np.random.Generator, sampler: random.Random) -> tuple[int, deque]:
    # Only load checkpoints you trust: PyTorch checkpoint files can execute code on load.
    data = torch.load(path, map_location=device, weights_only=False)
    network.load_state_dict(data["network"])
    optimizer.load_state_dict(data["optimizer"])
    rng.bit_generator.state = data["rng"]
    sampler.setstate(data["sampler"])
    torch.set_rng_state(data["torch_rng"].cpu())
    return data["iteration"], deque(data["replay"], maxlen=replay_size)


def run(args: argparse.Namespace) -> None:
    if args.simulations < 1 or args.max_plies < 1 or args.batch_size < 1 or args.steps < 1:
        raise ValueError("simulations, max-plies, batch-size and steps must be positive")
    if args.games < 1 or args.iterations < 1 or args.replay_size < 1:
        raise ValueError("games, iterations and replay-size must be positive")
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
    width, blocks = args.width, args.blocks
    if args.resume:
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint at {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        width, blocks = saved["width"], saved["blocks"]
    network = PolicyValueNet(width, blocks).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = deque(maxlen=args.replay_size)
    start = 0
    if args.resume:
        start, replay = load_checkpoint(path, network, optimizer, device,
                                        args.replay_size, rng, sampler)
        print(f"Resumed after iteration {start} ({len(replay)} stored positions)", flush=True)
    print(f"Training on {device}, width={width}, blocks={blocks}", flush=True)
    recorded_once = False
    for iteration in range(start + 1, start + args.iterations + 1):
        results = {}
        for game in range(args.games):
            record = [] if (args.record or args.record_video) and not recorded_once else None
            samples, result = self_play(network, device, args.simulations, args.max_plies,
                                        args.temperature_moves, rng, recorded_moves=record)
            if record is not None:
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
            print(f"Iteration {iteration}, game {game + 1}/{args.games}: {result}, "
                  f"{len(samples)} plies", flush=True)
        p_loss, v_loss = learn(network, optimizer, replay, device,
                               args.steps, args.batch_size, sampler)
        save_checkpoint(path, network, optimizer, iteration, replay, rng, sampler, width, blocks)
        print(f"Iteration {iteration}: games={results}, replay={len(replay)}, "
              f"policy_loss={p_loss:.4f}, value_loss={v_loss:.4f}; saved {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play chess NN + PUCT MCTS trainer")
    parser.add_argument("--iterations", type=int, default=10, help="additional iterations to run")
    parser.add_argument("--games", type=int, default=4, help="self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=16, help="MCTS simulations per move")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--temperature-moves", type=int, default=30)
    parser.add_argument("--steps", type=int, default=32, help="gradient updates per iteration")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--record", action="store_true",
                        help="save first self-play game as PGN and print a Lichess replay link")
    parser.add_argument("--record-video", action="store_true",
                        help="also save first game as MP4 (requires optional video dependencies)")
    parser.add_argument("--record-dir", "--video-dir", dest="record_dir", default="recordings",
                        help="directory for recorded PGN/MP4 files")
    parser.add_argument("--video-fps", type=float, default=2.0,
                        help="video frames per second (one frame per half-move)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()