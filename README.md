# Chess AI — self-play with neural networks and MCTS

A minimal, **experimental** AlphaZero-style chess trainer in Python. A **single shared
policy/value neural network** plays both White and Black. It learns from its own games,
without handcrafted material evaluation or supervised engine data. **PUCT MCTS** searches
future positions using the network's policy priors and value estimates.

This is a learning baseline, **not** a strong or production-grade chess engine. Starting
from random network weights can require substantial compute to learn meaningful chess.

## Setup (Windows PowerShell)

Requires Python 3.10+ (3.11 or 3.12 recommended). In a fresh clone:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch installation varies for GPU/CUDA; see https://pytorch.org/get-started/locally/.
CPU training works with standard PyTorch wheels.

## Quick smoke run

```powershell
python chess_ai.py --iterations 1 --games 1 --simulations 2 --max-plies 30 --steps 2 --batch-size 8 --width 16 --blocks 1
```

## Start training

```powershell
python chess_ai.py --iterations 10 --games 4 --simulations 16 --steps 32
```

On a CPU, try lower `--simulations`, `--width`, and `--blocks` for speed.
Larger simulations generally yield better search targets but **dramatically fewer games per hour**.
`--max-plies` terminates overlong games as draws, which is a training approximation.
Save files are created under `checkpoints/latest.pt` after every iteration.

Resume for **10 more iterations**:

```powershell
python chess_ai.py --resume --iterations 10
```

Model architecture is restored automatically; keep other training settings consistent
when resuming. The replay buffer and optimizer state are stored in the checkpoint.
**Never load untrusted `.pt` files** (PyTorch pickle serialization).

## What's implemented

- `python-chess` handles **legal moves and all standard termination rules**, including
  checkmate, stalemate, repetition and the 50-move rule (draws can be claimed).
- State features: 12 piece planes, side to move, castling rights, en-passant square,
  halfmove clock and repetition flags. The rules engine retains full game history.
- A small residual CNN predicts **move logits** and a value from the player-to-move's view.
- 20,480 possible actions distinguish ordinary moves and all four promotion types;
  **only legal actions** are normalized and sampled.
- MCTS uses the policy as a prior, the value head at new leaves, alternating value signs
  as it backs up, and root Dirichlet noise during self-play.
- Policy loss matches **search visit distributions** (not the network's own raw choices),
  value loss uses final game outcomes (+1 / 0 / −1).
- Training samples recent games from a replay buffer and saves resumable optimizer,
  network, RNG and replay states after each iteration.

## Testing

```powershell
python -m pip install pytest
python -m pytest -q
```

## Known limitations / next steps

- Simple single-process CPU self-play; search expands one state at a time (no batched GPU
  inference), so training speed is modest. No guaranteed Elo improvement.
- No arena against older versions or Stockfish yet, so loss alone is **not** a strength
  metric. Add a fixed evaluation harness to measure progress.
- Max-ply draws and abbreviated repetition features are approximations; full chess
  adjudication remains the responsibility of `python-chess`.
- Memory/checkpoint sizes scale with `--replay-size`.