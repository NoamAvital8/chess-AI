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


## Replay the first game with `--record`

Use `--record` to capture only **game 1 of this command**, even when training runs
multiple games or iterations. Once game 1 finishes, a Lichess analysis URL is
printed immediately, a `.pgn` file is saved, and **training continues normally**.
No video encoding, external upload, or API account is required:

```powershell
python chess_ai.py --resume --iterations 3 --games 2 --simulations 8 --max-plies 200 --steps 16 --record
```

For example, a recording would produce:

```text
Recorded first game (123 plies). PGN: C:\...\recordings\selfplay_iteration_5_game_1.pgn
Watch/replay on Lichess: https://lichess.org/analysis/pgn/1._e4_e5_2._Nf3_...
Iteration 5, game 1/2: 1-0, 123 plies
...
Iteration 5, game 2/2: ...
```

Open the printed Lichess link to step through all of the moves and analyze them.
The complete sequence is encoded in the URL: it is **not a hosted/archived game**.
Because browsers limit URL lengths, extremely long games use the PGN fallback.
To see one on **Chess.com**, open https://www.chess.com/analysis and import
the saved `.pgn` file (Chess.com does not expose a documented direct-PGN URL).
Set `--record-dir my_records` to change the output directory.

### Optional: export an MP4 too

MP4 encoding takes additional time; install the optional packages and use:

```powershell
python -m pip install -r requirements-video.txt
python chess_ai.py --resume --iterations 1 --games 4 --simulations 8 --record-video
```

This does everything `--record` does **and** saves the first game's board as an
MP4 under `recordings/` (one frame per half-move, including the starting board).
Use `--video-fps 3` for faster playback. Board pieces use letters inside disks
(`N` = knight, `K` = king).

**Chess rules:** Games use `python-chess` legal moves and `board.outcome(claim_draw=True)`.
That covers checkmate, stalemate, insufficient material, threefold repetition,
claimable 50-move draw, automatic fivefold/75-move draws, castling, promotion,
and en passant (a capture, **not** a draw condition).

**Move-cap caveat:** Currently, unfinished games at `--max-plies` are treated as
draws in the training targets (a simplification we plan to improve). The exported
PGN instead correctly uses `Result "*"` for these unfinished games.

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