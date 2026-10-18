# Chess AI — self-play with neural networks and MCTS

A minimal, **experimental** AlphaZero-style chess trainer in Python. A **single shared
policy/value neural network** plays both White and Black. It learns from self-play and
optional human games,
without handcrafted material evaluation or supervised engine data. **Gumbel AlphaZero
MCTS** searches future positions using the network's policy priors and value estimates.

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
python chess_ai.py --iterations 10
```

Defaults: 8 self-play games per iteration (played concurrently), 16 simulations per
move, a 64-channel × 4-block network, 32 updates of batch size 64, and a quarter of the
games started from simple endgames. On a CPU, `--threads 4` usually helps; lower
`--simulations`, `--width` or `--blocks` for speed. More simulations give better search
targets but **fewer games per hour**.
`--max-plies` (default 300) stops overlong games as **unfinished**, keeping their policy
targets and excluding them from value loss. Only actual completed draws receive value zero.
Save files are created under `checkpoints/latest.pt` after every iteration.
Use a fresh `--checkpoint-dir` for a new experiment; an existing checkpoint requires
`--resume` so it cannot be overwritten accidentally.

Resume for **10 more iterations**:

```powershell
python chess_ai.py --resume --iterations 10
```

Model architecture is restored automatically; keep other training settings consistent
when resuming. The replay buffer and optimizer state are stored in the checkpoint.
The saved learning rate is retained unless `--lr` is explicitly supplied.
**Never load untrusted `.pt` files** (PyTorch pickle serialization).

### Models from before format 2

Checkpoints and snapshots record a `format_version`. Format 2 changed the board
encoding (side-to-move view), the network and the replay storage, so **older
(format 1) models cannot be resumed, played against or used as evaluation opponents**.
Loading one stops with an explanation. Start a new run in a fresh directory, for example
`--checkpoint-dir checkpoints/v2`, or move the old `checkpoints/` folder aside first so
that `play.py` and `chess_ai.py` keep sharing the default `checkpoints/latest.pt`.

## What's implemented

- `python-chess` handles **legal moves and termination**. Threefold repetition and the
  50-move rule end a game **when they actually happen** (`rules.game_outcome`); a game is
  not drawn merely because the side to move *could* repeat with its next move.
  Checkmate, stalemate, insufficient material, fivefold repetition and the 75-move
  rule apply as usual. Training, search, evaluation, play and recordings share this rule.
- Inputs are encoded **from the side to move's view**: Black's positions are mirrored
  so "my pieces" always move up the board, and there is no colour plane. A position
  and its mirror image are the same input, so the network learns each pattern once
  and cannot use "White to move" as a shortcut for the result.
  Planes: own/opponent pieces, castling rights, en-passant square, halfmove clock,
  repetition flags and a constant plane. The rules engine keeps the full game history.
- Network: a batch-normalised residual CNN (default 64 channels × 4 blocks, so each
  square sees the whole board). The policy head scores each (from, to) square pair with
  a dot product of per-square embeddings plus a promotion term, instead of a
  20,480-wide dense layer. The value head predicts the result for the side to move.
- 20,480 possible actions distinguish ordinary moves and all four promotion types;
  **only legal actions** are normalized and sampled.
- **Gumbel AlphaZero search** (Danihelka et al., 2022): the root samples candidate
  moves with Gumbel noise and spends simulations by sequential halving. Interior nodes
  use the improved policy. Values back up with alternating signs. Self-play keeps the
  noise for every move; evaluation and human play search deterministically.
- The policy target is the **improved policy** `softmax(logits + σ(completed Q))`. Every
  legal move gets a target, including unvisited ones (they use a mixed value estimate),
  so even 16 simulations give a useful, dense target.
- Value loss uses observed final outcomes (+1 / 0 / −1); unfinished positions have no
  value label. If a minibatch has no value labels, only policy loss is optimized.
- **Endgame curriculum:** `--endgame-fraction` (default 0.25) starts that share of
  self-play games from random legal positions in KQ-K, KR-K, KRR-K, KQR-K or KP-K, with
  either colour stronger. Checkmates happen there quickly, so the value head sees
  decisive results early. Their labels are still real game outcomes. The share is spread
  evenly over the run, and logs record each game's start.
- **Batched self-play:** all games of an iteration run together, and their search
  leaves are evaluated in one network call per step. Evaluation games are batched the
  same way.
- Training uses AdamW and a vectorised policy loss. Replay stores compact uint8 board
  planes. Each iteration saves resumable optimizer, network, RNG and replay states.

## Evaluation and training metrics

Every iteration saves a lightweight model under `checkpoints/snapshots/<run-id>/`,
including an initial snapshot before training. Full resumable state remains in
`latest.pt`. Snapshots accumulate; allow roughly 1.5 MB per default-size model.

By default, every 10 iterations the model plays 12 games **per opponent** (all six
fixed openings, both colours) against:

- A uniformly random legal-move player.
- `origin`: the experiment's first model, stored in the checkpoint. This is a fixed
  anchor that stays the same across resumed sessions.
- `session_start`: the model at the start of this invocation, when that differs.
- `previous`: the preceding iteration's model, when that differs.

Games use fixed opening pairs with colors swapped, a separate fixed evaluation
seed, deterministic search moves, and no exploration noise. Both neural players use the
same simulation budget. Evaluation does not train the model or add games to replay.
The suite has six openings, so more than 12 games repeat them; between two neural
players a repeated opening replays the identical game. For more meaningful
comparisons, use a larger search budget:

```powershell
python chess_ai.py --resume --iterations 5 --eval-every 5 --eval-games 12 --eval-simulations 64 --eval-max-plies 600
```

Use `--eval-every 0` to disable evaluation. Add trusted older models with repeatable
`--eval-checkpoint path/to/model.pt` options. Keep evaluation settings the same when
comparing runs. A few games are a functionality check, not an Elo estimate.

Unfinished evaluation games count separately from draws. `completed_score` uses
only completed games and is `null` when none finish. Because that score can be
biased by truncation, metrics also include completion rate and lower/upper score
bounds treating unfinished games as possible losses/wins. These are bounds on
unfinished outcomes, **not statistical confidence intervals**.

Append-only JSONL records are saved to `checkpoints/metrics.jsonl`; override with
`--metrics-file`. Each invocation has a run ID and records its effective configuration.
Events are flushed after every game and phase, including:

- Game result, termination reason, plies, move-cap status, time and throughput.
- Policy/value losses, sampled value-label coverage (including decisive/draw counts), learning rate, replay composition,
  capped-game rate, training/checkpoint timing and immutable snapshot paths.
- Per-opponent evaluation W/D/L, unfinished games, score bounds and game details.

`value_loss: null` means no value-labeled positions were sampled; it does not mean
the model achieved zero error. Repetition and 50-move draws count only when they
actually occur (see above).

## Play against the model and learn after each game

### Click-to-move window

```powershell
python play.py --gui
```

On Windows you can also double-click `play_window.pyw`. This launcher uses the
project directory automatically, so it finds the same shared model. Choose White
or Black and click **Start game**, then click a piece and one of its highlighted
destination squares. Click another piece to change selection; right-click or Esc
clears it. Promotion presents Queen/Rook/Bishop/Knight buttons. Castling uses the
king's destination square, and en passant works through normal legal moves.

The window shows move history, highlights the last move and check, and supports
flipping the board. **Resign** finishes a loss and triggers learning. **End game**
saves an unfinished game without learning. Completed games use exactly the same
post-game reanalysis, mixed replay and shared `checkpoints/latest.pt` as terminal
play and self-play. **New game** reloads the latest saved model.

Search and training run in a worker thread, keeping the window responsive. If you
close it during completed-game learning, it finishes saving before closing. All
GUI operations run on the main thread. Tkinter is included in standard Windows
Python installations with Tcl/Tk enabled; no new pip package is required.

Search and learning options also work with `--gui`, for example:

```powershell
python play.py --gui --color black --simulations 64
```

### Terminal play

From the project directory, start a terminal game as White:

```powershell
python play.py --color white
```

Human play and self-play both load and update **the same** `checkpoints/latest.pt`
by default. After your completed game, the next self-play run resumes those updated
weights, optimizer and both replay buffers. Snapshots preserve older versions for
evaluation; there is one current model. To play with a different experiment,
point both modes at that experiment's checkpoint directory:

```powershell
python play.py --checkpoint-dir checkpoints/pilot-first-pass-20261018
```

Choose `--color black`, or play multiple games with `--games 4 --color alternate`.
The board is oriented to your color. Uppercase pieces are White; lowercase pieces
are Black. Enter SAN (`e4`, `Nf3`, `O-O`) or UCI (`e2e4`, `e7e8q`) moves. Commands:

- `legal`: list legal moves; `board`: redisplay the board; `help`: show commands.
- `resign`: finish the game as a loss and learn from it.
- `quit`, EOF, or Ctrl+C during play: save an unfinished game and skip learning.

Weights remain fixed throughout each game. After checkmate, an actual draw, or
resignation, the trainer uses the observed outcome from each position's side-to-move
perspective. After the game, stronger MCTS reanalyzes its positions to replace the
search targets generated before the result was known. Matching older human replay
positions also receive refreshed policy targets while retaining their actual
outcomes. Human moves are not automatically used as imitation targets.

The sampler covers new-game positions before repeating them, within the human
batch quota, and includes recent experience in every update. Short games are
fully covered when the update budget is sufficient; actual distinct-position
coverage is printed and logged. Resigning before any moves creates no positions.

Defaults are 32 simulations per move, 512 reanalysis simulations per position, and
learning rate 0.0001. By default the number of post-game updates is chosen so that each
position of the finished game is trained on about once (`--game-passes 1`). Many passes
over one game make the model memorise that game's result instead of learning from it.
`--steps N` sets an exact number of updates instead. Post-game reanalysis takes extra
time, especially on CPU; adjust `--reanalyze-simulations`, `--game-passes`,
`--batch-size`, or `--lr` as needed. Console output and metrics show the game's
refreshed-target policy/value losses before and after learning. These measure fit,
not strength.

Human play and its reanalysis also perform bounded proofs of forcing checking
mates (up to three checking moves, with a 300-node budget per candidate). Proven
winning moves are preferred, and proven losing moves are excluded when alternatives
exist. This uses legal chess outcomes, with no material evaluator. Exhausted or
unsuccessful proofs remain unknown; this is not a guarantee of tactical safety.
Use `--tactical-checks 0` to disable or `--tactical-nodes` to adjust the budget.
Ordinary self-play and evaluation use plain search, so their results can help
distinguish learned strength from these additional human-mode search checks.
Human games use the same repetition and 50-move rules as training.
Games stopped by the 600-ply default cap are unfinished and skip learning.

Human games, their replay, and their metrics retain their source identity:

- PGNs: `checkpoints/games/` (including unfinished games).
- Metrics: `checkpoints/human_metrics.jsonl`.
- Learned model, optimizer, both replay buffers, RNG states and completed-human-game
  count: `checkpoints/latest.pt` (shared with self-play).
- Model snapshots before the session and after each completed game:
  `checkpoints/snapshots/<run-id>/`.

**Why separate buffers?** Both sources train the same network. Separate buffers let
us control their mixture, measure their effects, and retain the origin of the labels.
This helps limit overfitting to one human opponent and avoids a large self-play
buffer drowning out a newly finished human game. It is bookkeeping and sampling
control; it does not withhold human experience from the model.

`--human-fraction 0.25` targets roughly 25% human positions per batch when enough
positions from both sources exist. Available data can change the actual ratio;
with no self-play data (for example, loading a lightweight snapshot), updates use
human data alone. Metrics record actual human, self-play, recent-game, decisive,
and draw sample counts. Separate buffers are retained in the same resumable checkpoint.

To continue ordinary self-play with the human-trained model and both data sources:

```powershell
python chess_ai.py --resume --iterations 2 --human-fraction 0.25
```

Ordinary training also accepts `--human-replay-size` (default 5000) and
`--human-fraction`; setting the latter to zero disables human sampling while
preserving the stored human replay. Switch between self-play and human play;
both commands take an exclusive lock on the checkpoint directory to prevent
simultaneous sessions from overwriting each other's updates. The lock releases
automatically when a session exits. An older already-running trainer should be
stopped before starting this shared-checkpoint workflow.

`--checkpoint` can seed a new experiment directory from a trusted full checkpoint
or lightweight snapshot when that directory has no `latest.pt` yet. Existing
experiments always resume their own current model.

## Short pilot before a long run

Use a separate directory to preserve existing training:

```powershell
python chess_ai.py --checkpoint-dir checkpoints/pilot --iterations 10 --threads 4
```

This is a small CPU pilot to verify data flow and observe draw/cap rates. Its match
sample is too small to establish strength improvement. Before scaling, inspect
value-label coverage, real draw reasons, unfinished-game rates, and throughput;
increase evaluation games/opening coverage for any strength claim.

Human play can provide decisive games and new positions, but losing supplies an
outcome signal rather than directly identifying the best alternative move.
Continue using fixed evaluation to check whether post-game updates improve strength.


## Replay the first game with `--record`

Use `--record` to capture only **the first standard-start game of this command**
(endgame-curriculum games are skipped), even when training runs multiple games or
iterations. Once game 1 finishes, a Lichess analysis URL is
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

**Chess rules:** Games use `python-chess` legal moves and `rules.game_outcome`.
That covers checkmate, stalemate, insufficient material, threefold repetition and the
50-move rule (once they have actually occurred), fivefold/75-move draws, castling,
promotion, and en passant (a capture, **not** a draw condition).

**Move caps:** Unfinished games use no value target and the exported PGN uses
`Result "*"`. They are reported separately from actual draws in training and evaluation.

## Testing

```powershell
python -m pip install pytest
python -m pytest -q
```

## Known limitations / next steps

- Single-process self-play. Leaves are batched across concurrent games, not within one
  search (no virtual loss), so the batch size equals the number of active games.
  Training speed is modest on CPU, and Elo improvement is not guaranteed.
- Small evaluation samples cannot establish Elo improvement; no Stockfish evaluation
  is implemented. Loss alone is not a strength metric.
- Abbreviated repetition features are approximations; full chess adjudication remains
  the responsibility of `python-chess`. Masking unfinished games fixes their labels
  but does not create decisive examples; the endgame curriculum supplies some.
- Memory/checkpoint sizes scale with `--replay-size`.
