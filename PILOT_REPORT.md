# Training pilots

## Format 2 pilot — 18 October 2026

This was the first run after the training fixes: real-repetition draw rule, side-to-move
encoding, a 64×4 batch-normalised network, Gumbel search, batched self-play and the
endgame curriculum. In 10 iterations from random weights, the model went from no
better than random to **8 wins, 3 draws and 1 loss against a random player**. It also
scored 6/5/1 against its own starting weights. This is early, small-sample evidence
of learning, not an Elo estimate.

### Configuration

All defaults (8 games per iteration, 16 simulations, 300-ply cap, a quarter of the games
from endgames, 32 updates of batch 64, AdamW at lr 0.001), seed 42, 4 CPU threads:

```powershell
python chess_ai.py --checkpoint-dir checkpoints/pilot-v2-20261018 --iterations 10 --eval-every 5 --threads 4 --record --record-dir recordings/pilot-v2-20261018
```

The pilot ran the version of the search just before one last change. That change
batches each sequential-halving round in a single network call, fixing the root
choices once per round instead of after every simulation. The tests and an end-to-end
smoke run used the final code.

### Self-play and training

- 80 games, **24 decisive**. The 60 standard games ended by checkmate (22), threefold
  repetition (19), the move cap (13), insufficient material (4) and the 50-move rule (2).
- The 20 endgame games mostly ended by insufficient material (10, the stronger side
  lost its extra piece) or the 50-move rule (6). Only 2 ended in mate, so the model
  cannot convert these endgames yet.
- Policy loss fell from 3.07 to 2.53; value loss from 0.38 to 0.08.
- Final replay: 12,564 positions; **69% have a value label** (2,446 decisive, 6,218
  draws, 3,900 from capped games). The format 1 run had 9% labeled.
- Throughput was 24–31 positions per second in self-play, or 35–74 seconds per
  iteration. Training took about 5 seconds per iteration. Iteration 8 was slower because
  a benchmark ran at the same time.

### Evaluation

Twelve games per opponent: six fixed openings, both colours, 16 simulations, 400-ply cap.

| Iteration | Opponent | Wins | Draws | Losses | Unfinished | Completed score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | Random legal moves | 3 | 6 | 0 | 3 | 66.7% |
| 5 | Origin (iteration 0) | 2 | 4 | 2 | 4 | 50.0% |
| 5 | Previous (iteration 4) | 1 | 8 | 3 | 0 | 41.7% |
| 10 | Random legal moves | 8 | 3 | 1 | 0 | 79.2% |
| 10 | Origin (iteration 0) | 6 | 5 | 1 | 0 | 70.8% |
| 10 | Previous (iteration 9) | 2 | 9 | 1 | 0 | 54.2% |

For comparison, the format 1 model scored 0 wins, 3 draws and 1 loss against random
after 75 iterations.

### Value head

On 200 random positions, the mean value is +0.03 with White to move and −0.11 with
Black to move. The format 1 model gave +0.63 and −0.65, which meant it had learned
"White wins". The new model has not yet learned that KQ vs K is winning: it gives
−0.07 with the stronger side to move.

### Notes for a long run

- One evaluation round at iteration 10 took about 6 minutes, so `--eval-every` now
  defaults to 10.
- Throughput on this machine varied by about 3× during the session, partly because
  other Python processes were running. Compare positions per second only within one
  session.
- Watch the score against `origin`, the capped-game rate and the endgame terminations.
  More mates in endgame games would show that the value signal is reaching won
  material balances.

## Earlier pilot (format 1 trainer) — 18 October 2026

The capped-game fix, evaluation, checkpoint snapshots, and persistent metrics worked
in a fresh two-iteration CPU run. This pilot provides **no evidence of improved
playing strength**. Repetition and scarce decisive training examples remain the
main concerns before a long run.

### Configuration

- Fresh weights, seed 42; width 16, one residual block; PyTorch 2.10.0+cpu, two threads.
- Two iterations, two self-play games each, 32 MCTS simulations per move, 300-ply cap.
- Eight gradient steps per iteration, batch size 32, learning rate 0.001.
- Evaluation every iteration: two color-swapped games per opponent, 16 simulations,
  400-ply cap after the opening, evaluation seed 2026. With only one opening pair,
  this checks functionality rather than broad opening coverage or Elo.
- Training plus evaluation took approximately **118 seconds**, excluding startup.
- The original `checkpoints/latest.pt` was not used as an output. Pilot artifacts
  are under `checkpoints/pilot-first-pass-20261018/`.

Command:

```powershell
python chess_ai.py --checkpoint-dir checkpoints/pilot-first-pass-20261018 --iterations 2 --games 2 --simulations 32 --max-plies 300 --steps 8 --batch-size 32 --width 16 --blocks 1 --eval-every 1 --eval-games 2 --eval-simulations 16 --eval-max-plies 400 --record --record-dir recordings/pilot-first-pass-20261018
```

That directory now contains a checkpoint. Use `--resume` to continue it, or choose
a new directory to repeat the experiment from scratch.

### Self-play and training

| Iteration | Game results | New positions | Policy loss | Value loss | Positions/second |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Two unfinished games, both at 300 plies | 600 | 2.9681 | Not applicable | 9.48 |
| 2 | Black checkmates at ply 4; repetition draw at ply 204 | 208 | 3.0770 | 0.000171 | 10.61 |

The final replay contains **808 positions**:

- 600 unfinished positions: policy targets retained, value targets absent.
- 204 positions from a genuine draw: value target zero.
- Four positions from a decisive game: two targets +1 and two targets -1.

Only **25.7%** of replay positions have observed value labels; only **0.5%** are
decisive. The first iteration correctly skipped value training entirely.

The tiny second-iteration value loss does **not** establish strong evaluation:
its 256 sampled positions comprised 193 unfinished positions and 63 real-draw
positions, with **no decisive positions sampled**. This was reconstructed from the
saved replay and the seeded replay sampler, using the logged iteration sizes and
batch/step settings; the reconstructed labeled count matches the logged count.
The trainer now records decisive and draw minibatch counts directly for future runs.

### Evaluation

All ten evaluation games completed as claimable threefold-repetition draws.
None reached the evaluation cap.

| Candidate | Opponent | Wins | Draws | Losses | Unfinished |
| --- | --- | ---: | ---: | ---: | ---: |
| Iteration 1 | Random legal moves | 0 | 2 | 0 | 0 |
| Iteration 1 | Initial frozen model | 0 | 2 | 0 | 0 |
| Iteration 2 | Random legal moves | 0 | 2 | 0 | 0 |
| Iteration 2 | Initial frozen model | 0 | 2 | 0 | 0 |
| Iteration 2 | Iteration 1 frozen model | 0 | 2 | 0 | 0 |

The model-versus-model games repeated after only 14–17 plies. Against random play,
repetition happened after 121–339 plies. This points to an ongoing tendency toward
repetition, rather than mislabeled move caps explaining all draws. Games retain
the trainer's existing `claim_draw=True` convention, which automatically accepts
available draw claims.

### Artifacts

- Metrics: `checkpoints/pilot-first-pass-20261018/metrics.jsonl`.
- Resumable model/replay: `checkpoints/pilot-first-pass-20261018/latest.pt`.
- Initial and iteration 1/2 models:
  `checkpoints/pilot-first-pass-20261018/snapshots/9164524315e24e1987889aeec76906eb/`.
- First-game PGN:
  `recordings/pilot-first-pass-20261018/selfplay_iteration_1_game_1.pgn`.

### Recommendation before heavy training

Collect more decisive experience and confirm that training actually samples it.
Human games or self-play from simple endgame positions could help. Uniform sampling
of positions naturally gives a 204-ply draw much more representation than a
four-ply loss; a later sampling experiment should consider that imbalance without
silently redefining all draws as losses.

For human play, keep the model fixed during each game, save complete games in a
separate buffer, use a limited mixture with self-play, and obtain policy targets
through search/reanalysis. The outcome supplies a value target; a loss alone does
not identify the best alternative move. Human play is a recommended next feature,
not part of this implementation.

Before judging strength, increase the evaluation sample and opening coverage while
keeping opponent versions, search settings, and seeds fixed. Do not infer an Elo
gain from this pilot.