import random
from collections import deque

import chess
import numpy as np
import torch
import pytest

from chess_ai import (ACTION_SIZE, INPUT_PLANES, PolicyValueNet, SearchResult, considered_visits,
                      encode, learn, make_optimizer, move_index, move_indices, search,
                      self_play, terminal_value)
from rules import game_outcome


@pytest.fixture(autouse=True)
def small_cpu_tests():
    torch.set_num_threads(1)
    torch.manual_seed(3)


def test_features_and_move_index():
    board = chess.Board()
    x = encode(board)
    assert x.shape == (INPUT_PLANES, 8, 8)
    assert x.dtype == np.uint8
    assert x[:12].sum() == 32
    assert x[0, 1].sum() == 8 and x[6, 6].sum() == 8  # own pawns rank 2, opponent's rank 7
    assert np.all(x[12] == 1)
    assert np.all(x[13:17] == 1)
    white, black = chess.WHITE, chess.BLACK
    assert move_index(chess.Move.from_uci('e7e8q'), white) != move_index(chess.Move.from_uci('e7e8n'), white)
    assert 0 <= move_index(chess.Move.from_uci('e2e4'), white) < ACTION_SIZE
    # Black's moves are seen from Black's side of the board.
    assert move_index(chess.Move.from_uci('e7e5'), black) == move_index(chess.Move.from_uci('e2e4'), white)
    board.push_san('e4')
    assert encode(board)[0, 6].sum() == 0 and encode(board)[0, 1].sum() == 8


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "r3k2r/pp3ppp/2n5/3pP3/8/8/PPP2PPP/R3K2R w Kq d6 0 12",
    "8/8/4k3/8/2P5/8/5K2/8 b - - 37 60",
])
def test_encoding_is_identical_for_mirror_image_positions(fen):
    board = chess.Board(fen)
    mirror = board.mirror()
    assert np.array_equal(encode(board), encode(mirror))
    for move in board.legal_moves:
        mirrored = chess.Move(chess.square_mirror(move.from_square),
                              chess.square_mirror(move.to_square), move.promotion)
        assert mirrored in mirror.legal_moves
        assert move_index(move, board.turn) == move_index(mirrored, mirror.turn)


def test_halfmove_plane_is_scaled_inside_the_network():
    board = chess.Board("8/8/4k3/8/8/8/5K2/7R w - - 73 90")
    assert encode(board)[18, 0, 0] == 73
    model = PolicyValueNet(width=8, blocks=0)
    scaled = torch.from_numpy(encode(board)).float() * model.plane_scale
    assert torch.isclose(scaled[18, 0, 0], torch.tensor(0.73)) and scaled[12, 0, 0] == 1
    assert "plane_scale" not in model.state_dict()  # derived, not a saved weight


def test_search_is_legal_normalized_and_dense():
    model = PolicyValueNet(width=8, blocks=1)
    board = chess.Board()
    result = search(board, model, torch.device('cpu'), simulations=3)
    assert set(result.moves) == set(board.legal_moves) and result.move in board.legal_moves
    assert np.isclose(result.policy.sum(), 1)
    # Completed Q-values give every legal move a target, not only the visited ones.
    assert np.count_nonzero(result.policy) == len(result.moves)
    assert board.fen() == chess.STARTING_FEN and not board.move_stack


SMOTHERED = "6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1"  # Nf7# among nine legal moves


def test_search_finds_mate_in_one_despite_untrained_network():
    board = chess.Board(SMOTHERED)
    model = PolicyValueNet(width=8, blocks=0)
    result = search(board, model, torch.device("cpu"), simulations=32)
    assert result.move == chess.Move.from_uci("h6f7")
    assert result.policy[result.moves.index(result.move)] == result.policy.max()


def test_root_restriction_and_single_simulation():
    board = chess.Board()
    allowed = {chess.Move.from_uci("g1f3"), chess.Move.from_uci("d2d4")}
    result = search(board, PolicyValueNet(8, 0), torch.device("cpu"), 1, root_moves=allowed)
    assert result.move in allowed
    assert all(p == 0 for move, p in zip(result.moves, result.policy) if move not in allowed)
    assert np.isclose(result.policy.sum(), 1)
    with pytest.raises(ValueError, match="restriction"):
        search(board, PolicyValueNet(8, 0), torch.device("cpu"), 1,
               root_moves={chess.Move.from_uci("e2e5")})


@pytest.mark.parametrize("considered,simulations", [(1, 5), (2, 8), (16, 16), (16, 64), (20, 7)])
def test_sequential_halving_schedule(considered, simulations):
    schedule = considered_visits(considered, simulations)
    assert len(schedule) == simulations
    if considered > 1:
        assert schedule[:min(considered, simulations)] == [0] * min(considered, simulations)


class CountingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner, self.batches = PolicyValueNet(8, 0), []

    def forward(self, states):
        self.batches.append(len(states))
        return self.inner(states)


@pytest.mark.parametrize("simulations,batches", [(16, [1, 16]), (32, [1, 16, 8, 4, 4])])
def test_each_halving_round_is_one_network_batch(simulations, batches):
    model = CountingNet()
    search(chess.Board(), model, torch.device("cpu"), simulations)
    assert model.batches == batches


def test_terminal_score_sign_and_draw():
    board = chess.Board()
    for move in ('f3', 'e5', 'g4', 'Qh4#'):
        board.push_san(move)
    assert terminal_value(board) == -1.0  # White is to move, Black won
    assert terminal_value(chess.Board()) is None
    assert terminal_value(chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')) == 0.0


def test_repetition_draw_needs_an_actual_threefold_repetition():
    board = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"):
        board.push_san(san)
    # Black could repeat with Ng8, which python-chess's claim_draw counts as a draw now.
    assert board.can_claim_threefold_repetition() and game_outcome(board) is None
    board.push_san("Ng8")
    assert game_outcome(board).termination == chess.Termination.THREEFOLD_REPETITION
    fifty = chess.Board("8/8/4k3/8/8/8/5K2/7R w - - 100 90")
    assert game_outcome(fifty).termination == chess.Termination.FIFTY_MOVES
    assert game_outcome(chess.Board("8/8/4k3/8/8/8/5K2/7R w - - 99 90")) is None
    mate_on_100 = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 99 90")
    mate_on_100.push_san("Rh8#")  # checkmate takes precedence over the 50-move rule
    assert game_outcome(mate_on_100).termination == chess.Termination.CHECKMATE


def test_search_leaf_rules_match_game_outcome():
    from chess_ai import _leaf

    rng = random.Random(4)
    for _ in range(40):
        board = chess.Board()
        while game_outcome(board) is None and board.ply() < 300:
            moves = list(board.legal_moves)
            quiet = [m for m in moves if not board.is_capture(m)
                     and board.piece_type_at(m.from_square) != chess.PAWN]
            board.push(rng.choice(quiet if quiet and rng.random() < .8 else moves))
            assert _leaf(board)[0] == terminal_value(board)


def test_self_play_train_and_resume(tmp_path):
    from chess_ai import save_checkpoint, load_checkpoint
    model = PolicyValueNet(width=8, blocks=1)
    device = torch.device('cpu')
    rng = np.random.default_rng(11)
    sampler = random.Random(11)
    examples, result = self_play(model, device, simulations=1, max_plies=4, rng=rng)
    assert len(examples) == 4
    assert result == '* (move cap)'
    assert all(np.isclose(pi.sum(), 1) and z is None for _, _, pi, z in examples)
    buffer = deque(examples, maxlen=16)
    optimizer = make_optimizer(model, 0.001)
    p, v = learn(model, optimizer, buffer, device, steps=1, batch_size=2, sampler=sampler)
    assert np.isfinite(p) and np.isfinite(v)
    path = tmp_path / 'latest.pt'
    save_checkpoint(path, model, optimizer, 1, buffer, rng, sampler, 8, 1)
    clone = PolicyValueNet(width=8, blocks=1)
    copy_opt = make_optimizer(clone, 0.001)
    iteration, restored = load_checkpoint(path, clone, copy_opt, device, 16,
                                           np.random.default_rng(), random.Random())
    assert iteration == 1 and len(restored) == 4
    assert all(ex[3] is None for ex in restored)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(),
                                                  clone.state_dict().values()))


def test_batched_games_match_sequential_games():
    from chess_ai import run_batched, self_play_steps

    model = PolicyValueNet(width=8, blocks=1).eval()
    device = torch.device("cpu")
    sequential = [self_play(model, device, 4, 6, np.random.default_rng(seed)) for seed in (1, 2, 3)]
    batched = run_batched([self_play_steps(model, 4, 6, np.random.default_rng(seed))
                           for seed in (1, 2, 3)], device)
    for (left, _), (right, _) in zip(sequential, batched):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            assert np.allclose(a[2], b[2], atol=1e-5)


def test_policy_loss_matches_per_example_reference():
    from chess_ai import batch_tensors, policy_cross_entropy

    model = PolicyValueNet(width=8, blocks=1).eval()
    batch = []
    for fen in (chess.STARTING_FEN, "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"):
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        pi = np.random.default_rng(1).dirichlet(np.ones(len(moves))).astype(np.float32)
        batch.append((encode(board), move_indices(board, moves), pi, None))
    states, indices, targets, valid = batch_tensors(batch, torch.device("cpu"))
    with torch.no_grad():
        logits, _ = model(states)
        losses = policy_cross_entropy(logits, indices, targets, valid)
        for row, (_, idx, pi, _) in enumerate(batch):
            reference = -(torch.from_numpy(pi) * torch.log_softmax(logits[row, idx], 0)).sum()
            assert torch.isclose(losses[row], reference, atol=1e-5)


def test_endgame_starts_are_legal_unfinished_and_reproducible():
    from chess_ai import ENDGAMES, endgame_start, is_endgame_game

    first = [endgame_start(np.random.default_rng(5)) for _ in range(2)]
    assert first[0][0].fen() == first[1][0].fen()
    rng = np.random.default_rng(9)
    names = set()
    for _ in range(60):
        board, name = endgame_start(rng)
        names.add(name)
        assert board.is_valid() and terminal_value(board) is None
        assert len(board.piece_map()) == 2 + len(ENDGAMES[name])
        assert not board.castling_rights and board.ep_square is None
    assert names == set(ENDGAMES)
    assert [is_endgame_game(n, .25) for n in range(8)] == [False, False, False, True] * 2
    assert not any(is_endgame_game(n, 0) for n in range(10))
    assert all(is_endgame_game(n, 1) for n in range(10))


def test_self_play_from_endgame_start_records_actual_outcome():
    from chess_ai import self_play_steps, run_batched

    board = chess.Board(SMOTHERED)
    info = {}
    model = PolicyValueNet(width=8, blocks=0)
    examples, result = run_batched([self_play_steps(model, 32, 10, np.random.default_rng(1),
                                                    board=board, game_info=info)],
                                   torch.device("cpu"))[0]
    assert board.fen() == SMOTHERED  # the caller's board is untouched
    assert result == "1-0" and info["termination"] == "CHECKMATE"
    assert len(examples) == 1 and examples[0][3] == 1.0


def test_record_moves_pgn_and_lichess_link(tmp_path):
    import chess.pgn
    from recording import lichess_url, write_pgn

    board = chess.Board()
    moves = []
    for san in ("e4", "e5", "Nf3", "Nc6"):
        move = board.parse_san(san)
        moves.append(move)
        board.push(move)
    path = tmp_path / "first.pgn"
    assert write_pgn(moves, path) == "*"  # unfinished != official draw
    with path.open(encoding="utf8") as source:
        saved = chess.pgn.read_game(source)
    assert list(saved.mainline_moves()) == moves
    assert saved.headers["Result"] == "*"
    assert "1._e4_e5_2._Nf3_Nc6" in lichess_url(moves)
    assert lichess_url(moves).startswith("https://lichess.org/analysis/pgn/")


def run_args(tmp_path, **overrides):
    from argparse import Namespace

    values = dict(simulations=1, max_plies=2, batch_size=2, steps=1, games=2, iterations=2,
                  replay_size=40, seed=1, device="cpu", threads=1,
                  checkpoint_dir=str(tmp_path / "ckpt"), width=8, blocks=1, lr=0.001,
                  resume=False, record=False, record_video=False,
                  record_dir=str(tmp_path / "recordings"), video_fps=2.0, eval_every=0,
                  endgame_fraction=0.0)
    values.update(overrides)
    return Namespace(**values)


def test_record_only_first_game_during_multi_game_run(tmp_path, capsys):
    from chess_ai import run
    import chess.pgn

    run(run_args(tmp_path, record=True, eval_every=1, eval_games=2, eval_simulations=1,
                 eval_max_plies=1))
    pgns = list((tmp_path / "recordings").glob("*.pgn"))
    assert len(pgns) == 1
    assert "iteration_1_game_1" in pgns[0].name
    with pgns[0].open(encoding="utf-8") as stream:
        game = chess.pgn.read_game(stream)
    assert len(list(game.mainline_moves())) == 2
    output = capsys.readouterr().out
    assert output.count("Watch/replay on Lichess:") == 1
    assert "Iteration 2, game 2/2" in output

    import json
    records = [json.loads(line) for line in
               (tmp_path / "ckpt" / "metrics.jsonl").read_text().splitlines()]
    iterations = [row for row in records if row["event"] == "training_iteration"]
    assert len(iterations) == 2
    assert all(row["value_loss"] is None and row["capped_game_rate"] == 1
               for row in iterations)
    # Evaluation games must never enter the training replay.
    assert iterations[-1]["replay"]["positions"] == 8
    evaluations = [row for row in records if row["event"] == "evaluation"]
    assert len(evaluations) == 5  # random/origin, then random/origin/previous
    assert {row["opponent"] for row in evaluations} == {"random", "origin", "previous"}
    assert all(row["truncated"] == 2 and row["draws"] == 0 and
               row["completed_score"] is None for row in evaluations)
    assert records[-1]["event"] == "run_end"
    assert len(list((tmp_path / "ckpt" / "snapshots").rglob("*.pt"))) == 3


def test_origin_anchor_survives_resume(tmp_path):
    import json
    from chess_ai import run

    args = run_args(tmp_path, iterations=1, games=1, max_plies=1, batch_size=1)
    run(args)
    args.resume, args.eval_every, args.eval_games = True, 1, 2
    args.eval_simulations, args.eval_max_plies = 1, 1
    run(args)
    records = [json.loads(line) for line in
               (tmp_path / "ckpt" / "metrics.jsonl").read_text().splitlines()]
    starts = [row for row in records if row["event"] == "run_start"]
    assert starts[1]["origin_snapshot"] == starts[0]["initial_snapshot"]
    labels = [row["opponent"] for row in records if row["event"] == "evaluation"]
    # The resumed session's start is also the previous iteration's model.
    assert labels == ["random", "origin", "session_start"]


def test_endgame_curriculum_runs_and_is_logged(tmp_path):
    import json
    from chess_ai import run

    run(run_args(tmp_path, games=4, iterations=1, max_plies=3, endgame_fraction=0.5))
    records = [json.loads(line) for line in
               (tmp_path / "ckpt" / "metrics.jsonl").read_text().splitlines()]
    games = [row for row in records if row["event"] == "self_play_game"]
    assert [row["start"] != "standard" for row in games] == [False, True, False, True]
    assert all("start_fen" in row for row in games if row["start"] != "standard")
    assert next(row for row in records if row["event"] == "training_iteration")["endgame_games"] == 2


def test_capped_positions_do_not_update_value_head():
    model = PolicyValueNet(width=8, blocks=1)
    board = chess.Board()
    moves = list(board.legal_moves)
    sample = (encode(board), move_indices(board, moves),
              np.eye(1, len(moves), dtype=np.float32)[0], None)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = make_optimizer(model, 0.001)
    stats = {}
    learn(model, optimizer, deque([sample]), torch.device("cpu"), 2, 1,
          random.Random(1), stats=stats)
    assert stats["value_samples"] == 0 and stats["value_loss"] is None
    for name, parameter in model.named_parameters():
        if name.startswith("value_"):
            assert parameter.grad is None
            assert torch.equal(parameter, before[name])
    assert not torch.equal(model.policy_from.weight, before["policy_from.weight"])


def test_mixed_batch_keeps_real_draw_targets_and_masks_unfinished():
    model = PolicyValueNet(width=8, blocks=1)
    board = chess.Board()
    indices = move_indices(board, list(board.legal_moves))
    pi = np.full(len(indices), 1 / len(indices), dtype=np.float32)
    x = encode(board)
    samples = [(x, indices, pi, z) for z in (None, 0.0, 1.0, -1.0)]
    with torch.no_grad():
        _, values = model(torch.from_numpy(np.stack([x] * 3)))
        expected = torch.nn.functional.mse_loss(values, torch.tensor([0., 1., -1.])).item()
    stats = {}
    # lr=0 makes the expected loss independent of the sampled order and update.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    _, loss = learn(model, optimizer, deque(samples), torch.device("cpu"), 1, 4,
                    random.Random(1), stats=stats)
    assert np.isclose(loss, expected)
    assert stats["value_samples"] == 3 and stats["value_label_fraction"] == 0.75
    assert stats["decisive_value_samples"] == 2 and stats["draw_value_samples"] == 1


@pytest.mark.parametrize("uci_moves,expected_result,expected_targets", [
    (["f2f3", "e7e5", "g2g4", "d8h4"], "0-1", [-1., 1., -1., 1.]),
    (["g1f3", "g8f6", "f3g1", "f6g8"] * 2, "1/2-1/2", None),
])
def test_real_outcomes_including_checkmate_at_move_cap(monkeypatch, uci_moves,
                                                      expected_result, expected_targets):
    import chess_ai

    def forced_search(board, *args, **kwargs):
        move = chess.Move.from_uci(uci_moves[board.ply()])
        assert move in board.legal_moves
        return SearchResult([move], np.array([1.0]), move)
        yield

    monkeypatch.setattr(chess_ai, "search_steps", forced_search)
    info = {}
    examples, result = self_play(None, torch.device("cpu"), 1, len(uci_moves),
                                 np.random.default_rng(1), game_info=info)
    assert result == expected_result and not info["truncated"]
    assert [ex[3] for ex in examples] == (expected_targets or [0.] * len(examples))


def test_format_1_checkpoints_are_rejected_with_guidance(tmp_path):
    from chess_ai import IncompatibleCheckpointError, load_checkpoint, run, save_checkpoint
    from evaluation import load_opponent

    model = PolicyValueNet(width=8, blocks=1)
    optimizer = make_optimizer(model, 0.001)
    path = tmp_path / "ckpt" / "latest.pt"
    save_checkpoint(path, model, optimizer, 3, deque(), np.random.default_rng(1),
                    random.Random(1), 8, 1)
    data = torch.load(path, weights_only=False)
    del data["format_version"]
    torch.save(data, path)
    with pytest.raises(IncompatibleCheckpointError, match="fresh --checkpoint-dir"):
        load_checkpoint(path, model, optimizer, torch.device("cpu"), 20,
                        np.random.default_rng(), random.Random())
    with pytest.raises(IncompatibleCheckpointError):
        load_opponent(path, torch.device("cpu"))
    with pytest.raises(IncompatibleCheckpointError):
        run(run_args(tmp_path, resume=True))


def test_evaluation_balances_colors_counts_truncations_and_preserves_rng(monkeypatch):
    import evaluation

    model = PolicyValueNet(width=8, blocks=1).train()
    state = torch.get_rng_state().clone()
    calls = []
    outcomes = iter(["win", "loss", "draw", "truncated"])

    def fake_game(candidate, opponent, color, simulations, max_plies, opening, seed):
        assert not candidate.training
        calls.append((color, opening, seed))
        return {"result": next(outcomes)}
        yield

    monkeypatch.setattr(evaluation, "play_game_steps", fake_game)
    scores = evaluation.evaluate(model, None, torch.device("cpu"), 4, 1, 10, 17)
    assert calls[0] == (chess.WHITE, (), 17)
    assert calls[1] == (chess.BLACK, (), 17)
    assert calls[2][1:] == calls[3][1:]
    assert scores["wins"] == scores["losses"] == scores["draws"] == scores["truncated"] == 1
    assert scores["completed_score"] == 0.5
    assert scores["score_lower_bound"] == 0.375 and scores["score_upper_bound"] == 0.625
    assert model.training and torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError, match="even"):
        evaluation.evaluate(model, None, torch.device("cpu"), 3, 1, 10, 17)


def test_evaluation_games_run_for_real_against_random_and_a_model():
    import evaluation

    candidate, opponent = PolicyValueNet(8, 0), PolicyValueNet(8, 0)
    for other in (None, opponent):
        scores = evaluation.evaluate(candidate, other, torch.device("cpu"), 4, 2, 6, 3)
        assert scores["games"] == 4 and len(scores["game_details"]) == 4
        assert all(game["plies"] <= 6 for game in scores["game_details"])


def test_loading_snapshot_does_not_change_training_rng(tmp_path):
    from chess_ai import save_snapshot
    from evaluation import load_opponent

    model = PolicyValueNet(width=8, blocks=1)
    path = tmp_path / "snapshot.pt"
    save_snapshot(path, model, 2, 8, 1)
    state = torch.get_rng_state().clone()
    opponent = load_opponent(path, torch.device("cpu"))
    assert not opponent.training and torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(),
                                                  opponent.state_dict().values()))


@pytest.mark.parametrize("color,result", [(chess.WHITE, "loss"), (chess.BLACK, "win")])
def test_evaluation_checkmate_on_last_ply_is_completed(monkeypatch, color, result):
    import evaluation

    def mate_search(board, network, simulations, noise):
        assert noise is False
        move = board.parse_san("Qh4#")
        return SearchResult([move], np.array([1.0]), move)
        yield

    monkeypatch.setattr(evaluation, "search_steps", mate_search)
    game = evaluation.play_game(object(), object(), color, torch.device("cpu"), 1, 1,
                                ("f2f3", "e7e5", "g2g4"), 1)
    assert game["result"] == result
    assert game["termination"] == "CHECKMATE" and game["pgn_result"] == "0-1"


def test_resume_appends_metrics_and_honors_explicit_learning_rate(tmp_path):
    import json
    from chess_ai import run

    args = run_args(tmp_path, max_plies=1, batch_size=1, games=1, iterations=1,
                    replay_size=10, checkpoint_dir=str(tmp_path), blocks=0, lr=0.002)
    run(args)
    args.resume = True
    args.lr = None
    run(args)
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["iteration"] == 2 and saved["optimizer"]["param_groups"][0]["lr"] == 0.002
    args.lr = 0.003
    run(args)
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["iteration"] == 3 and saved["optimizer"]["param_groups"][0]["lr"] == 0.003
    assert saved["config"]["lr"] == 0.003
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len({row["run_id"] for row in records}) == 3
    assert [row["iteration"] for row in records if row["event"] == "training_iteration"] == [1, 2, 3]
    args.resume = False
    with pytest.raises(FileExistsError, match="resume"):
        run(args)
