import json
import random
from argparse import Namespace
from collections import deque

import chess
import chess.pgn
import numpy as np
import pytest
import torch

import play
from chess_ai import (PolicyValueNet, SearchResult, encode, learn, make_optimizer, move_index,
                      move_indices, save_checkpoint)
from recording import write_pgn


@pytest.fixture(autouse=True)
def small_cpu_tests():
    torch.set_num_threads(1)
    torch.manual_seed(7)


def forced_search(board, network, device, simulations, noise=False, **kwargs):
    assert not noise
    # Human moves f3/g4, but the policy targets recommend other legal moves.
    suggested = ("e2e4", "e7e5", "b1c3", "d8h4")[board.ply()]
    legal = list(board.legal_moves)
    pi = np.array([float(move.uci() == suggested) for move in legal])
    assert pi.sum() == 1
    return SearchResult(legal, pi, legal[int(np.argmax(pi))])


def scripted_input(commands):
    iterator = iter(commands)

    def read(prompt):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    return read


def source_checkpoint(path):
    model = PolicyValueNet(width=8, blocks=0)
    optimizer = make_optimizer(model, 0.001)
    board = chess.Board()
    indices = move_indices(board, list(board.legal_moves))
    pi = np.full(len(indices), 1 / len(indices), dtype=np.float32)
    replay = deque([(encode(board), indices.copy(), pi.copy(), 0.0) for _ in range(20)])
    save_checkpoint(path, model, optimizer, 7, replay, np.random.default_rng(1),
                    random.Random(1), 8, 0)
    return model


def play_args(tmp_path, games=1):
    return Namespace(checkpoint=str(tmp_path / "source" / "latest.pt"),
                     checkpoint_dir=str(tmp_path / "human"), color="white", games=games,
                     simulations=1, max_plies=20, steps=2, batch_size=8,
                     human_fraction=0.25, human_replay_size=100, replay_size=100,
                     lr=0.0001, seed=1, threads=1, device="cpu")


def test_human_moves_are_not_imitation_targets_and_weights_stay_frozen(monkeypatch):
    monkeypatch.setattr(play, "search", forced_search)
    model = PolicyValueNet(width=8, blocks=0)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    output = []
    game = play.human_game(model, torch.device("cpu"), chess.WHITE, 1, 10,
                           input_fn=scripted_input(["garbage", "--", "legal", "help", "board", "f3", "g2g4"]),
                           output=output.append)
    assert game.completed and game.result == "0-1" and game.termination == "CHECKMATE"
    assert [move.uci() for move in game.moves] == ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert [ex[3] for ex in game.examples] == [-1., 1., -1., 1.]
    indices, pi = game.examples[0][1:3]
    assert indices[int(np.argmax(pi))] == move_index(chess.Move.from_uci("e2e4"), chess.WHITE)
    assert indices[int(np.argmax(pi))] != move_index(game.moves[0], chess.WHITE)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert sum("Illegal or unclear" in line for line in output) == 2
    assert any("Legal moves:" in line for line in output)


@pytest.mark.parametrize("commands,max_plies,termination", [
    (["f3", "quit"], 20, "ABANDONED"),
    (["f3"], 20, "ABANDONED"),  # EOF behaves like quit.
    (["f3"], 1, "MOVE_CAP"),
])
def test_unfinished_human_games_have_no_value_labels(monkeypatch, commands, max_plies, termination):
    monkeypatch.setattr(play, "search", forced_search)
    game = play.human_game(PolicyValueNet(8, 0), torch.device("cpu"), chess.WHITE, 1, max_plies,
                           input_fn=scripted_input(commands), output=lambda text: None)
    assert not game.completed and game.result == "*" and game.termination == termination
    assert all(ex[3] is None for ex in game.examples)


def test_black_human_resignation_has_correct_result_and_orientation(monkeypatch):
    monkeypatch.setattr(play, "search", forced_search)
    game = play.human_game(PolicyValueNet(8, 0), torch.device("cpu"), chess.BLACK, 1, 20,
                           input_fn=scripted_input(["resign"]), output=lambda text: None)
    assert game.result == "1-0" and game.termination == "RESIGNATION"
    assert len(game.moves) == 1 and game.examples[0][3] == 1.
    assert play.board_text(chess.Board(), chess.BLACK).splitlines()[-2] == "   h g f e d c b a"


def test_resignation_pgn_and_conflicting_result_validation(tmp_path):
    board = chess.Board()
    moves = []
    for san in ("e4", "e5"):
        move = board.parse_san(san)
        moves.append(move)
        board.push(move)
    path = tmp_path / "resigned.pgn"
    write_pgn(moves, path, result_override="0-1", termination="RESIGNATION",
              headers={"White": "Human", "Black": "Chess AI"})
    with path.open() as stream:
        game = chess.pgn.read_game(stream)
    assert game.headers["Result"] == "0-1" and game.headers["Termination"] == "RESIGNATION"
    assert game.headers["White"] == "Human" and list(game.mainline_moves()) == moves
    board, moves = chess.Board(), []
    for san in ("f3", "e5", "g4", "Qh4#"):
        move = board.parse_san(san)
        moves.append(move)
        board.push(move)
    with pytest.raises(ValueError, match="contradicts"):
        write_pgn(moves, path, result_override="1-0")


def test_mixed_sampling_limits_human_share_and_includes_latest_game():
    model = PolicyValueNet(8, 0)
    optimizer = make_optimizer(model, 0.0001)
    board = chess.Board()
    indices = move_indices(board, list(board.legal_moves))
    pi = np.full(len(indices), 1 / len(indices), dtype=np.float32)
    selfplay = deque([(encode(board), indices, pi, 0.) for _ in range(20)])
    human = deque([(encode(board), indices, pi, 1.) for _ in range(20)])
    recent = list(human)[-2:]
    stats = {}
    learn(model, optimizer, selfplay, torch.device("cpu"), 3, 8, random.Random(1),
          stats=stats, human_replay=human, human_fraction=0.25, recent_human=recent)
    assert stats["human_samples"] == 6 and stats["self_play_samples"] == 18
    assert stats["recent_human_samples"] >= 3
    assert len(selfplay) == len(human) == 20
    learn(model, optimizer, selfplay, torch.device("cpu"), 1, 8, random.Random(1),
          stats=stats, human_replay=human, human_fraction=0)
    assert stats["human_samples"] == 0 and stats["self_play_samples"] == 8


def test_completed_games_learn_persist_resume_and_survive_selfplay(monkeypatch, tmp_path):
    monkeypatch.setattr(play, "search", forced_search)
    args = play_args(tmp_path, games=2)
    original = source_checkpoint(tmp_path / "source" / "latest.pt")
    original_weights = {key: value.clone() for key, value in original.state_dict().items()}
    play.run(args, input_fn=scripted_input(["f3", "g4"] * 2), output=lambda text: None)
    checkpoint = tmp_path / "human" / "latest.pt"
    data = torch.load(checkpoint, weights_only=False)
    assert data["human_games"] == 2 and data["iteration"] == 7
    assert len(data["human_replay"]) == 8 and len(data["replay"]) == 20
    assert any(not torch.equal(original_weights[key], value) for key, value in data["network"].items())
    source = torch.load(tmp_path / "source" / "latest.pt", weights_only=False)
    assert all(torch.equal(original_weights[key], value) for key, value in source["network"].items())
    records = [json.loads(line) for line in (tmp_path / "human" / "human_metrics.jsonl").read_text().splitlines()]
    learning = [row for row in records if row["event"] == "human_training"]
    assert len(learning) == 2 and all(row["recent_human_samples"] >= args.steps for row in learning)
    assert len(list((tmp_path / "human" / "games").glob("*.pgn"))) == 2
    snapshots = sorted((tmp_path / "human" / "snapshots").rglob("after_game_*.pt"))
    first, second = [torch.load(path, weights_only=False)["network"] for path in snapshots]
    assert any(not torch.equal(first[key], second[key]) for key in first)

    # A later invocation resumes the learned branch even if the original source moved.
    args.checkpoint = str(tmp_path / "no-longer-present.pt")
    args.games = 1
    play.run(args, input_fn=scripted_input(["f3", "g4"]), output=lambda text: None)
    data = torch.load(checkpoint, weights_only=False)
    assert data["human_games"] == 3 and len(data["human_replay"]) == 12

    # Continuing ordinary self-play preserves human data and mixes it during learning.
    from chess_ai import run
    run(Namespace(simulations=1, max_plies=1, batch_size=8, steps=1, games=1,
                  iterations=1, replay_size=100, human_replay_size=100,
                  human_fraction=0.25, seed=1, device="cpu", threads=1,
                  checkpoint_dir=str(tmp_path / "human"), width=8, blocks=0,
                  lr=None, resume=True, temperature_moves=1,
                  record=False, record_video=False, eval_every=0))
    data = torch.load(checkpoint, weights_only=False)
    assert data["human_games"] == 3 and len(data["human_replay"]) == 12
    rows = [json.loads(line) for line in (tmp_path / "human" / "metrics.jsonl").read_text().splitlines()]
    training = next(row for row in rows if row["event"] == "training_iteration")
    assert training["human_samples"] == 2 and training["self_play_samples"] == 6


def test_quitting_does_not_train_or_create_learned_checkpoint(tmp_path):
    args = play_args(tmp_path)
    source_checkpoint(tmp_path / "source" / "latest.pt")
    play.run(args, input_fn=scripted_input(["quit"]), output=lambda text: None)
    assert not (tmp_path / "human" / "latest.pt").exists()
    rows = [json.loads(line) for line in (tmp_path / "human" / "human_metrics.jsonl").read_text().splitlines()]
    assert all(row["event"] != "human_training" for row in rows)
    with next((tmp_path / "human" / "games").glob("*.pgn")).open() as stream:
        game = chess.pgn.read_game(stream)
    assert game.headers["Result"] == "*" and game.headers["Termination"] == "ABANDONED"


def test_snapshot_without_selfplay_data_can_learn(monkeypatch, tmp_path):
    from chess_ai import save_snapshot

    monkeypatch.setattr(play, "search", forced_search)
    args = play_args(tmp_path)
    save_snapshot(tmp_path / "source" / "latest.pt", PolicyValueNet(8, 0), 2, 8, 0)
    play.run(args, input_fn=scripted_input(["f3", "g4"]), output=lambda text: None)
    data = torch.load(tmp_path / "human" / "latest.pt", weights_only=False)
    assert len(data["replay"]) == 0 and len(data["human_replay"]) == 4
    rows = [json.loads(line) for line in (tmp_path / "human" / "human_metrics.jsonl").read_text().splitlines()]
    training = next(row for row in rows if row["event"] == "human_training")
    assert training["self_play_samples"] == 0 and training["human_samples"] == 8


def test_default_commands_update_one_shared_model(monkeypatch, tmp_path):
    import builtins
    import sys
    import chess_ai

    monkeypatch.chdir(tmp_path)
    source_checkpoint(tmp_path / "checkpoints" / "latest.pt")
    monkeypatch.setattr(play, "search", forced_search)
    monkeypatch.setattr(builtins, "input", scripted_input(["f3", "g4"]))
    monkeypatch.setattr(sys, "argv", ["play.py", "--simulations", "1", "--steps", "1",
                                     "--batch-size", "8", "--threads", "1"])
    play.main()
    shared = tmp_path / "checkpoints" / "latest.pt"
    after_human = torch.load(shared, weights_only=False)
    assert after_human["human_games"] == 1 and len(after_human["human_replay"]) == 4
    assert not (tmp_path / "checkpoints" / "human-play").exists()

    monkeypatch.setattr(sys, "argv", ["chess_ai.py", "--resume", "--iterations", "1",
                                     "--games", "1", "--simulations", "1", "--max-plies", "1",
                                     "--steps", "1", "--batch-size", "8", "--threads", "1",
                                     "--eval-every", "0"])
    chess_ai.main()
    after_selfplay = torch.load(shared, weights_only=False)
    assert after_selfplay["human_games"] == 1 and len(after_selfplay["human_replay"]) == 4
    assert after_selfplay["iteration"] == after_human["iteration"] + 1
    rows = [json.loads(line) for line in (tmp_path / "checkpoints" / "metrics.jsonl").read_text().splitlines()]
    initial_snapshot = next(row["initial_snapshot"] for row in rows if row["event"] == "run_start")
    resumed_weights = torch.load(initial_snapshot, weights_only=False)["network"]
    assert all(torch.equal(resumed_weights[key], value) for key, value in after_human["network"].items())
    training = next(row for row in rows if row["event"] == "training_iteration")
    assert training["human_samples"] > 0 and training["self_play_samples"] > 0


def test_play_and_selfplay_cannot_write_same_checkpoint_concurrently(tmp_path):
    import chess_ai
    from checkpoint_access import checkpoint_lock

    args = Namespace(checkpoint_dir=str(tmp_path))
    with checkpoint_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another training or play session"):
            play.run(args)
        with pytest.raises(RuntimeError, match="Another training or play session"):
            chess_ai.run(args)
    # The persistent lock file does not prevent a later session from acquiring it.
    with checkpoint_lock(tmp_path):
        pass


def test_current_game_is_covered_before_positions_repeat():
    model = PolicyValueNet(8, 0)
    optimizer = make_optimizer(model, 0.0001)
    board = chess.Board()
    indices = move_indices(board, list(board.legal_moves))
    pi = np.full(len(indices), 1 / len(indices), dtype=np.float32)
    selfplay = deque([(encode(board), indices, pi, 0.) for _ in range(40)])
    recent = [(encode(board), indices, pi, 1.) for _ in range(11)]
    stats = {}
    learn(model, optimizer, selfplay, torch.device("cpu"), 2, 32, random.Random(1),
          stats=stats, human_replay=deque(recent), human_fraction=0.25, recent_human=recent)
    assert stats["recent_unique_positions"] == stats["recent_game_positions"] == 11
    assert stats["human_samples"] == 16 and stats["self_play_samples"] == 48


def test_default_post_game_learning_covers_the_game_about_once(monkeypatch, tmp_path):
    monkeypatch.setattr(play, "search", forced_search)
    args = play_args(tmp_path)
    args.steps = None
    source_checkpoint(tmp_path / "source" / "latest.pt")
    play.run(args, input_fn=scripted_input(["f3", "g4"]), output=lambda text: None)
    rows = [json.loads(line) for line in (tmp_path / "human" / "human_metrics.jsonl").read_text().splitlines()]
    training = next(row for row in rows if row["event"] == "human_training")
    # Two human slots per batch of 8, so two updates train each of the 4 positions once.
    assert training["recent_unique_positions"] == training["recent_game_positions"] == 4
    assert training["human_samples"] == 4 and training["self_play_samples"] == 12
    data = torch.load(tmp_path / "human" / "latest.pt", weights_only=False)
    assert (tmp_path / "human" / data["origin_snapshot"]).is_file()
