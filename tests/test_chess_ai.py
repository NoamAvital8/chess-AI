import random
from collections import deque

import chess
import numpy as np
import torch

from chess_ai import (ACTION_SIZE, INPUT_PLANES, PolicyValueNet, choose_move,
                      encode, learn, move_index, search, self_play, terminal_value)


def test_features_and_move_index():
    board = chess.Board()
    x = encode(board)
    assert x.shape == (INPUT_PLANES, 8, 8)
    assert x.dtype == np.float32
    assert x[:12].sum() == 32
    assert x[12, 0, 0] == 1
    assert np.all(x[13:17] == 1)
    assert move_index(chess.Move.from_uci('e7e8q')) != move_index(chess.Move.from_uci('e7e8n'))
    assert 0 <= move_index(chess.Move.from_uci('e2e4')) < ACTION_SIZE
    board.push_san('e4')
    assert encode(board)[12, 0, 0] == 0


def test_search_is_legal_and_normalized():
    torch.set_num_threads(1)
    model = PolicyValueNet(width=8, blocks=1)
    board = chess.Board()
    moves, pi = search(board, model, torch.device('cpu'), simulations=3)
    assert set(moves) == set(board.legal_moves)
    assert np.isclose(pi.sum(), 1)
    assert np.count_nonzero(pi) <= 3
    assert choose_move(moves, pi, 0, np.random.default_rng(5)) in board.legal_moves


def test_terminal_score_sign_and_draw():
    board = chess.Board()
    for move in ('f3', 'e5', 'g4', 'Qh4#'):
        board.push_san(move)
    assert terminal_value(board) == -1.0  # White is to move, Black won
    assert terminal_value(chess.Board()) is None
    assert terminal_value(chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')) == 0.0


def test_self_play_train_and_resume(tmp_path):
    from chess_ai import save_checkpoint, load_checkpoint
    torch.set_num_threads(1)
    model = PolicyValueNet(width=8, blocks=1)
    device = torch.device('cpu')
    rng = np.random.default_rng(11)
    sampler = random.Random(11)
    examples, result = self_play(model, device, simulations=1, max_plies=4,
                                 temperature_moves=2, rng=rng)
    assert len(examples) == 4
    assert result == '1/2-1/2 (move cap)'
    assert all(np.isclose(pi.sum(), 1) and z == 0 for _, _, pi, z in examples)
    buffer = deque(examples, maxlen=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    p, v = learn(model, optimizer, buffer, device, steps=1, batch_size=2, sampler=sampler)
    assert np.isfinite(p) and np.isfinite(v)
    path = tmp_path / 'latest.pt'
    save_checkpoint(path, model, optimizer, 1, buffer, rng, sampler, 8, 1)
    clone = PolicyValueNet(width=8, blocks=1)
    copy_opt = torch.optim.Adam(clone.parameters(), lr=0.001)
    iteration, restored = load_checkpoint(path, clone, copy_opt, device, 16,
                                           np.random.default_rng(), random.Random())
    assert iteration == 1 and len(restored) == 4
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters()))