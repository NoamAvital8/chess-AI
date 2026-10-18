from collections import deque

import chess
import numpy as np
import torch

from chess_ai import ACTION_SIZE, encode, move_index, move_indices
from play import PlayedGame, analyze_position, reanalyze_game, refresh_matching_policies
from tactics import tactical_moves


TRAP = ("e4", "d6", "Bc4", "b6", "Qh5", "Bb7", "Bxf7+", "Kd7", "Be6+", "Kxe6", "Qf5#")


def trap_board(plies):
    board = chess.Board()
    for san in TRAP[:plies]:
        board.push_san(san)
    return board


class BadPriorNet(torch.nn.Module):
    def forward(self, states):
        logits = torch.zeros((len(states), ACTION_SIZE), device=states.device)
        logits[:, move_index(chess.Move.from_uci("c8b7"), chess.BLACK)] = 100.
        return logits, torch.zeros(len(states), device=states.device)


def test_repeated_human_trap_is_proved_losing_without_mutating_history():
    board = trap_board(5)
    fen, history = board.fen(), board.move_stack.copy()
    allowed = tactical_moves(board, checks=3, nodes_per_move=300)
    assert board.parse_san("Bb7") not in allowed
    assert board.parse_san("g6") in allowed
    assert board.fen() == fen and board.move_stack == history


def test_exhausted_proof_is_unknown_and_all_losing_moves_remain_legal():
    board = trap_board(5)
    assert board.parse_san("Bb7") in tactical_moves(board, checks=3, nodes_per_move=1)
    doomed = trap_board(9)
    assert tactical_moves(doomed) == set(doomed.legal_moves)
    assert tactical_moves(board, checks=0) == set(board.legal_moves)


def test_forced_mate_is_found_despite_arbitrarily_bad_policy_prior():
    torch.set_num_threads(1)
    model = BadPriorNet()
    board = trap_board(5)
    result = analyze_position(board, model, torch.device("cpu"), 2, 3, 300)
    assert set(result.moves) == set(board.legal_moves)
    assert result.policy[result.moves.index(board.parse_san("Bb7"))] == 0
    assert np.isclose(result.policy.sum(), 1) and result.move != board.parse_san("Bb7")
    winning = trap_board(10)
    winning.push(analyze_position(winning, model, torch.device("cpu"), 1, 3, 300).move)
    assert winning.is_checkmate()


def test_postgame_reanalysis_replaces_losing_targets_in_older_replay():
    torch.set_num_threads(1)
    board, moves, old = chess.Board(), [], []
    for san in TRAP:
        move = board.parse_san(san)
        legal = list(board.legal_moves)
        pi = np.array([float(m == move) for m in legal], dtype=np.float32)
        old.append((encode(board), move_indices(board, legal), pi,
                    1. if board.turn else -1.))
        moves.append(move)
        board.push(move)
    game = PlayedGame(moves, old, "1-0", "CHECKMATE", chess.WHITE)
    refreshed = reanalyze_game(game, BadPriorNet(), torch.device("cpu"), 2, 3, 300)
    bad_index = move_index(chess.Move.from_uci("c8b7"), chess.BLACK)
    root_indices = list(refreshed[5][1])
    assert old[5][2][root_indices.index(bad_index)] == 1
    assert refreshed[5][2][root_indices.index(bad_index)] == 0
    historical = deque(old + [(old[5][0], old[5][1], old[5][2], 1.)])
    assert refresh_matching_policies(historical, refreshed) == 12
    assert historical[-1][3] == 1.  # Refresh policy, retain actual observed outcomes.
    assert historical[5][2][root_indices.index(bad_index)] == 0
