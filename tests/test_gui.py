import json
import threading
from argparse import Namespace

import chess
import numpy as np
import pytest
import torch

from gui import BoardInteraction, BoardLayout, WindowBridge


@pytest.mark.parametrize("flipped", [False, True])
def test_board_geometry_roundtrips_all_squares_after_resize(flipped):
    for width, height in ((650, 650), (980, 600), (450, 710)):
        layout = BoardLayout.fit(width, height, flipped)
        for square in chess.SQUARES:
            assert layout.square_at(*layout.center(square)) == square
        assert layout.square_at(layout.left - 1, layout.top) is None
        assert layout.square_at(layout.left + 8 * layout.cell, layout.top) is None
    layout = BoardLayout.fit(650, 650, flipped)
    assert layout.square_at(layout.left + 1, layout.top + 1) == (chess.H1 if flipped else chess.A8)


def test_click_piece_then_legal_destination_and_deselect():
    control = BoardInteraction(chess.Board(), chess.WHITE)
    assert control.click(chess.E2) == []
    assert control.destinations == {chess.E3, chess.E4}
    assert control.click(chess.E5) == []  # Illegal destination keeps selection.
    assert control.click(chess.E4) == [chess.Move.from_uci("e2e4")]
    assert control.selected is None
    control.click(chess.G1)
    assert control.destinations == {chess.F3, chess.H3}
    control.click(chess.G1)
    assert not control.destinations
    control.click(chess.B1)
    control.click(None)
    assert control.selected is None
    control.board.push_uci("e2e4")
    assert control.click(chess.G1) == [] and control.selected is None  # Model's turn.


def test_click_controller_supports_castling_en_passant_and_pins():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    control = BoardInteraction(board, chess.WHITE)
    control.click(chess.E1)
    castle = control.click(chess.G1)
    assert len(castle) == 1 and board.is_castling(castle[0])
    board = chess.Board()
    for san in ("e4", "a6", "e5", "d5"):
        board.push_san(san)
    control = BoardInteraction(board, chess.WHITE)
    control.click(chess.E5)
    capture = control.click(chess.D6)
    assert len(capture) == 1 and board.is_en_passant(capture[0])
    control = BoardInteraction(chess.Board("k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"), chess.WHITE)
    control.click(chess.E2)
    assert control.click(chess.F2) == []
    assert control.click(chess.E3) == [chess.Move.from_uci("e2e3")]


@pytest.mark.parametrize("black", [False, True])
def test_promotion_offers_all_four_pieces(black):
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    if black:
        board = board.mirror()
    control = BoardInteraction(board, board.turn)
    control.click(chess.A2 if black else chess.A7)
    moves = control.click(chess.A1 if black else chess.A8)
    assert {move.promotion for move in moves} == {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}


def test_bridge_passes_board_copies_and_cancels_waiting_input():
    bridge = WindowBridge()
    board = chess.Board()
    bridge.position(board, chess.WHITE, "Your move")
    board.push_uci("e2e4")
    event, (snapshot, color, status) = bridge.events.get(timeout=1)
    assert event == "position" and snapshot.ply() == 0
    answers = []
    thread = threading.Thread(target=lambda: answers.append(bridge.input("Your move")), daemon=True)
    thread.start()
    assert bridge.events.get(timeout=1)[0] == "awaiting"
    bridge.cancel()
    thread.join(timeout=2)
    assert not thread.is_alive() and answers == ["quit"]


def test_window_click_bridge_uses_existing_training_pipeline(monkeypatch, tmp_path):
    import play
    from chess_ai import PolicyValueNet, SearchResult, save_snapshot

    torch.set_num_threads(1)
    source = tmp_path / "source.pt"
    save_snapshot(source, PolicyValueNet(8, 0), 0, 8, 0)

    def scripted_search(board, *args, **kwargs):
        legal = list(board.legal_moves)
        suggested = ("e2e4", "e7e5", "b1c3", "d8h4")[board.ply()]
        pi = np.array([float(move.uci() == suggested) for move in legal])
        return SearchResult(legal, pi, legal[int(np.argmax(pi))])

    monkeypatch.setattr(play, "search", scripted_search)
    args = Namespace(checkpoint=str(source), checkpoint_dir=str(tmp_path / "shared"),
                     color="white", games=1, simulations=1, max_plies=20, steps=1,
                     reanalyze_simulations=1, tactical_checks=0, tactical_nodes=1,
                     batch_size=8, human_fraction=.25, human_replay_size=100,
                     replay_size=100, lr=.0001, seed=1, threads=1, device="cpu")
    bridge, errors = WindowBridge(), []

    def work():
        try:
            play.run(args, input_fn=bridge.input, output=bridge.output, on_position=bridge.position,
                     cancel_requested=bridge.cancelled.is_set)
        except Exception as error:
            errors.append(error)
        finally:
            bridge.events.put(("finished", None))

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    control, completed = None, False
    try:
        while True:
            event, value = bridge.events.get(timeout=15)
            if event == "position":
                board, color, status = value
                control = BoardInteraction(board, color)
                completed |= status.startswith("Result: 0-1")
            elif event == "awaiting":
                start, end = (chess.F2, chess.F3) if control.board.ply() == 0 else (chess.G2, chess.G4)
                control.click(start)
                moves = control.click(end)
                assert len(moves) == 1
                bridge.submit(moves[0].uci())
            elif event == "finished":
                break
    finally:
        bridge.cancel()
        worker.join(timeout=2)
    assert not errors and not worker.is_alive() and completed
    checkpoint = torch.load(tmp_path / "shared" / "latest.pt", weights_only=False)
    assert checkpoint["human_games"] == 1 and len(checkpoint["human_replay"]) == 4
    rows = [json.loads(line) for line in (tmp_path / "shared" / "human_metrics.jsonl").read_text().splitlines()]
    assert any(row["event"] == "human_training" and row["recent_unique_positions"] == 4 for row in rows)


def test_gui_flag_keeps_shared_checkpoint_default():
    import play
    args = play.build_parser().parse_args(["--gui"])
    assert args.gui and args.checkpoint_dir == "checkpoints"
