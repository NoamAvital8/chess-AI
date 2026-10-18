"""PGN and Lichess replay links, plus optional MP4 recording of self-play."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import chess
import chess.pgn

BOARD_LEFT = 64
BOARD_TOP = 76
SQUARE = 80
FRAME_W = 768
FRAME_H = 832


def write_pgn(moves: Sequence[chess.Move], path: Path) -> str:
    """Save exact played moves; a training move cap is NOT an official draw."""
    path.parent.mkdir(parents=True, exist_ok=True)
    game = chess.pgn.Game()
    game.headers["Event"] = "Chess AI self-play"
    game.headers["White"] = "Shared neural network"
    game.headers["Black"] = "Shared neural network"
    board = chess.Board()
    node = game
    for move in moves:
        if move not in board.legal_moves:
            raise ValueError(f"Illegal recorded move: {move.uci()} in {board.fen()}")
        node = node.add_variation(move)
        board.push(move)
    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    game.headers["Result"] = result
    if result == "*":
        game.headers["Termination"] = "Unfinished (training move cap)"
    with path.open("w", encoding="utf-8") as stream:
        print(game, file=stream, end="\n\n")
    return result


def lichess_url(moves: Sequence[chess.Move]) -> str:
    """A Lichess analysis URL embedding played SAN moves, without uploading."""
    if not moves:
        raise ValueError("Cannot replay an empty game")
    san = chess.Board().variation_san(moves)
    return "https://lichess.org/analysis/pgn/" + quote(san.replace(" ", "_"), safe="._")


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    choices = (
        ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for name in choices:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _frame(board: chess.Board, last_move: chess.Move | None, ply: int,
           last_san: str, result: str):
    from PIL import Image, ImageDraw

    picture = Image.new("RGB", (FRAME_W, FRAME_H), "#101827")
    draw = ImageDraw.Draw(picture)
    title_font = _font(24, bold=True)
    piece_font = _font(37, bold=True)
    label_font = _font(15)
    info_font = _font(20)
    draw.text((64, 24), "Chess AI  |  self-play", font=title_font, fill="#ecf2fa")

    for row in range(8):
        rank = 7 - row
        for col in range(8):
            square = chess.square(col, rank)
            x, y = BOARD_LEFT + col * SQUARE, BOARD_TOP + row * SQUARE
            color = "#e9d5b4" if (row + col) % 2 == 0 else "#779d82"
            if last_move is not None and square in (last_move.from_square, last_move.to_square):
                color = "#dbc66b" if (row + col) % 2 == 0 else "#b6a044"
            draw.rectangle((x, y, x + SQUARE - 1, y + SQUARE - 1), fill=color)
            piece = board.piece_at(square)
            if piece is not None:
                white = piece.color == chess.WHITE
                draw.ellipse((x + 11, y + 11, x + 69, y + 69),
                             fill="#f9f5e8" if white else "#1f2937",
                             outline="#253342" if white else "#e8e9ea", width=3)
                draw.text((x + SQUARE / 2, y + SQUARE / 2 - 2), piece.symbol().upper(),
                          anchor="mm", font=piece_font,
                          fill="#253342" if white else "#f9f5e8")
        draw.text((38, BOARD_TOP + row * SQUARE + 32), str(rank + 1),
                  font=label_font, fill="#d0dae8")
    for col in range(8):
        draw.text((BOARD_LEFT + col * SQUARE + 36, BOARD_TOP + 8 * SQUARE + 7),
                  chr(ord("a") + col), font=label_font, fill="#d0dae8")
    draw.text((64, 750), f"Ply: {ply}    Last move: {last_san or 'Starting position'}",
              font=info_font, fill="#ecf2fa")
    draw.text((64, 786), f"To move: {'White' if board.turn else 'Black'}    Result: {result}",
              font=label_font, fill="#c3cfdd")
    return picture


def write_mp4(moves: Sequence[chess.Move], path: Path, fps: float = 2.0) -> None:
    """Save initial board and each actual subsequent position to MP4."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    try:
        import imageio.v2 as imageio
        import numpy as np
        import imageio_ffmpeg  # noqa: F401; ensures optional encoder package is installed
        from PIL import Image  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Video recording needs optional dependencies: "
            "python -m pip install -r requirements-video.txt"
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    board = chess.Board()
    final = board.copy()
    for move in moves:
        if move not in final.legal_moves:
            raise ValueError(f"Illegal recorded move: {move.uci()}")
        final.push(move)
    result = final.result(claim_draw=True) if final.is_game_over(claim_draw=True) else "* (move cap)"
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7,
                            macro_block_size=16, output_params=["-pix_fmt", "yuv420p"]) as writer:
        writer.append_data(np.asarray(_frame(board, None, 0, "", result)))
        for ply, move in enumerate(moves, 1):
            san = board.san(move)
            prefix = str(board.fullmove_number) + ("." if board.turn else "...")
            board.push(move)
            writer.append_data(np.asarray(_frame(board, move, ply, prefix + san, result)))
