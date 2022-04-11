import chess
from utils import get_uci


def flip_fen(fen):
    """
    Flip the fen if black is on the top of the board.
    """
    fields = fen.split(" ")
    fields[0] = fields[0][::-1]
    flipped_fen = " ".join(fields)
    return flipped_fen


def fens_to_board(fen_df):
    # Determine if white or black is on top
    top_left_piece = fen_df["fen"].str[0].values[0]
    if top_left_piece == "R":
        # Black is on top so flip all the FENs
        fen_df["fen"] = fen_df["fen"].apply(flip_fen)
    last_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    fen_df["confirmed_move"] = False
    board = chess.Board(last_fen)
    board.set_castling_fen("KQkq")
    move_count = 0
    for i, d in fen_df.iterrows():
        fen = d["fen"]
        frame = d["frame"]
        if last_fen == fen:
            continue
        new_board = chess.Board(fen)
        if move_count % 2 == 0:
            who_moved = "w"
            not_moved = "b"
        else:
            who_moved = "b"
            not_moved = "w"

        move = get_uci(board, new_board, who_moved)
        try:
            # if True:
            board.push_san(move)
            last_fen = fen
            move_count += 1
        except Exception as e:
            try:
                move = get_uci(board, new_board, not_moved)
                board.push_san(move)
                last_fen = fen
                move_count += 1
            except Exception as e:
                print(f"failed move {move_count} with exception {e} frame {frame}")
                print(f"Board before:\n{board}")
                print(f"Board next:\n{new_board}")
                print(f"Mode {move}")
                return fen_df, board
        fen_df.loc[i, "confirmed_move"] = True
        fen_df.loc[i, "move_uci"] = move
    return fen_df, board
