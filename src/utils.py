import pandas as pd


def get_uci(board1, board2, who_moved):
    """
    Reference: https://stackoverflow.com/questions/66770587/how-do-i-get-the-played-move-by-comparing-two-different-fens
    """
    nums = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h"}

    str_board = str(board1).split("\n")
    str_board2 = str(board2).split("\n")
    move = ""
    flip = False
    if who_moved == "w":
        for i in range(8)[::-1]:
            for x in range(15)[::-1]:
                if str_board[i][x] != str_board2[i][x]:
                    if str_board[i][x] == "." and move == "":
                        flip = True
                    move += str(nums.get(round(x / 2) + 1)) + str(9 - (i + 1))
    else:
        for i in range(8):
            for x in range(15):
                if str_board[i][x] != str_board2[i][x]:
                    if str_board[i][x] == "." and move == "":
                        flip = True
                    move += str(nums.get(round(x / 2) + 1)) + str(9 - (i + 1))
    if flip:
        move = move[2] + move[3] + move[0] + move[1]

    # Castling
    if move == "h1g1f1e1":
        move = "e1g1"
    elif move == "e8f8g8h8":
        move = "e8g8"
    elif move == "e1d1c1a1":
        move = "e1c1"
    elif move == "e8d8c8a8":
        move = "e8c8"

    return move


def get_fen_df(vbe, min_frames=5):

    fen_df = pd.DataFrame(vbe.fens.items(), columns=["frame", "fen"])

    fen_df["frame_count"] = fen_df["fen"].map(fen_df["fen"].value_counts().to_dict())
    fen_df = (
        fen_df.query("frame_count >= @min_frames")
        .drop_duplicates(subset=["fen"])
        .reset_index(drop=True)
        .copy()
    )
    return fen_df
