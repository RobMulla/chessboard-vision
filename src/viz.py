import matplotlib.pylab as plt
import chess
from chess import svg
from cairosvg import svg2png
import cv2
import numpy as np

def plot_frame_results(vbe, frame):
    # Looping through frames where the predicted
    # FEN position has changed
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Show the image with the GT board removed
    axs[0].imshow(vbe.masks[frame])
    # Actual GT board pulled from the video
    axs[1].imshow(vbe.gt_boards[frame])
    fen = vbe.fens[frame]
    fig.suptitle(f"Frame {frame}\n{fen}", fontsize=20, y=1.15)
    # Display the predicted FEN
    board = chess.Board(vbe.fens[frame])
    boardsvg = chess.svg.board(board=board)
    boardpng = svg2png(boardsvg)
    fen_png = cv2.imdecode(np.frombuffer(boardpng, np.uint8), -1)
    fen_png = cv2.cvtColor(fen_png, cv2.COLOR_BGR2RGB)
    axs[2].imshow(fen_png)
    axs[0].axis("off")
    axs[1].axis("off")
    axs[2].axis("off")
    axs[0].set_title("masked image")
    axs[1].set_title("gt board from video")
    axs[2].set_title("predicted position")
    plt.tight_layout()
    plt.show()