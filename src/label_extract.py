import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("../tensorflow_chessbot/")
from tensorflow_chessbot import ChessboardPredictor


def line_flattener(line):
    linenew = line
    for i in range(9, 1, -1):
        linenew = linenew.replace("1" * i, str(i))
    return linenew


class VideoBoardExtractor:
    def __init__(
        self,
        video_fn,
        gt_board_loc=[20, 415, 440, 835],
        predict_fen=False,
        store_masks=False,
        store_gt_boards=False,
    ):
        self.video_fn = video_fn
        self.vidcap = None
        self.first_frame_img = None
        self.avg_frame_color = {}
        self.last_frame_color = None
        self.frame_change = None
        self.extract_board = False
        self.gt_board_loc = gt_board_loc
        self.predict_fen = predict_fen
        self.last_fen = None
        if predict_fen:
            self.fens = {}
            self.fen_predictor = ChessboardPredictor(
                frozen_graph_path="../tensorflow_chessbot/saved_models/frozen_graph.pb"
            )
        self.store_masks = store_masks
        if store_masks:
            self.masks = {}
        self.store_gt_boards = store_gt_boards
        if store_gt_boards:
            self.gt_boards = {}

    def load_videocap(self):
        self.vidcap = cv2.VideoCapture(self.video_fn)

    def process_video(self, stop_frame=-1):
        if self.vidcap is None:
            self.vidcap = self.load_videocap()

        success, image = self.vidcap.read()
        frame = 1
        pbar = tqdm(total=stop_frame)
        while success:
            success, image = self.vidcap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            succ = self.process_frame(image, frame)
            if succ is False:
                # No FEX extracted, stop running
                self.vidcap.release()
                return
            if frame == stop_frame:
                self.vidcap.release()
                return
            frame += 1
            pbar.update(1)
        pbar.close()
        self.vidcap.release()

    def get_average_color(self, img, frame):
        average = img.mean(axis=0).mean(axis=0)

        self.avg_frame_color[frame] = average

        if self.last_frame_color is None:
            self.last_frame_color = np.mean(average)
            return

        self.frame_change = np.mean(average) - self.last_frame_color
        if self.frame_change > 50:
            self.extract_board = True
            # print(f'AGHHHH - FRAME {frame}')
        self.last_frame_color = np.mean(average)

    def extract_fen(self, gt_board, frame):
        results = self.fen_predictor.makePrediction(gt_board)
        fen = results[0]
        if fen is None:
            return False, True
        flat_fen = "/".join([line_flattener(line) for line in fen.split("/")])
        self.fens[frame] = flat_fen

        if self.last_fen != flat_fen:
            changed = True
        else:
            changed = False
        self.last_fen = flat_fen
        return True, changed

    def process_frame(self, image, frame):
        if frame == 1:
            self.first_frame_img = image.copy()
        self.get_average_color(image, frame)
        self.this_frame_img = image.copy()
        self.masked_img, self.gt_board = self.extract_gt_board(
            image,
            self.gt_board_loc[0],
            self.gt_board_loc[1],
            self.gt_board_loc[2],
            self.gt_board_loc[3],
        )
        if self.extract_board:
            succeed, fen_changed = self.extract_fen(self.gt_board, frame)
            if self.store_gt_boards and fen_changed:
                self.gt_boards[frame] = self.gt_board
            if self.store_masks and fen_changed:
                self.masks[frame] = self.masked_img
            return succeed
        return True

    def extract_gt_board(self, img, top=20, bottom=415, left=440, right=835):
        gt_board = img[top:bottom, left:right, :].copy()
        if self.first_frame_img is None:
            img[top:bottom, left:right, :] = 0
            return img, gt_board
        img[top:bottom, left:right, :] = self.first_frame_img[top:bottom, left:right, :]
        return img, gt_board
