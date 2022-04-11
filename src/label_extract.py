import os
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
        irl_board_loc=[420, -1, 350, -350],
        predict_fen=False,
        store_masks=False,
        store_gt_boards=False,
        store_irl_boards=False,
        store_dir="../data/processed",
        predictor_model_file="../tensorflow_chessbot/saved_models/frozen_graph.pb",
    ):
        """Object for extracting features from a chess video mp4 file.
        The video is expected to have a ground truth board placed at
        the top middle of the screen and the live in real life board
        at the bottom middle of the screen. Video should be in mp4 format.

        Args:
            video_fn (str): The video filename .mp4 format
            gt_board_loc (list, optional): The location of the ground
                truth board. Defaults to [20, 415, 440, 835].
            irl_board_loc (list, optional): The location of the real
                life board. Defaults to [420, -1, 350, -350].
            predict_fen (bool, optional): Option to predict the FEN board
                setup on frames where a ground truth board exists.
                Defaults to False.
            store_masks (bool, optional): Option to store a masked image
                of the video for each frame. Defaults to False.
            store_gt_boards (bool, optional): Option to store the ground truth
                boards image from the video. Defaults to False.
            store_irl_boards (bool, optional): Option to store the real life
                board imags. Defaults to False.
            store_dir (str, optional): The location to store the outputs of
                the processing. Will be placed in a subfolder with the video_id.
                Defaults to "../data/processed".
        """
        self.video_fn = video_fn
        self.vidcap = None
        self.first_frame_img = None
        self.avg_frame_color = {}
        self.last_frame_color = None
        self.frame_change = None
        self.at_first_gt_frame = False
        self.gt_board_loc = gt_board_loc
        self.predict_fen = predict_fen
        self.last_fen = None
        self.store_dir = store_dir
        self.video_id = video_fn.split("/")[-1].strip(".mp4")
        self.full_store_dir = f"{self.store_dir}/{self.video_id}"
        self.predictor_model_file = predictor_model_file
        if predict_fen:
            self.fens = {}
            self.fen_predictor = ChessboardPredictor(
                frozen_graph_path=self.predictor_model_file
            )
        self.store_masks = store_masks
        # if store_masks:
        # self.masks = {}
        self.store_gt_boards = store_gt_boards
        # if store_gt_boards:
        # self.gt_boards = {}
        # if store_irl_boards:
        # self.irl_boards = {}
        self.irl_board_loc = irl_board_loc
        self.store_irl_boards = store_irl_boards
        if store_irl_boards or store_gt_boards or store_masks:
            self.make_dirs()

    def load_videocap(self):
        self.vidcap = cv2.VideoCapture(self.video_fn)
        self.frame_count = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)

    def process_video(self, stop_frame=-1, stop_on_first_board=False):
        if self.vidcap is None:
            self.vidcap = self.load_videocap()

        success, image = self.vidcap.read()
        frame = 1
        if stop_frame == -1 and self.frame_count > 0:
            max_frame = self.frame_count
        else:
            max_frame = stop_frame
        pbar = tqdm(total=max_frame)
        while success:
            success, image = self.vidcap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            succ = self.process_frame(image, frame)
            if stop_on_first_board and self.at_first_gt_frame:
                self.vidcap.release()
                pbar.close()
                return
            if succ is False:
                # No FEX extracted, stop running
                self.vidcap.release()
                return
            if frame == max_frame:
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
            self.at_first_gt_frame = True
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

    def make_dirs(self):
        dirs_to_make = []
        dirs_to_make.append(f"{self.full_store_dir}/gt/")
        dirs_to_make.append(f"{self.full_store_dir}/irl/")
        dirs_to_make.append(f"{self.full_store_dir}/mask/")
        for d in dirs_to_make:
            if not os.path.exists(d):
                os.makedirs(d)

    def process_frame(self, image, frame):
        if frame == 1:
            self.first_frame_img = image.copy()
        self.get_average_color(image, frame)
        self.this_frame_img = image.copy()
        self.masked_img, self.gt_board, self.irl_board = self.extract_gt_board(
            image, self.gt_board_loc, self.irl_board_loc
        )
        if self.at_first_gt_frame and self.predict_fen:
            succeed, fen_changed = self.extract_fen(self.gt_board, frame)
            if self.store_gt_boards:
                # print("Saving:")
                # print(f"{self.full_store_dir}/gt/{self.video_id}_{frame}.png")
                gt_board = cv2.cvtColor(self.gt_board, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    f"{self.full_store_dir}/gt/{self.video_id}_{frame}.jpg", gt_board,
                )
            if self.store_masks:
                mask = cv2.cvtColor(self.masked_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    f"{self.full_store_dir}/mask/{self.video_id}_{frame}.jpg", mask,
                )
            if self.store_irl_boards:
                irl_board = cv2.cvtColor(self.irl_board, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    f"{self.full_store_dir}/irl/{self.video_id}_{frame}.jpg", irl_board,
                )

            return succeed
        return True

    def extract_gt_board(self, img, gt_board_loc, irl_board_loc):
        top, bottom, left, right = gt_board_loc
        gt_board = img[top:bottom, left:right, :].copy()
        irl_top, irl_bottom, irl_left, irl_right = irl_board_loc
        irl_board = img[irl_top:irl_bottom, irl_left:irl_right, :].copy()
        if self.first_frame_img is None:
            img[top:bottom, left:right, :] = 0
            return img, gt_board
        img[top:bottom, left:right, :] = self.first_frame_img[top:bottom, left:right, :]
        return img, gt_board, irl_board
