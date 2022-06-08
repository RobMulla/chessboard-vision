import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from label_extract import VideoBoardExtractor


def process_ls_labels(json_fn="../labels/result.json"):
    with open(json_fn) as f:
        j = json.load(f)

    annotations = pd.json_normalize(j["annotations"])
    imgs = pd.json_normalize(j["images"])
    results = annotations.merge(
        imgs, left_on=["image_id"], right_on=["id"], how="left", suffixes=("", "_image")
    )
    cats = pd.json_normalize(j["categories"])
    results = results.merge(
        cats,
        left_on=["category_id"],
        right_on=["id"],
        how="left",
        suffixes=("", "_cat"),
    )
    results["x"] = results["bbox"].str[0]
    results["y"] = results["bbox"].str[1]
    results["w"] = results["bbox"].str[2]
    results["h"] = results["bbox"].str[3]
    results["png"] = results["file_name"].str[-25:]
    results["video_name"] = results["file_name"].str[-25:-14]
    results = results.drop([0, 1]).reset_index(drop=True)
    return results


if __name__ == "__main__":
    label = process_ls_labels()
    for v in label["video_name"].unique():
        print(f"===== RUNNING FOR VIDEO {v} ===== ")

        x = label.query("video_name == @v and category_id == 0")["x"].values[0]
        y = label.query("video_name == @v and category_id == 0")["y"].values[0]
        h = label.query("video_name == @v and category_id == 0")["h"].values[0]
        w = label.query("video_name == @v and category_id == 0")["w"].values[0]

        x2 = x + w
        y2 = y + h

        irl_x = label.query("video_name == @v and category_id == 1")["x"].values[0]
        irl_y = label.query("video_name == @v and category_id == 1")["y"].values[0]
        irl_h = label.query("video_name == @v and category_id == 1")["h"].values[0]
        irl_w = label.query("video_name == @v and category_id == 1")["w"].values[0]

        if irl_w % 2 == 1:
            irl_w += 1
        if irl_h % 2 == 1:
            irl_h += 1

        irl_x2 = irl_x + irl_w
        irl_y2 = irl_y + irl_h

        vbe = VideoBoardExtractor(
            video_fn=f"../data/CoffeeChess/{v}.mp4",
            at_first_gt_frame=True,
            predict_fen=True,
            store_gt_boards=False,
            store_irl_boards=False,
            store_irl_video=True,
            save_img_freq=None,
            gt_board_loc=[y, y2, x, x2],
            irl_board_loc=[irl_y, irl_y2, irl_x, irl_x2],
        )
        vbe.load_videocap()
        vbe.process_video(start_frame=0, stop_frame=-1)
