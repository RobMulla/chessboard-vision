import argparse
from label_extract import VideoBoardExtractor
from utils import get_fen_df
from fens_to_board import fens_to_board

def get_argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_id',
                        help='Youtube Video ID'
                        )
    parser.add_argument('-c', '--channel_name',
                        help='Youtube Channel Name'
                        )
    parser.add_argument('-f', '--max_frame',
                        default=-1,
                        type=int,
                        help='Max frame to extract'
                        )
    
    args = parser.parse_args()
    return args

def do_pipeline(video_id, channel_name, max_frame):

    vbe = VideoBoardExtractor(
        f"../data/{channel_name}/{video_id}.mp4",
        gt_board_loc=[10, 405, 445, 835],
        predict_fen=True,
        store_gt_boards=True,
        store_masks=True,
    )

    vbe.load_videocap()
    # vbe.process_video(30 * 60 * 1)
    vbe.process_video(max_frame)
    
    fen_df = get_fen_df(vbe, 8)
    fen_df2, board = fens_to_board(fen_df)
    fen_df2.to_csv(f"../data/labels/{video_id}.csv", index=False)
    
if __name__ == "__main__":
    args = get_argparser()
    print('*'*30)
    print(args.video_id, args.channel_name, args.max_frame)
    print('*'*30)
    do_pipeline(args.video_id, args.channel_name, args.max_frame)