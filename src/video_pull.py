from pytube import YouTube
import argparse
import shutil
import os
from tqdm import tqdm


def get_argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_id", help="Youtube Video ID")
    parser.add_argument("-s", "--save_dir", help="Directory to save")

    args = parser.parse_args()
    return args


def pull_video(video_id, save_dir):
    out_fn = f"{save_dir}/{video_id}.mp4"
    if os.path.exists(out_fn):
        return

    stream = YouTube(
        f"http://www.youtube.com/watch?v={video_id}"
    ).streams.get_highest_resolution()
    stream.download()
    fn = stream.default_filename
    shutil.move(fn, out_fn)


def pull_channel(channel_url, save_dir, max_videos=10):
    """
    Pull all videos from a channel. For example:
    'https://www.youtube.com/c/CoffeeChess'
    """
    c = Channel(channel_url)

    for i, video in enumerate(tqdm(c.videos)):
        pull_video(video.video_id, save_dir)
        if i > max_videos:
            return


if __name__ == "__main__":

    args = get_argparser()
    pull_video(args.video_id, args.save_dir)

