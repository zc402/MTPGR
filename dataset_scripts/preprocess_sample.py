# For testing usage

from pathlib import Path
import os
import torch

from taper.preprocess import Video2Smpl

if __name__ == '__main__':
    os.chdir(Path(os.getcwd()).parent)  # chdir to TAPER root, because VIBE model path is hardcoded.
    data_folder = Path('.') / 'data' / 'sample'
    video_files = data_folder.glob('*.mp4')
    for video in video_files:
        v2m = Video2Smpl()
        v2m.display_render = True
        v2m.display_trace = False
        v2m(video)
