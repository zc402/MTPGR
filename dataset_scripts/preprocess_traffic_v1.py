from pathlib import Path
import os

from taper.preprocess import Video2Smpl

if __name__ == '__main__':
    os.chdir(Path(os.getcwd()).parent)
    data_folder = Path('.') / 'data'
    v1_train: Path = data_folder / 'PoliceGestureLong' / 'train'
    v1_test: Path = data_folder / 'PoliceGestureLong' / 'sample'
    for v1 in [v1_train, v1_test]:
        if not v1.is_dir():
            exit(f'dataset folder {v1} not found')
        video_files = v1.glob('*.mp4')
        for video in video_files:
            Video2Smpl()(video)
