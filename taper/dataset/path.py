from pathlib import Path

data_folder = Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/')
train_videos = data_folder.glob('*.mp4')
tracks = data_folder.glob('*.track')


def to_track_correct(video: Path):
    return video.parent / Path(video.stem).with_suffix('.track_correct')


def to_vibe_params(video: Path):
    return video.parent / Path(video.stem).with_suffix('.vibe')


def to_gesture_label(video: Path):
    return video.parent / Path(video.stem).with_suffix('.json5')