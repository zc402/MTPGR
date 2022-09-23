import datetime
from pathlib import Path
from mtpgr.config import get_cfg_defaults
import cv2

cfg = get_cfg_defaults()

videos = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
videos = videos.glob('*.m4v')

result = []
for video in videos:
    cap = cv2.VideoCapture(str(video))

    # count the number of frames
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # calculate duration of the video
    seconds = round(frames / fps)
    # video_time = datetime.timedelta(seconds=seconds)
    result.append((video.stem, seconds))
print(result)
print(f'Sum:{sum(list(zip(*result))[1])}')
print(f'Max:{max(result, key=lambda x:x[1])}')
print(f'Min:{min(result, key=lambda x:x[1])}')