import cv2
from pathlib import Path
import json  # My file saved in json
import json5  # Only used when loading .llc file
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.utils.log import log
"""
Example of data format in .llc file:
{
  version: 1,
  mediaFileName: '4K9A0217.m4v',
  cutSegments: [
    {
      end: 54.945968,
      name: 'F',
    },
    {
      start: 54.945968,
      end: 107.503396,
      name: 'L',
    },
    ...
  ]
}

"""

def ori_llc_to_frame(video: Path, label: Path, save_path: Path, show: bool = True):
    """
    Convert orientation labels (.llc file) to per-frame labels

    """

    cap = cv2.VideoCapture(str(video))
    with label.open() as f:
        label_json = json5.load(f)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    class_list = [0 for i in range(frame_count)]  # Init an all-zero per-frame orientation class list
    

    for segment in tqdm(label_json['cutSegments']):
        """
        Example of a segment:
        - First block
        {
            end: 54.945968,
            name: 'F',
        }
        - Middle blocks
        {
            start: 54.945968,
            end: 107.503396,
            name: 'L',
        }
        - Last block
        {
            start: 416.52772,
            name: 'F',
        }
        """
        assert segment['name'], "A block has no name field"

        start_time = None  # Timestamp of block start
        if 'start' in segment:
            start_time = segment['start']
        else:
            start_time = 0.0

        # Convert timestamp to frame number
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        start_frame = int(start_frame)
        for i in range(start_frame, frame_count):
            class_list[i] = segment['name']

    cap.release()
    if show:
        plt.plot(class_list)
        plt.show()

    with save_path.open('w') as f:
        json.dump(class_list, f)
    pass


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    assert Path(cfg.DATA_ROOT).is_dir(), 'MTPGR/data not found. Expecting "./MTPGR" as working directory'
    # Folder of per-frame label files
    target_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.ORI_LABEL_DIR
    target_folder.mkdir(exist_ok=True)
    # Folder of .llc files
    source_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.ORIENTATION_LLC_DIR
    # Corresponding videos
    video_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    video_paths = video_folder.glob('*.m4v')
    video_paths = list(video_paths)
    log.debug("video folder is:")
    log.debug(video_folder.absolute())
    log.debug("videos:")
    log.debug(video_paths)

    for video in video_paths:
        log.info(f'Now processing: "{video}"')
        # Source .llc path
        source_file = source_folder / (video.stem + '-proj.llc')
        # Target .json path
        target_file = target_folder / (video.stem + '.json')

        ori_llc_to_frame(video, source_file, target_file, show=False)
        log.info(f'Timestamp label saved to {target_file.absolute()}')
