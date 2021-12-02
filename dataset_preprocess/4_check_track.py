"""
The police is not always tracked correctly. Wrong tracks needs to be fixed manually.
"""
import pickle
from pathlib import Path
import logging

import numpy as np

import dataset_preprocess.path as data_path
log = logging.getLogger(__name__)
logging.root.setLevel(logging.ERROR)


def load_tracking_results(track_file: Path) -> list:
    with track_file.open('rb') as f:
        track = pickle.load(f)
    # This converts dict: {pid: res} into list: (pid, res). res: {'bbox': ndarray, 'frames': ndarray}
    track_sort = sorted(track.items(), key=lambda x: len(x[1]['frames']), reverse=True)
    return track_sort


def is_tracked_every_frame(track_file: Path):
    # Check if tracked frames equal total frames. This indicates a correct tracking result.

    log.info(f"Checking {track_file}")
    track_sort = load_tracking_results(track_file)
    tracked_len = len(track_sort[0][1]['frames'])

    image_folder = track_file.parent / (track_file.stem + '.images')
    img_num = len(list(image_folder.glob('*.jpg')))

    if img_num == tracked_len:
        log.info('Tracked frames equals to the num of images')
    else:
        log.error(f'Incorrect track:{track_file}, {img_num} images, {tracked_len} tracked frames')


def non_maximum_suppression_1d(track_file: Path):
    def overlap(t1, t2):  # t1: (412,{'bbox':ndarray, 'frames':ndarray}
        before, after = sorted((t1, t2), key=lambda x: x[1]['frames'][0])  # by first frame_num
        return before[1]['frames'][-1] >= after[1]['frames'][0]

    track_sort = load_tracking_results(track_file)
    id_longest = 0
    while True:
        if id_longest == len(track_sort):
            break  # nms complete
        longest = track_sort[id_longest]
        for i in range(len(track_sort)-1, id_longest, -1):  # Reverse to properly delete from list
            # If overlap
            is_overlap = overlap(longest, track_sort[i])
            if is_overlap:
                track_sort.pop(i)
        id_longest = id_longest + 1
    return track_sort



def concat_long_tracks(track_file: Path):
    # For interrupted tracks, concat all no-overlapping tracks into same track.
    track_sort = non_maximum_suppression_1d(track_file)
    track_time_order = sorted(track_sort, key=lambda x: x[1]['frames'][0])
    new_frames = np.concatenate([x[1]['frames'] for x in track_time_order])
    new_bbox = np.concatenate([x[1]['bbox'] for x in track_time_order])
    new_dict = {1: {'bbox': new_bbox, 'frames': new_frames}}
    save_path = track_file.with_suffix('.track_correct')
    with save_path.open('wb') as f:
        pickle.dump(new_dict, f)

    lost = list(range(9000))
    [lost.remove(x) for x in new_frames.tolist()]

    log.info(f'{track_file.name} no detection:' + str(lost))

    pass

concat_long_tracks(Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/4K9A0227.track'))
# non_maximum_suppression_1d(Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/4K9A0227.track'))

# for track in data_path.tracks:
#     is_tracked_every_frame(track)
