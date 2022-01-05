"""
Find the track of police with non-maximum suppression
The police is not always tracked correctly. Wrong tracks needs to be fixed manually.
"""
import pickle
from pathlib import Path
import logging
import numpy as np
from taper.config import get_cfg_defaults


log = logging.getLogger('check_track')
log.setLevel(logging.INFO)


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


def find_concat_police_tracks(track_mul: Path, save_path: Path):
    # For interrupted tracks, concat all no-overlapping tracks into same track.
    track_sort = non_maximum_suppression_1d(track_mul)
    track_time_order = sorted(track_sort, key=lambda x: x[1]['frames'][0])
    new_frames = np.concatenate([x[1]['frames'] for x in track_time_order])
    new_bbox = np.concatenate([x[1]['bbox'] for x in track_time_order])
    police_dict = {'bbox': new_bbox, 'frames': new_frames}

    with save_path.open('wb') as f:
        pickle.dump(police_dict, f)


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    assert Path(cfg.DATA_ROOT).is_dir(), 'TAPER/data not found, check working directory (./TAPER expected) '

    track_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.TRACK_DIR
    track_folder.mkdir(exist_ok=True)
    tracks = track_folder.glob('*')

    track_crct_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.TK_CRCT_DIR
    for track in tracks:
        save_path = track_crct_folder / track.stem
        find_concat_police_tracks(track, save_path)
        print(f'track saved into {save_path.absolute()}')
# non_maximum_suppression_1d(Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/4K9A0227.track'))

# for track in data_path.tracks:
#     is_tracked_every_frame(track)
