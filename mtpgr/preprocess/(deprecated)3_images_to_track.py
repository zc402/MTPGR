import cv2  # unused, but must be imported, otherwise the MPT raises segment fault
from multi_person_tracker import MPT
from pathlib import Path
import torch
import pickle
from mtpgr.config import get_cfg_defaults


def to_track(image_folder: Path, save_path: Path):
    """
    Run multi-person tracker on video
    :param image_folder:
    :param save_path:
    :return:
    """

    assert image_folder.is_dir()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mot = MPT(
        device=device,
        batch_size=5,
        display=False,
        detector_type='yolo',
        output_format='dict',
    )
    tracking_results = mot(image_folder)

    with save_path.open('wb') as f:
        pickle.dump(tracking_results, f)


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    assert Path(cfg.DATA_ROOT).is_dir(), 'MTPGR/data not found. "./MTPGR" is expected as working directory'

    img_root = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.IMG_DIR
    assert img_root.is_dir(), f'{img_root.absolute()} not found.'
    track_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.TRACK_DIR
    track_folder.mkdir(exist_ok=True)
    for img_folder in img_root.glob('*'):
        track_path = track_folder / (img_folder.stem + '.pkl')
        if track_path.is_file():
            print(f'track file {track_path.absolute()} already exists')
        else:
            to_track(img_folder, track_path)
