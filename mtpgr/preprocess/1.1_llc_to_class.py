import cv2
from pathlib import Path
import json
import json5
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtpgr.config.defaults import get_cfg_defaults


def llc_to_class(video: Path, label: Path, save_path: Path, show: bool = True):
    """
    Convert losslesscut .llc format to class-per-frame format (json)
    :param video: path of a videos, used to get FRAME_COUNT
    :param label: path of .llc file corresponding to the video.
    :param save_path: path of output json per-frame label file
    :param show: display labels as a plot
    :return:
    """
    cap = cv2.VideoCapture(str(video))
    with label.open() as f:
        label_json = json5.load(f)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
    cls_list = [0 for i in range(frame_count)]
    for i, seg in enumerate(tqdm(label_json['cutSegments'])):
        # seg: {'start': 1.0113, 'end': 3.4433, 'name': '1'}
        cls = seg['name']
        cls = int(cls)
        assert cls >= 1 and cls <= 8
        cap.set(cv2.CAP_PROP_POS_MSEC, seg['start'] * 1000)
        s_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_MSEC, seg['end'] * 1000)
        e_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        for f in range(int(s_frame), int(e_frame)):
            cls_list[f] = cls
    cap.release()

    if show:
        plt.plot(cls_list)
        plt.show()

    with save_path.open('w') as f:
        json.dump(cls_list, f)
    pass


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    assert Path(cfg.DATA_ROOT).is_dir(), 'MTPGR/data not found. Expecting "./MTPGR" as working directory'
    (Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.GES_LABEL_DIR).mkdir(exist_ok=True)
    videos = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    videos = videos.glob('*.m4v')
    for video in videos:
        print(f'Processing: "{video}"')
        llc = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.GESTURE_LLC_DIR / (video.stem + '-proj.llc')
        output = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.GES_LABEL_DIR / (video.stem + '.json')
        llc_to_class(video, llc, output, False)
        print(f'Timestamp label saved to {output.absolute()}')
