import cv2
from pathlib import Path
import json5
import matplotlib.pyplot as plt
from tqdm import tqdm
from taper.config import get_cfg_defaults


def llc_to_class(video: Path, label: Path, output: Path, show: bool = True):
    """
    Convert losslesscut .llc format (json5) to class-per-frame format (json5)
    :param video: path of a videos, used to get FRAME_COUNT
    :param label: path of .llc file corresponding to the video.
    :param output: path of output json5 per-frame label file
    :param show: display labels as a plot
    :return:
    """
    cap = cv2.VideoCapture(str(video))
    with label.open() as f:
        label_json = json5.load(f)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cls_list = [0 for i in range(duration)]
    default_num_ges_to_cls = list(range(1, 9)) * 8  # [1,2,...,8,1,2,...]
    for num_ges, seg in enumerate(tqdm(label_json['cutSegments'])):
        # seg: {'start': 1.0113, 'end': 3.4433}
        cls = default_num_ges_to_cls[num_ges]
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

    with output.open('w') as f:
        json5.dump(cls_list, f)
    pass


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    assert Path(cfg.DATA_ROOT).is_dir(), 'TAPER/data not found, check working directory (./TAPER expected) '
    (Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.LABEL_DIR).mkdir(exist_ok=True)
    videos = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    videos = videos.glob('*.mp4')
    for video in videos:
        print(f'Processing: "{video}"')
        llc = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.LLC_DIR / (video.stem + '-proj.llc')
        output = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.LABEL_DIR / (video.stem + '.json5')
        llc_to_class(video, llc, output, False)
        print(f'Timestamp label saved to {output.absolute()}')
