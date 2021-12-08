import cv2
from pathlib import Path
import json5
import matplotlib.pyplot as plt
from tqdm import tqdm


def llc_to_class(video: Path, label: Path):
    """
    Convert losslesscut .llc format (json5) to class-per-frame format (json5)
    :param video:
    :param label:
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
        cap.set(cv2.CAP_PROP_POS_MSEC, seg['start']*1000)
        s_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_MSEC, seg['end']*1000)
        e_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        for f in range(int(s_frame), int(e_frame)):
            cls_list[f] = cls

    plt.plot(cls_list)
    plt.show()

    cls_path = video.with_suffix('.json5')
    with cls_path.open('w') as f:
        json5.dump(cls_list, f)
    pass


if __name__ == '__main__':
    videos = Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/').glob('*.mp4')
    for video in videos:
        print(f'Processing: "{video}"')
        llc = video.parent / (video.stem + '-proj.llc')
        llc_to_class(video, llc)
