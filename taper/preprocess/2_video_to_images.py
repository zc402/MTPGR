import cv2
from pathlib import Path
from taper.config import get_cfg_defaults


def video2images(video: Path, img_folder: Path):
    """
    Split the video into frame images
    :param video:
    :param img_folder: image folder of the video with same name.
    :return:
    """
    img_folder.mkdir(exist_ok=True)
    frame_num = 0
    cap = cv2.VideoCapture(str(video))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_path = img_folder / str(frame_num).zfill(6)
        img_path = img_path.with_suffix(".jpg")  # 000005.jpg
        cv2.imwrite(str(img_path), frame)
        assert img_path.is_file()  # when failed, cv2.imwrite do not raise exception.
        frame_num = frame_num + 1


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    assert Path(cfg.DATA_ROOT).is_dir(), 'TAPER/data not found, check working directory (./TAPER expected) '
    videos = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    videos = videos.glob('*.mp4')
    # img_root: The folder containing '.images' folders
    img_root = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.IMG_DIR
    img_root.mkdir(exist_ok=True)

    for video in videos:
        img_folder = img_root / video.stem
        print(f'Processing: "{video.absolute()}"')
        video2images(video, img_folder)
