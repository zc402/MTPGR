import cv2
from pathlib import Path


def video2images(video: Path):
    img_folder = video.with_suffix('.images')
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
        frame_num = frame_num + 1


if __name__ == '__main__':
    videos = Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/').glob('*.mp4')
    for video in videos:
        print(f'Processing: "{video}"')
        llc = video.parent / (video.stem + '-proj.llc')
        video2images(video)
