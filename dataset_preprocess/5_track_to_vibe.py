import pickle
from pathlib import Path
from vibe.rt import RtVibe
import cv2

def tracking_to_vibe(image_folder: Path, track_file: Path):

    assert image_folder.is_dir()
    assert track_file.is_file()

    with track_file.open('rb') as f:
        tracking_results = pickle.load(f)

    rt_vibe = RtVibe()
    rt_vibe.render = False

    images = image_folder.glob('*.jpg')
    for img in sorted(images):
        img_cv2 = cv2.imread(str(img))
        result = rt_vibe(img_cv2)

    pass

tracking_to_vibe(Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/4K9A0217.images'), Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/4K9A0217.track'))