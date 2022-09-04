from pathlib import Path
from mtpgr.config.defaults import get_cfg_defaults
import cv2

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/no_camera.yaml')

anime_save_folder = Path('output', cfg.OUTPUT, 'anime')
images = anime_save_folder.glob("*.jpg")
images = sorted(images, key=lambda img: int(img.stem))

frameSize = (int(3000), int(2000))
save_path = anime_save_folder / 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(save_path), cv2.CAP_FFMPEG, fourcc, 60, frameSize)

for img in images:
    img = cv2.imread(str(img))
    out.write(img)