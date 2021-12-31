import pickle
from pathlib import Path
from vibe.rt import RtVibe
import cv2

from taper.config import get_cfg_defaults
from taper.dataset.path import train_videos
import torch
import numpy as np
from tqdm import tqdm


class VibeInputTrack(RtVibe):
    @torch.no_grad()
    def __call__(self, image: np.ndarray, track_res: dict):
        # Tracking
        tracking_results = self._track(track_res)
        # Run VIBE inference
        vibe_results = self._vibe_infer(image, tracking_results)
        # Render
        if self.render:
            self._render_and_show(image, vibe_results)

        return vibe_results

    def _track(self, track_res: dict):
        # Keep hidden state for each individual.
        self.hidden_states.update_person_id(track_res)
        return track_res


def tracking_to_vibe(image_folder: Path,
                     track_file: Path,
                     save_path: Path):

    assert image_folder.is_dir()
    assert track_file.is_file()

    with track_file.open('rb') as f:
        track_res = pickle.load(f)

    vibe = VibeInputTrack()
    vibe.render = False

    images = image_folder.glob('*.jpg')
    # {frame_num: image_path}
    frame_image_dict = {int(image.stem): image for image in images}

    vibe_results = []
    for i in tqdm(range(len(track_res[1]['frames']))):
        single_track_res = {1: {
            'bbox': track_res[1]['bbox'][i:i+1],  # No dimension reduction
            'frames': track_res[1]['frames'][i:i+1]
        }}
        frame_num = track_res[1]['frames'][i]
        image_path = frame_image_dict[frame_num]
        image = cv2.imread(str(image_path))
        v_res = vibe(image, single_track_res)
        del v_res[1]['verts']  # Too large and no usage as dataset label.
        vibe_results.append(v_res)
    # save_path = track_file.parent / (track_file.stem + '.vibe')
    with save_path.open('wb') as f:
        pickle.dump(vibe_results, f)
    pass


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    assert Path(cfg.DATA_ROOT).is_dir(), 'TAPER/data not found, check working directory. (./TAPER expected) '
    track_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.TRACK_DIR
    tracks = track_folder.glob('*')
    img_root = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.IMG_DIR
    tk_crct_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.TK_CRCT_DIR
    tk_crct_folder.mkdir(exist_ok=True)
    for track in tracks:
        img_folder = img_root / track.stem
        save_path = tk_crct_folder / (track.stem + '.npy')
        tracking_to_vibe(img_folder, track, save_path)


# tracking_to_vibe(Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/4K9A0217.images'),
#                  Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/4K9A0217.track_correct'))
