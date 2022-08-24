import pickle
from pathlib import Path
from vibe.rt import RtVibe
import cv2
import logging
from mtpgr.config import get_cfg_defaults
import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

class RtVibeExtTrace(RtVibe):
    """
    A real-time VIBE class that read external trace.
    """
    @torch.no_grad()
    def from_img_and_trace(self, image: np.ndarray, track_res: dict):
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


def convert_trace_to_vibe(video_file: Path,
                     trace_file: Path,
                     save_path: Path,
                     render: bool = True):

    assert video_file.is_file()
    assert trace_file.is_file()

    with trace_file.open('rb') as f:
        track_res = pickle.load(f)

    cap = cv2.VideoCapture(str(video_file))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vibe_results = []
    vibe = RtVibeExtTrace(render=render)
    current_trace_frame = 0  # Trace has fewer frames than video, because sometimes the police is occludded and not tracked.
    for i in tqdm(range(frame_count)):
        # i: current video frame
        ret, image = cap.read()
        if not ret:
            logger.warning(f"cv2 video capture returns false at frame {i}")
        if i not in track_res['frames']:
            logger.debug(f"Frame {i} skipped. Reason: no frame {i} in trace data")
        else:
            # trace_wrap is used to suit the vibe input format

            trace_wrap = {1: {  # num 1 is the person id, set in "find_police_trace". 
                'bbox': track_res['bbox'][current_trace_frame:current_trace_frame+1],  # No dimension reduction
                'frames': track_res['frames'][current_trace_frame:current_trace_frame+1]
            }}
            vibe_output = vibe.from_img_and_trace(image, trace_wrap)
            del vibe_output[1]['verts']  # The vertices are too large. Currently not used.
            vibe_results.append(vibe_output)
            current_trace_frame = current_trace_frame + 1  # Only increases if the frame was tracked.

    cap.release()

    with save_path.open('wb') as f:
        pickle.dump(vibe_results, f)
    pass


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    assert Path(cfg.DATA_ROOT).is_dir(), 'MTPGR/data not found. Check current working directory, expect "./MTPGR"'
    logger.info("Running VIBE on videos")

    trace_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.TRACE_SINGLE_DIR
    assert trace_folder.is_dir()

    videos = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    videos = videos.glob('*.m4v')

    vibe_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.VIBE_DIR
    vibe_folder.mkdir(exist_ok=True)
    for video in videos:
        trace_file = trace_folder / (video.stem + '.pkl')
        target_path = vibe_folder / (video.stem + '.pkl')
        if target_path.is_file():
            logger.info(f'Skipping {video.stem}. Target already exists.')
        else:
            logger.info(f'Processing {video.stem}...')
            convert_trace_to_vibe(video, trace_file, target_path, render=False)


