from inspect import currentframe
import pickle
import cv2  # unused, but must be imported, otherwise the MPT raises segment fault
from multi_person_tracker import MPT
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.utils.log import log
from multi_person_tracker import Sort
from torchvision.transforms.functional import to_tensor

class VideoFrameIterDataset(IterableDataset):
    """
    The frames in a video is accessed sequentially. Random access causes slow performance. Total length is known.
    """
    def __init__(self, video_path: Path):
        assert video_path.is_file()
        self.video_path = video_path
    
    def _frame_generator(self):
        cap = cv2.VideoCapture(str(self.video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break
            frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            current_frame = current_frame + 1
            if current_frame % 100 == 0:
                print(f"Processing frame {current_frame} / {frame_count}")
            yield to_tensor(frame_RGB)

    def __iter__(self):
        frame_generator = self._frame_generator()
        frame_gen_iter = iter(frame_generator)
        return frame_gen_iter

class MPT_Iterate(MPT):
    def run_on_video(self, video_path):
        image_dataset = VideoFrameIterDataset(video_path)

        dataloader = DataLoader(image_dataset, batch_size=self.batch_size, num_workers=0)

        trackers = self.run_tracker(dataloader)
        # if self.display:
        #     self.display_results(image_folder, trackers, output_file)

        if self.output_format == 'dict':
            result = self.prepare_output_tracks(trackers)
        elif self.output_format == 'list':
            result = trackers

        return result

    # Overwrite
    @torch.no_grad()
    def run_tracker(self, dataloader):
        '''
        Run tracker on an input video

        :param video (ndarray): input video tensor of shape NxHxWxC. Preferable use skvideo to read videos
        :return: trackers (ndarray): output tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        '''

        # initialize tracker
        self.tracker = Sort()

        print('Running Multi-Person-Tracker')
        trackers = []
        for batch in dataloader:
            batch = batch.to(self.device)

            predictions = self.detector(batch)

            for pred in predictions:
                bb = pred['boxes'].cpu().numpy()
                sc = pred['scores'].cpu().numpy()[..., None]
                dets = np.hstack([bb,sc])
                dets = dets[sc[:,0] > self.detection_threshold]

                # if nothing detected do not update the tracker
                if dets.shape[0] > 0:
                    track_bbs_ids = self.tracker.update(dets)
                else:
                    track_bbs_ids = np.empty((0, 5))
                trackers.append(track_bbs_ids)

        # fps = len(dataloader.dataset) / runtime
        # print(f'Finished. Detection + Tracking FPS {fps:.2f}')
        return trackers

def to_trace(video_path: Path, save_path: Path):
    assert video_path.is_file()

    if torch.cuda.is_available():
        log.debug("CUDA available")
        device = torch.device('cuda')
    else:
        log.warning("CUDA not available!")
        device = torch.device('cpu')

    mpt = MPT_Iterate(
        device=device,
        batch_size=5,
        display=False,
        detector_type='yolo',
        output_format='dict',
    )

    tracking_result = mpt.run_on_video(video_path)

    with save_path.open('wb') as f:
        pickle.dump(tracking_result, f)
    
    # log.debug(tracking_result)


if __name__ == '__main__':

    cfg = get_cfg_defaults()

    assert Path(cfg.DATA_ROOT).is_dir(), 'MTPGR/data not found. Expecting "./MTPGR" as working directory'

    videos = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    # videos = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / "debug"
    videos = videos.glob('*.m4v')

    target_folder: Path = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.TRACK_DIR
    target_folder.mkdir(exist_ok=True)
    for video in videos:
        log.info(f"Processing file {video.stem}")
        target_path = target_folder / (video.stem + ".pkl")
        if target_path.is_file():
            log.info(f'track file {video.stem} already exists. Ignored.')
        else:
            to_trace(video, target_path)