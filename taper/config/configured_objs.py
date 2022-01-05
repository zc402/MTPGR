"""
Configured objects for TAPER.
"""
from pathlib import Path
from torch.utils.data import DataLoader

from .config import get_cfg_defaults
from taper.dataset import SingleVideo, ConcatVideo
from taper.kinematic import dense_indices

cfg = get_cfg_defaults()

# ---------- Training dataset ----------
_vibe_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.VIBE_DIR
_vibe_list = _vibe_folder.glob('*.npy')
_label_list = [Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.LABEL_DIR / (name + '.json5') for name in _vibe_list]

_video_dataset_list = [SingleVideo(v, l, dense_indices) for v, l in zip(_vibe_list, _label_list)]
_concat_dataset = ConcatVideo(_video_dataset_list, cfg.TRAIN.CLIP_LEN)
train_loader = DataLoader(_concat_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)

#