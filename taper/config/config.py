from yacs.config import CfgNode as CN


_C = CN()

_C.DATA_ROOT = 'data'

_C.DATASET = CN()  # Raw dataset

_C.DATASET.PGDS2_DIR = 'police_gesture_v2'  # Root folder of pgdv2
_C.DATASET.VIDEO_DIR = 'video'  # '.mp4' videos
_C.DATASET.LLC_DIR = 'label_timestamp'  # '-proj.llc' gesture start and end timestamps annotated with losslesscut

_C.GENDATA = CN()  # Preprocessed generated data

_C.GENDATA.LABEL_DIR = 'label_frame'  # '.json5' labels. essentially a list of gesture class: gesture_cls_list[frames]
_C.GENDATA.IMG_DIR = 'images'  # '.jpg' images generated from each video.
_C.GENDATA.TRACK_DIR = 'track_mul'  # '.npy' raw multi-person track result
_C.GENDATA.TK_CRCT_DIR = 'track_nms'  # '.npy' person track of the police, generated via 1d non-maximum-suppression from raw tracks
_C.GENDATA.VIBE = 'vibe'  # '.npy' vibe parameters of each **TRACKED** frames.


def get_cfg_defaults():
    return _C.clone()
