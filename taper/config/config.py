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
_C.GENDATA.VIBE_DIR = 'vibe'  # '.npy' vibe parameters of **TRACKED** frames.

_C.MODEL = CN()  # Network configs and save paths

_C.MODEL.CKPT_DIR = 'checkpoints'  # Checkpoint folder
_C.MODEL.TAPER_CKPT = 'taper_cam.ckpt'  # TAPER ckpt file
_C.MODEL.DEVICE = 'cuda'  # 'cpu'  # 'cuda'
_C.MODEL.USE_CAM_POSE = True  # Add camera pose to joint features

_C.TRAIN = CN()  # Training

_C.TRAIN.CLIP_LEN = 300  # Length of video sample for training
_C.TRAIN.BATCH_SIZE = 10  # Num of clips in one batch
# Training set videos
_C.TRAIN.SET = ['4K9A0217', '4K9A0218', '4K9A0219', '4K9A0220', '4K9A0221',
                '5566', '5568', '5570']

_C.EVAL = CN()
_C.EVAL.CLIP_LEN = 150  # The length of the sliding window.
# Eval set videos
_C.EVAL.SET = ['4K9A0222', '4K9A0223', '4K9A0224', '4K9A0226', '4K9A0227',
               '5571', '5573']

def get_cfg_defaults():
    return _C.clone()
