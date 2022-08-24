from yacs.config import CfgNode as CN


_C = CN()

_C.DATA_ROOT = 'data'

_C.DATASET = CN()  # Raw dataset

_C.DATASET.PGDS2_DIR = 'police_gesture_v2'  # Root folder of pgdv2
_C.DATASET.VIDEO_DIR = 'video'  # '.m4v' videos
_C.DATASET.GESTURE_LLC_DIR = 'label_gesture_timestamp'  # '-proj.llc' gesture start and end timestamps annotated with losslesscut
_C.DATASET.ORIENTATION_LLC_DIR = 'label_orientation_timestamp'  # losslesscut annotated human body orientation label

# Videos that dataloader loads for training (to speed up, the preprocessed skeleton sequence is loaded instead)
_C.DATASET.TRAIN_VIDEOS = ['4K9A0217', '4K9A0218', '4K9A0219', '4K9A0220', '4K9A0221', '5566', '5568', '5570']
# Videos that dataloader loads for testing
_C.DATASET.TEST_VIDEOS = ['4K9A0222', '4K9A0223', '4K9A0224', '4K9A0226', '4K9A0227', '5571', '5573']
# _C.DATASET.VIDEOS = ['0000']  # Name of videos that dataloader will load 
_C.DATASET.MODE = "TRAIN"  # TRAIN / TEST, this decides which set of videos (TRAIN_VIDEOS / TEST_VIDEOS) are used.
_C.DATASET.GROUND_TRUTH = "33"  # options: "9" or "33". "9": 8 gestures + 1 no gesture; "33": 8 gestures, 4 directions, + 1 no gesture

_C.GENDATA = CN()  # Preprocessed generated data

_C.GENDATA.GES_LABEL_DIR = 'label_gesture_frame'  # '.json' labels. essentially a list of gesture class: gesture_cls_list[frames]
_C.GENDATA.ORI_LABEL_DIR = 'label_ori_frame'  # per-frame orientation labels
_C.GENDATA.COMBINE_LABEL_DIR = 'label_combine_frame'
_C.GENDATA.IMG_DIR = 'images'  # '.jpg' images generated from each video.
_C.GENDATA.TRACK_DIR = 'track_mul'  # '.npy' raw multi-person track result
_C.GENDATA.TRACE_SINGLE_DIR = 'track_single'  # '.npy' person track of the police, generated with 1d non-maximum-suppression from raw tracks
_C.GENDATA.VIBE_DIR = 'vibe'  # '.npy' vibe parameters of **TRACKED** frames.

_C.MODEL = CN()  # Network configs and save paths

_C.MODEL.CKPT_DIR = 'checkpoints'  # Checkpoint folder
_C.MODEL.MTPGR_CKPT = 'mtpgr_cam.ckpt'  # MTPGR ckpt file
_C.MODEL.DEVICE = 'cuda'  # 'cpu'  # 'cuda'
_C.MODEL.USE_CAMERA_POSE = True  # Add camera pose to joint features
_C.MODEL.CLIP_LEN = 300  # Length of video sample for graph network
_C.MODEL.BATCH_SIZE = 10  # Num of clips in one batch
_C.MODEL.NO_SPATIAL_EDGES = False  # No spatial edges on the graph, for ablation study.

_C.OUTPUT = 'output.pkl'  # Test result output

# _C.TRAIN.CLIP_LEN = 300
# _C.TRAIN.BATCH_SIZE = 10
# Training set videos
# _C.TRAIN.SET =

# _C.VAL = CN()
# _C.VAL.CLIP_LEN = 150  # The length of the sliding window.
# Eval set videos
# _C.VAL.SET = ['4K9A0222', '4K9A0223', '4K9A0224', '4K9A0226', '4K9A0227',
#                '5571', '5573']

def get_cfg_defaults():
    return _C.clone()
