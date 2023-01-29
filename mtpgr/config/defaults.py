from yacs.config import CfgNode as CN


_C = CN()

_C.DATA_ROOT = 'data'
_C.NUM_CPU = 16

_C.DATASET = CN()  # Raw dataset

_C.DATASET.PGDS2_DIR = 'police_gesture_v2'  # Root folder of pgdv2
_C.DATASET.VIDEO_DIR = 'video'  # '.m4v' videos
_C.DATASET.GESTURE_LLC_DIR = 'label_gesture_timestamp'  # '-proj.llc' gesture start and end timestamps annotated with losslesscut
_C.DATASET.ORIENTATION_LLC_DIR = 'label_orientation_timestamp'  # losslesscut annotated human body orientation label

# Videos that dataloader loads for training (to speed up, the preprocessed skeleton sequence is loaded instead)
_C.DATASET.TRAIN_VIDEOS = ['4K9A0217', '4K9A0218', '4K9A0219', '4K9A0220','4K9A0226', '4K9A0227', '5566', '5568', ]
# Videos that dataloader loads for testing
_C.DATASET.TEST_VIDEOS = ['4K9A0221', '4K9A0222', '4K9A0223', '4K9A0224', '5571', '5573', '5570']

_C.DATASET.NUM_CLASSES = 33  # options: 9 or 33. # 9: 8 gestures + 1 no gesture; # 33: 8 gestures, 4 directions, + 1 no gesture
_C.DATASET.EPOCHS = 200  # Max epochs for CTPGesture v2

_C.GENDATA = CN()  # Preprocessed generated data

_C.GENDATA.GES_LABEL_DIR = 'label_gesture_frame'  # '.json' labels. essentially a list of gesture class: gesture_cls_list[frames]
_C.GENDATA.ORI_LABEL_DIR = 'label_ori_frame'  # per-frame orientation labels
_C.GENDATA.COMBINE_LABEL_DIR = 'label_combine_frame'
_C.GENDATA.IMG_DIR = 'images'  # '.jpg' images generated from each video.
_C.GENDATA.TRACK_DIR = 'track_mul'  # '.npy' raw multi-person track result
_C.GENDATA.TRACE_SINGLE_DIR = 'track_single'  # '.npy' person track of the police, generated with 1d non-maximum-suppression from raw tracks
_C.GENDATA.VIBE_DIR = 'vibe'  # '.npy' vibe parameters of **TRACKED** frames.
_C.GENDATA.ISO_GESTURE_LABEL_DIR = 'iso_gesture_label_dir'  # Isolated gestures and labels.

_C.MODEL = CN()  # Network configs and save paths

_C.MODEL.DEBUG = True
_C.MODEL.BATCH_SIZE = 200

_C.MODEL.NAME = "auto"  # The checkpoint and output folder name. auto = GRAPH_STRATEGY_FUSE_NUMCLASSES
_C.MODEL.CKPT_DIR = 'checkpoints'  # Checkpoint folder
# _C.MODEL.MTPGR_CKPT = 'mtpgr_cam.ckpt'  # MTPGR ckpt file
_C.MODEL.DEVICE = 'cuda'  # 'cpu'  # 'cuda'
_C.MODEL.ATTENTION = True # Use attention in GCN
# _C.MODEL.USE_CAMERA_POSE = True  # Root rotation 
# _C.MODEL.USE_ROTATIONS = True  # All joint rotations (Include root rotation)
_C.MODEL.GRAPH = 'PRCB'  # C: camera, P: position, R: rotation. Allow: 'P', 'R', 'PR', 'CP', 'CPR'
_C.MODEL.GCN_DEPTH = 10  # 4 or 10. The depth of the GCN network
_C.MODEL.CLIP_LEN = 300  # Length of video sample for graph network
_C.MODEL.BATCH_SIZE = 10  # Num of clips in one batch
# _C.MODEL.NO_SPATIAL_EDGES = False  # No spatial edges on the graph, for ablation study.
_C.MODEL.STRATEGY = "RHPS"  # "RHPS" / "SCPS"
_C.MODEL.FUSE = "mean"  # mean / sparse  # deprecated, use mean. sparse fuse in this code is not what it should be.

_C.MODEL.ISO_CLIP_LEN = 120
_C.MODEL.ISO_INTERVAL = 1  # Sample interval
_C.MODEL.LEARNING_RATE = 5e-4

def get_cfg_defaults():
    return _C.clone()

def get_auto_name(cfg):
    name = f"{cfg.MODEL.GRAPH}_{cfg.MODEL.STRATEGY}_{cfg.MODEL.FUSE}_{cfg.DATASET.NUM_CLASSES}"
    return name