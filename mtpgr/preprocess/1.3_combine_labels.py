# Combine gesture labels and orientation labels
# Gesture 0 (stand in attention) is the same in all 4 directions

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.utils.log import log

def combine(ges_label: Path, ori_label: Path, output: Path, show=True):
    with open(ges_label, 'r') as f:
        ges_json = json.load(f)
    with open(ori_label, 'r') as f:
        ori_json = json.load(f)
    # print(len(ges_json))
    # print(len(ori_json))
    assert len(ges_json) == len(ori_json), "2 label files must have same length"

    combine_list = list()
    new_byte: int = None  # Combined label for 1 frame
    for i in tqdm(range(len(ges_json))):
        ges = ges_json[i]
        ori = ori_json[i]
        if ges == 0:
            new_byte = 0  # No mater which direction the police is facing, no gesture class is the same.
        elif ges > 0 and ges <= 8:
            if ori == 'F':
                new_byte = ges + 0
            elif ori == 'L':
                new_byte = ges + 8
            elif ori == 'B':
                new_byte = ges + 16
            elif ori == 'R':
                new_byte = ges + 24
            else:
                raise NotImplementedError(f"Unexcepted orientation class {ori}")
            pass
        else:
            raise NotImplementedError(f"Unexpected gesture class {ges}")
        combine_list.append(new_byte)
    
    with open(output, 'w') as f:
        json.dump(combine_list, f)
        

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    assert Path(cfg.DATA_ROOT).is_dir(), 'MTPGR/data not found. Expecting "./MTPGR" as working directory'
    ges_folder: Path = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.GES_LABEL_DIR
    ori_folder: Path = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.ORI_LABEL_DIR
    combine_folder: Path = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.COMBINE_LABEL_DIR
    combine_folder.mkdir(exist_ok=True)
    video_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    video_paths = video_folder.glob('*.m4v')
    video_paths = list(video_paths)

    for video in video_paths:
        video_name = video.stem
        ges_file: Path = ges_folder / (video_name + '.json')
        assert ges_file.is_file()
        ori_file: Path = ori_folder / (video_name + '.json')
        assert ori_file.is_file()
        combine_file: Path = combine_folder / (video_name + '.json')
        log.info(f"Processing labels of video: {video_name}")
        combine(ges_file, ori_file, combine_file, show=True)
    log.info("Completed: combine_labels")