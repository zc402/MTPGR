
import itertools
from pathlib import Path
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader

from mtpgr.config.defaults import get_cfg_defaults
import pickle

from mtpgr.new_features.vibe_frame_dataset import VibeFrameDataset

def sequence_to_isolated_gestures(seq):
    combine_label = seq["combine"]
    idx_a = 0
    clips = []
    for label, group in itertools.groupby(combine_label.numpy().tolist()):
        idx_b = idx_a + len(list(group))
        # Slice
        clip_label = {}
        for key, val in seq.items():
            val_clip = val[idx_a: idx_b]
            clip_label[key] = val_clip
        clips.append(clip_label)
        idx_a = idx_b
    return clips
    pass

def get_isolated_gestures(file_names):

    labels = []
    for name in tqdm(file_names):
        seq_dataset = VibeFrameDataset.from_config(_cfg)(name)
        seq_loader = DataLoader(seq_dataset, batch_size=len(seq_dataset), shuffle=False)
        seq_data = list(seq_loader)[0]  # only 1 batch in entire sequence
        seq_labels = sequence_to_isolated_gestures(seq_data)
        labels.extend(seq_labels)
    return labels

_cfg = get_cfg_defaults()
train_names = _cfg.DATASET.TRAIN_VIDEOS
test_names = _cfg.DATASET.TEST_VIDEOS

train_labels = get_isolated_gestures(train_names)
test_labels = get_isolated_gestures(test_names)
res_dict = {'train': train_labels, 'test': test_labels}

save_folder = Path(_cfg.DATA_ROOT, _cfg.DATASET.PGDS2_DIR, _cfg.GENDATA.ISO_GESTURE_LABEL_DIR)

for name_of_set in res_dict.keys():
    labels = res_dict[name_of_set]
    # save_folder = Path(save_folder, name_of_set)
    # shutil.rmtree(save_folder)
    for i, clip_label in enumerate(tqdm(labels)):
        save_path = Path(save_folder, name_of_set, f'{i}.pkl')
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open('wb') as f:
            pickle.dump(clip_label, f)
