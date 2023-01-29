from pathlib import Path
import pickle
import random
from torch.utils.data import Dataset
import glob
import numpy as np

class IsolatedDataset(Dataset):

    def __init__(self, cfg, name_of_set, sampling='random', do_augment=True) -> None:
        # train_names = cfg.DATASET.TRAIN_VIDEOS
        # test_names = cfg.DATASET.TEST_VIDEOS

        # if name_of_set == 'train':
        #     names = train_names
        # elif name_of_set == 'test':
        #     names = test_names
        
        label_folder = Path(cfg.DATA_ROOT, cfg.DATASET.PGDS2_DIR, cfg.GENDATA.ISO_GESTURE_LABEL_DIR, name_of_set)
        self.iso_pkl_path_list = glob.glob(str(label_folder / '*.pkl'))
        self.step_interval = cfg.MODEL.ISO_INTERVAL
        self.clip_len = cfg.MODEL.ISO_CLIP_LEN
        self.sampling = sampling
    
    def __len__(self):
        return len(self.iso_pkl_path_list)

    def _random_sampling(self, seq_len, clip_len):
        possible_start_idx = seq_len - clip_len
        possible_start_idx = max(0, possible_start_idx)
        start_idx = random.randint(0, possible_start_idx)  # (randint: start/end both included)
        clip_indices = range(start_idx, start_idx + clip_len, self.step_interval)
        clip_indices = [i % seq_len for i in clip_indices]  # If clip is larger than video length, then pick from start
        return clip_indices
    
    def _uniform_sampling(self, seq_len, clip_len):
        clips = []
        if(seq_len <= clip_len):
            clips.append(self._random_sampling(seq_len, clip_len))
        else:
            t = 0
            for t in range(0, seq_len - clip_len, clip_len):
                clip_indices = range(t, t + clip_len, self.step_interval)
                clips.append(clip_indices)
        return clips

    def _get_features_from_indices(self, features, indices):
        sample_features = {}
        for key, val in features.items():
            val_clip = np.array(val)[indices]
            sample_features[key] = val_clip
        return sample_features

    def __getitem__(self, index):
        iso_pkl_path = self.iso_pkl_path_list[index]
        with Path(iso_pkl_path).open('rb') as f:
            original_label = pickle.load(f)
        
        # Sample
        seq_len = len(original_label['combine'])
        if self.sampling == 'random':
            # Random sampling, take 1
            clip_indices = self._random_sampling(seq_len, self.clip_len)
            features = self._get_features_from_indices(original_label, clip_indices)
            return features

        elif self.sampling == "global":  # Return 1 entire video. Must use batchsize 1
            clip_indices = list(range(0, seq_len, self.step_interval))
            collected_feature = self._get_features_from_indices(original_label, clip_indices)
            return collected_feature

        return original_label


if __name__ == '__main__':

    from mtpgr.config.defaults import get_cfg_defaults
    import numpy as np
    from torch.utils.data import DataLoader

    _cfg = get_cfg_defaults()

    train_dataset = IsolatedDataset(_cfg, 'train')
    train_loader = DataLoader(train_dataset, 10, shuffle=True,)
    for batch in train_loader:
        print(batch)
    # dataset = IsolatedDataset(_cfg, 'train')
    # data = list(dataset)
    # length = [len(x['combine']) for x in data]
    # print(np.max(length), np.min(length), np.mean(length))  # Result: 578 17 81.46091644204851
