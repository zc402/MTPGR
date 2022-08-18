# get jaccard score from dumped test result 

from pathlib import Path
import pickle
from sklearn.metrics import jaccard_score

from mtpgr.config.defaults import get_cfg_defaults

val_cfg = get_cfg_defaults()
val_cfg.merge_from_file(Path('configs', 'default_model.yaml'))

save_path = Path('output') / val_cfg.OUTPUT
with open(save_path, 'rb') as f:
    res = pickle.load(f)

pred = res['pred_T']
label = res['label_T']

js = jaccard_score(label, pred, average='micro')

print("Jaccard score:")
print(js)