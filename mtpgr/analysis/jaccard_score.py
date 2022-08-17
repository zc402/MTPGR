# get jaccard score from dumped test result 

from pathlib import Path
import pickle
from sklearn.metrics import jaccard_score

save_path = Path('output') / 'j14_nocam_cls8' / 'result.pkl'
with open(save_path, 'rb') as f:
    res = pickle.load(f)

pred = res['pred_T']
label = res['label_T']

js = jaccard_score(label, pred, average='micro')

print(js)