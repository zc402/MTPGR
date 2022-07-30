import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1)[:, np.newaxis]

result_path = Path('output') / 'j14_nocam_cls8' / 'result.pkl'

with result_path.open('rb') as f:
    res = pickle.load(f);    
score_TC = res['pred_TC']
score_TC = np.array(score_TC)

score_TC = score_TC[2000:2100]
score_TC = softmax(score_TC)

fig = plt.figure()
ax = plt.axes()
for k in range(score_TC.shape[1]):
    ys = score_TC[:, k]
    ax.plot(range(2000, 2100), ys)

# ax.set_xlabel('frame')
# ax.set_ylabel('probability')
plt.show()












