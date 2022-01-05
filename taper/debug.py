import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from taper.dataset.concat_video import VideoConcat


loader = DataLoader(VideoConcat(), batch_size=50, drop_last=False)

for i in range(10):
    a = next(iter(loader))
    pass

# vibe_path = Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/') / Path('4K9A0217').with_suffix('.vibe')
# with vibe_path.open('rb') as f:
#     vibe = pickle.load(f)
# pass