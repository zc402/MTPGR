from pathlib import Path
data_folder = Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/')
videos = data_folder.glob('*.mp4')
tracks = data_folder.glob('*.track')
